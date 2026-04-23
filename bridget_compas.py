#!/usr/bin/env python
# coding: utf-8

# # BRIDGET: compas test
# 

# 
# ## Dataset Preprocessing
# 

# ### Libraries, Retrieving data

# In[1]:
# In[2]:


import os
import yaml
import joblib

import pickle

import copy


import pandas as pd



from river import tree

from river import preprocessing
from river import ensemble, forest, compose


from sklearn.preprocessing import MinMaxScaler
from bridget_utils import *
from orchestrator import *
from master_config import *
from classes import BetaUser
from bridget_main import BRIDGET, HiC, MiC

"""
from baselines.compare_confidence import *
from baselines.differentiable_triage import *
from baselines.lce_surrogate import *
from baselines.mix_of_exps import *
from baselines.one_v_all import *
from baselines.selective_prediction import *
"""


# In[3]:

if __name__ == "__main__":
     
    set_all_seeds(42)


    # In[4]:


    ds_name= 'compas'


    # In[5]:


    data= pd.read_csv(fr"./datasets/{ds_name}.csv")
    data.info()


    # In[6]:


    for c in data:
        print(data[c].value_counts().sum)


    # ### Preprocessing Pipeline
    # 1. Drop duplicates
    # 
    # 2. Map 'sex', 'race', 'c_charge_degree'
    # 

    # In[7]:


    data= clean_compas(data, 'compas_score', col_to_strip= 'c_charge_degree', drop_duplicates=True)


    # In[8]:


    data.info()
    data.head(n=5)


    # In[9]:


    mapping= { 'sex': {'Female':0, 'Male':1},

                'race': {'Native American': 0,
                'Asian': 1,
                'Other': 2,
                'Hispanic': 3,
                'Caucasian': 4,
                'African-American': 5  },

                'c_charge_degree': {
                    'X': 0, 'TCX': 0, 'NI0': 0, 'MO3': 0, 'CO3': 0, 

                    'M2': 1,   
                    'M1': 2,  
                    'F5': 3, 'F6': 3,'F3': 3,                    

                    'F7': 4, 'F2': 4, # apparently lvl 7 felonies can get u up to 15 years in prison                                                  
                    'F1': 5
                    }

    }

    data= apply_map(data, mapping)


    # ### Splitting and Transforming data
    # 
    # 1. Apply stratified sampling
    # 
    # 2. Get pre-training/HiC/calibration/Mic data
    # 
    # 3. Apply scaler
    # 
    # 4. Get X, y

    # In[10]:


    # Qui definiamo i vari split dei flussi 
    #set_all_seeds(42)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle iniziale

    class_0 = data[data['did_recid'] == 0]
    class_1= data[data['did_recid'] == 1]

    #  split ufficiale

    splits= {
        'calibration': (0.6, 0.8),
        'mic': (0.8, 1.0),
        'avv_train': (0.0, 0.07),
        'avv_test': (0.07, 0.1),
        'hic_train': (0.1, 0.5),
        'hic_test': (0.5, 0.6)
    }

    dfs= {}

    for name, (start, end) in splits.items():
        dfs[name]= stratif(start, end, class_0, class_1)



    # In[11]:


    for name, df in dfs.items():
        print(f"{name} length: {len(df)}")


    # In[12]:


    target= 'did_recid'

    categoricals= ['sex', 'race', 'c_charge_degree']
    numericals= [c for c in data if c not in categoricals and c != target]


    prepr_transf = (
        (compose.Select(*numericals, *categoricals) | preprocessing.MinMaxScaler())
    )


    # In[13]:


    ## ora divisione in x e y

    #set_all_seeds(42)

    # avviamento 
    X_avv_train, y_avv_train = x_y_split(dfs['avv_train'], target)
    X_avv_test, y_avv_test = x_y_split(dfs['avv_test'], target)


    # hic
    X_hic_train, y_hic_train = x_y_split(dfs['hic_train'], target)
    X_hic_test, y_hic_test = x_y_split(dfs['hic_test'], target)

    # validation
    X_val, y_val = x_y_split(dfs['calibration'], target)

    # mic
    X_mic, y_mic = x_y_split(dfs['mic'], target)



    # ## Calibration Phase: Experts and Incremental Model Selection

    # ### Calibrating Incremental Model
    # 
    # The incremental model to be chosen for Bridget is trained on the X_avv, y_avv portion of the dataset,then evaluated on the X_avv_test and y_avv_test
    # 
    # The calibration phase starts by assessing the results of the learning for several configurations:
    # 
    #     - HoeffdingTreeClassifier
    # 
    #     - ExtremelyFastDecisionTreeClassifier
    # 
    #     - AdaBoostClassifier            (base= SGTClassifier)
    # 
    #     - AdwinBaggingClassifier        (base= SGTClassifier)
    # 
    #     - SRPClassifier                 (base= SGTClassifier)
    # 
    #     - AdaptiveRandomForestClassifier
    # 
    # 
    # The metrics observed are the Accuracy, the F1Score and the Counters for the classes

    # In[14]:


    # since all River models work with dicts, lets first transform the dfs to dict
    #set_all_seeds(42)
    X_avv_dict= X_avv_train.to_dict(orient='records')
    X_avv_dict_test= X_avv_test.to_dict(orient='records')

    df_batch_1 = (dfs['hic_train']).reset_index(drop=True)
    mic_data= dfs['mic'].reset_index(drop=True)
    df_avv= pd.concat([dfs['avv_train'], dfs['avv_test']]).reset_index(drop=True)

    test_batch_1= X_hic_test.copy()
    test_batch_1[target]= y_hic_test


    # setting the init params required by HIC class
    RULE = False
    PAST = True
    SKEPT = True
    GROUP = True
    EVA=    True
    N_BINS = 10
    N_VAR = 3
    MAX = 5

    rule_att = 'priors_count' #random rule
    rule_value = -0.7

    protected= ['race', 'age', 'sex']


    # In[15]:


    # then the models are instantiated and trained by the HiC.train function
    # the HiC object is initialized by passing a random user model, its not relevant since it won't interact with the IL anyways

    #set_all_seeds(42)

    base = tree.HoeffdingTreeClassifier(grace_period=100)

    htree= tree.HoeffdingAdaptiveTreeClassifier(grace_period= 100, seed= 42)
    efdt= tree.ExtremelyFastDecisionTreeClassifier(grace_period=100)
    ada= ensemble.AdaBoostClassifier(model= base, n_models= 15, seed= 42)  
    adwin= ensemble.ADWINBaggingClassifier(model= base, n_models= 15, seed= 42)
    srp= ensemble.SRPClassifier(model= base, n_models=15, seed= 42)
    arf= forest.ARFClassifier(n_models= 15, grace_period= 100, max_features='sqrt', seed=42)

    models= {
        'HoeffdingATC': htree, 
        "EFDT": efdt, 
        "AdaBoostCl":ada, 
        "ADWINBAGGING": adwin, 
        "SRPCL": srp, 
        "ARF":arf   
        }

    prepr_dir= DATASETS[ds_name]['base_obj_paths']['preprocessors']
    os.makedirs(prepr_dir, exist_ok=True)

    model_dir= DATASETS[ds_name]['base_obj_paths']['incremental_learners']
    os.makedirs(model_dir, exist_ok=True)

    for model_name, model_obj in models.items():

        bridget_inst= HiC(RULE, PAST, SKEPT, GROUP, EVA, N_BINS, N_VAR, MAX, 
                    rule_att, rule_value, 'placeholder', model_obj,
                    start_performance= DATASETS[ds_name]['start_performance'],
                    allocated_budget= DATASETS[ds_name]['allocated_budget'][0],
                    skepticism_threshold= DATASETS[ds_name]['skepticism_threshold'],
                    performance_delta= DATASETS[ds_name]['performance_delta'],
                    dataset_name= ds_name,
                    user_name= 'placeholder',
                    batch1=df_batch_1, batch3=mic_data, batch1_test=test_batch_1, 
                    target=target, 
                    user_model='placeholder', 
                    protected=protected, cats=categoricals, num=numericals,
                    preprocessor=prepr_transf,
                    training_iter= 0 
                    )

        bridget_inst.train(X_avv_dict, y_avv_train, X_avv_dict_test, y_avv_test)

        base_preprocessor = bridget_inst.preprocessor
        base_model = bridget_inst.hic_model


        #prepr_filename = f"{model_name}_preprocessor.pkl"
        #model_filename = f"{model_name}_model.pkl"

        #joblib.dump(base_preprocessor, os.path.join(prepr_dir, prepr_filename))
        #joblib.dump(base_model, os.path.join(model_dir, model_filename))


    trained_arf= arf


    # ### Calibrating Experts
    # 

    # In[16]:


    with open(fr"./experts_{ds_name}.yaml", "r") as f:
        config= yaml.safe_load(f)


    params_dict= config['experts']['groups']['w_dict']


    # In[17]:


    #set_all_seeds(42)
    X_exp= X_hic_train.to_dict(orient='records')

    X_exp_scaled= []

    for x in X_exp:
        X_exp_scaled.append(prepr_transf.transform_one(x))

    X_exp_final = pd.DataFrame(X_exp_scaled)


    # In[18]:


    #set_all_seeds(42)
    experts_obj= {}

    expert_names = ['accurate_trusting', 'accurate_not_trusting', 
                    'inaccurate_trusting', 'inaccurate_not_trusting']

    for name in expert_names:
        expert_type= config['experts']['groups'][name]

        experts_obj[name]= BetaUser(
            belief_level= expert_type['belief_value'],
            rethink_level= 0.8, # as suggested by the og FRANK implementation
            fairness= True,
            fpr= expert_type['target_FPR'],
            fnr= expert_type['target_FNR'],
            alpha= 0.9,
            features_dict= params_dict,
            seed= expert_type['group_seed']
            )
        res = experts_obj[name].fit(X_exp_final, y_hic_train, tol= 0.001)

        save_dir= fr"./trained_experts/{ds_name}"
        os.makedirs(save_dir, exist_ok= True)
        file_path = os.path.join(save_dir, f"{name}.pkl")

        with open(file_path, 'wb') as f:
            pickle.dump(experts_obj[name], f)
        print(f"Expert saved in: {file_path}")


        print(f"{'='*30}")
        print(f" EXPERT CALIBRATION REPORT ")
        print(f"{'='*30}")

        print(f"\n[EXPERT: {name}]")
        print(f"\n[FALSE POSITIVE RATE]")
        print(f"  - Iters:      {res['fpr iters number']}")
        print(f"  - Beta:       {res['calibrated_fpr_beta']:.4f}")
        print(f"  - Target:     {res['target_fpr']}")
        print(f"  - Achieved:   {res['achieved_fpr']:.4f}")

        print(f"\n[FALSE NEGATIVE RATE]")
        print(f"  - Iters:      {res['fnr iters number']}")
        print(f"  - Beta:       {res['calibrated_fnr_beta']:.4f}")
        print(f"  - Target:     {res['target_fnr']}")
        print(f"  - Achieved:   {res['achieved_fnr']:.4f}")



    # ## BRIDGET decision making
    # 
    # 

    # In[19]:


    # base params for all users
    initial_ordering= [c for c in data if c != target]

    # retrieve preprocessor and incremental learner
    rules= FRANK_RULES


    base_params= {
        "cats": categoricals, #lst
        "num": numericals, #lss
        "warm_up_set": df_avv.copy(),
        "batch1": df_batch_1.copy(),
        "batch1_test": test_batch_1.copy(),
        "batch3":dfs['mic'].copy(),
        "validation_set": dfs['calibration'].copy(), #obtained by hic
        "feature_order": initial_ordering.copy(), #without the label ?
        "incremental_learner_name":"ARF",
        "n_cols": len(initial_ordering), #obtained by hic
        "mic_model_name": "Def_Net",
        "strat_1_name": "Confidence",
        "strat_2_name": "Mao",

    }


    learner_path= DATASETS[ds_name]['base_obj_paths']['incremental_learners']
    path= os.path.join(learner_path, f"{base_params["incremental_learner_name"]}_model.pkl")
    trained_learner= joblib.load(path)

    prepr_path= DATASETS[ds_name]['base_obj_paths']['preprocessors']
    path= os.path.join(prepr_path, f"{base_params["incremental_learner_name"]}_preprocessor.pkl")
    base_prepr= joblib.load(path)

    base_objects = {
        "preprocessor": copy.deepcopy(base_prepr), #or take from path
        "incremental_learner":copy.deepcopy(trained_learner), # or take from path
        "scaler": None
    } 

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device= torch.device("cpu")
    #add these once you get to mic phase: "run_confidence": True, "run_mao": True


    # ### Expert: Accurate, Trusting 

    # #### HIC

    # In[20]:


    #setting up fixed params

    current_user_name= "accurate_trusting"
    user_suffix= 'acc_t'

    exp_path= fr"./trained_experts/{ds_name}/{current_user_name}.pkl"
    current_expert= joblib.load(exp_path)


    params = base_params.copy()  # Start with the base
    params.update({
        "user_name": current_user_name,
        "user_suffix": user_suffix
    })

    objects = base_objects.copy() # Start with the base clones
    objects.update({
        "user_model": current_expert
    })


    # In[21]:


    #run_hic(ds_name, params, objects)

    
    # #### CALIB

    # In[ ]:

    
    #chosen nets: "large", 32/16

    net_layers= [32,16]
    hic_df_general_path= DATASETS[ds_name]['paths']['hic_df_save_path']
    net_general_path= DATASETS[ds_name]['paths']['def_net_save_path']

    # retrieving df and best net path
    for i in range (1, 4):

        df_switch_path= os.path.join(hic_df_general_path,
                                f"iter_{i}",
                                f"results_{params['user_name']}",
                                f"hic_{params['user_name']}.parquet"
                                )

        df_switch= pd.read_parquet(df_switch_path)


        def_net_path= os.path.join(net_general_path,
                            f"iter_{i}",
                            f"{user_suffix}_models",
                            f"{net_layers[0]}_{net_layers[1]}"
                            )

        pattern = f"{net_layers[0]}_{net_layers[1]}_model_*.pt"
        search_query = os.path.join(def_net_path, pattern)
        matches = glob.glob(search_query)

        if matches:
            def_net_path = max(matches, key=os.path.getctime)
            run_calibration(def_net_path= def_net_path, 
                            df_switch=df_switch, 
                            ds_name=ds_name, 
                            params=params,
                            device=device, 
                            iteration=i,
                            def_net_layers=net_layers
                            ) #default had batch seize 128, epochs = 20 


    # #### MIC

    # In[ ]:


    current_user_name= "accurate_trusting"
    user_suffix= 'acc_t'

    exp_path= fr"./trained_experts/{ds_name}/{current_user_name}.pkl"
    current_expert= joblib.load(exp_path)


    params = base_params.copy()  # Start with the base
    params.update({
        "user_name": current_user_name,
        "user_suffix": user_suffix,
        "run_confidence": True,
        "run_mao": True
    })

    objects = base_objects.copy() # Start with the base clones
    objects.update({
        "user_model": current_expert
    })


    def_net_layers= [32, 16]
    strat_1=None
    strat_2=None

    for i in range(1,4):

        strat_1, strat_2= run_mic( 
                                ds_name= ds_name,
                                device= device,
                                layers= def_net_layers,
                                params=params,
                                objects=objects,
                                iteration=i,
                                strat_1_res=strat_1,
                                strat_2_res=strat_2
                                )




    
    # ### Expert: Inaccurate, Trusting 

    # #### HIC

    # In[ ]:


    #setting up fixed params

    current_user_name= "inaccurate_trusting"
    user_suffix= 'inacc_t'

    exp_path= fr"./trained_experts/{ds_name}/{current_user_name}.pkl"
    current_expert= joblib.load(exp_path)


    params = base_params.copy()  # Start with the base
    params.update({
        "user_name": current_user_name,
        "user_suffix": user_suffix
    })

    objects = base_objects.copy() # Start with the base clones
    objects.update({
        "user_model": current_expert
    })


    # In[ ]:


    #run_hic(ds_name, params, objects)


    
    # #### Calib

    # In[ ]:


    #chosen nets: "large", 32/16

    layers= [32,16]
    hic_df_general_path= DATASETS[ds_name]['paths']['hic_df_save_path']
    net_general_path= DATASETS[ds_name]['paths']['def_net_save_path']
    # retrieving df and best net path

    for i in range (1, 4):

        df_switch_path= os.path.join(hic_df_general_path,
                                f"iter_{i}",
                                f"results_{params['user_name']}",
                                f"hic_{params['user_name']}.parquet"
                                )

        df_switch= pd.read_parquet(df_switch_path)


        def_net_path= os.path.join(net_general_path,
                            f"iter_{i}",
                            f"{user_suffix}_models",
                            f"{layers[0]}_{layers[1]}"
                            )

        pattern = f"{layers[0]}_{layers[1]}_model_*.pt"
        search_query = os.path.join(def_net_path, pattern)
        matches = glob.glob(search_query)

        if matches:
            def_net_path = max(matches, key=os.path.getctime)
            run_calibration(def_net_path= def_net_path, 
                            df_switch=df_switch, 
                            ds_name=ds_name, 
                            params=params,
                            device=device, 
                            iteration=i,
                            def_net_layers=net_layers
                            ) #default had batch seize 128, epochs = 20 


    # #### MIC

    # In[ ]:


    current_user_name= "inaccurate_trusting"
    user_suffix= 'inacc_t'

    exp_path= fr"./trained_experts/{ds_name}/{current_user_name}.pkl"
    current_expert= joblib.load(exp_path)


    params = base_params.copy()  # Start with the base
    params.update({
        "user_name": current_user_name,
        "user_suffix": user_suffix,
        "run_confidence": True,
        "run_mao": True
    })

    objects = base_objects.copy() # Start with the base clones
    objects.update({
        "user_model": current_expert
    })


    def_net_layers= [32, 16]
    strat_1=None
    strat_2=None

    for i in range(1,4):

        strat_1, strat_2= run_mic( 
                                ds_name= ds_name,
                                device= device,
                                layers= def_net_layers,
                                params=params,
                                objects=objects,
                                iteration=i,
                                strat_1_res=strat_1,
                                strat_2_res=strat_2)



    
    # ### Expert: Accurate, Not Trusting 

    # #### HIC

    # In[ ]:


    #setting up fixed params

    current_user_name= "accurate_not_trusting"
    user_suffix= 'acc_nt'

    exp_path= fr"./trained_experts/{ds_name}/{current_user_name}.pkl"
    current_expert= joblib.load(exp_path)


    params = base_params.copy()  # Start with the base
    params.update({
        "user_name": current_user_name,
        "user_suffix": user_suffix
    })

    objects = base_objects.copy() # Start with the base clones
    objects.update({
        "user_model": current_expert
    })


    # In[ ]:


    #run_hic(ds_name, params, objects)

    
    # #### CALIB

    # In[ ]:


    #chosen nets: "large", 32/16

    layers= [32,16]
    hic_df_general_path= DATASETS[ds_name]['paths']['hic_df_save_path']
    net_general_path= DATASETS[ds_name]['paths']['def_net_save_path']
    # retrieving df and best net path
    for i in range (1, 4):

        df_switch_path= os.path.join(hic_df_general_path,
                                f"iter_{i}",
                                f"results_{params['user_name']}",
                                f"hic_{params['user_name']}.parquet"
                                )

        df_switch= pd.read_parquet(df_switch_path)


        def_net_path= os.path.join(net_general_path,
                            f"iter_{i}",
                            f"{user_suffix}_models",
                            f"{layers[0]}_{layers[1]}"
                            )

        pattern = f"{layers[0]}_{layers[1]}_model_*.pt"
        search_query = os.path.join(def_net_path, pattern)
        matches = glob.glob(search_query)

        if matches:
            def_net_path = max(matches, key=os.path.getctime)
            run_calibration(def_net_path= def_net_path, 
                            df_switch=df_switch, 
                            ds_name=ds_name, 
                            params=params,
                            device=device, 
                            iteration=i,
                            def_net_layers=net_layers
                            ) #default had batch seize 128, epochs = 20 


    # #### MIC

    # In[ ]:


    current_user_name= "accurate_not_trusting"
    user_suffix= 'acc_nt'

    exp_path= fr"./trained_experts/{ds_name}/{current_user_name}.pkl"
    current_expert= joblib.load(exp_path)


    params = base_params.copy()  # Start with the base
    params.update({
        "user_name": current_user_name,
        "user_suffix": user_suffix,
        "run_confidence": True,
        "run_mao": True
    })

    objects = base_objects.copy() # Start with the base clones
    objects.update({
        "user_model": current_expert
    })


    def_net_layers= [32, 16]
    strat_1=None
    strat_2=None

    for i in range(1,4):

        strat_1, strat_2= run_mic( 
                                ds_name= ds_name,
                                device= device,
                                layers= def_net_layers,
                                params=params,
                                objects=objects,
                                iteration=i,
                                strat_1_res=strat_1,
                                strat_2_res=strat_2)



    
    # ### Expert: Inaccurate, Not Trusting 

    # #### HIC

    # In[ ]:


    #setting up fixed params

    current_user_name= "inaccurate_not_trusting"
    user_suffix= 'inacc_nt'

    exp_path= fr"./trained_experts/{ds_name}/{current_user_name}.pkl"
    current_expert= joblib.load(exp_path)


    params = base_params.copy()  # Start with the base
    params.update({
        "user_name": current_user_name,
        "user_suffix": user_suffix
    })

    objects = base_objects.copy() # Start with the base clones
    objects.update({
        "user_model": current_expert
    })


    # In[ ]:


    #run_hic(ds_name, params, objects)

    
    # #### CALIB

    # In[ ]:


    #chosen nets: "large", 32/16

    layers= [32,16]
    hic_df_general_path= DATASETS[ds_name]['paths']['hic_df_save_path']
    net_general_path= DATASETS[ds_name]['paths']['def_net_save_path']

    # retrieving df and best net path
    for i in range (1, 4):

        df_switch_path= os.path.join(hic_df_general_path,
                                f"iter_{i}",
                                f"results_{params['user_name']}",
                                f"hic_{params['user_name']}.parquet"
                                )

        df_switch= pd.read_parquet(df_switch_path)


        def_net_path= os.path.join(net_general_path,
                            f"iter_{i}",
                            f"{user_suffix}_models",
                            f"{layers[0]}_{layers[1]}"
                            )

        pattern = f"{layers[0]}_{layers[1]}_model_*.pt"
        search_query = os.path.join(def_net_path, pattern)
        matches = glob.glob(search_query)

        if matches:
            def_net_path = max(matches, key=os.path.getctime)
            run_calibration(def_net_path= def_net_path, 
                            df_switch=df_switch, 
                            ds_name=ds_name, 
                            params=params,
                            device=device, 
                            iteration=i,
                            def_net_layers=net_layers
                            ) #default had batch seize 128, epochs = 20 


    # #### MIC

    # In[ ]:


    current_user_name= "inaccurate_not_trusting"
    user_suffix= 'inacc_nt'

    exp_path= fr"./trained_experts/{ds_name}/{current_user_name}.pkl"
    current_expert= joblib.load(exp_path)


    params = base_params.copy()  # Start with the base
    params.update({
        "user_name": current_user_name,
        "user_suffix": user_suffix,
        "run_confidence": True,
        "run_mao": True
    })

    objects = base_objects.copy() # Start with the base clones
    objects.update({
        "user_model": current_expert
    })


    def_net_layers= [32, 16]
    strat_1=None
    strat_2=None

    for i in range(1,4):

        strat_1, strat_2= run_mic( 
                                ds_name= ds_name,
                                device= device,
                                layers= def_net_layers,
                                params=params,
                                objects=objects,
                                iteration=i,
                                strat_1_res=strat_1,
                                strat_2_res=strat_2)






    

