#!/usr/bin/env python
# coding: utf-8

"""
BRIDGET Demo

Reduced experimental pipeline

Dataset: Dutch Census
From: https://github.com/tailequy/fairness_dataset/tree/main/Dutch_census
Paper: [A survey on datasets for fairness-aware machine learning (Tai Le Quy et al.)](https://arxiv.org/abs/2110.00530)

This script executes the entire pipeline interacting with one expert archetype (Accurate, Trusting).
It is organized as follows:
    * Brief data preparation and splitting
    * Incremental Learners Validation
"""

# ## Dataset Preprocessing
# 

# ### Libraries, Retrieving data

 


import os
import yaml
import joblib
import pickle
import copy
import pandas as pd

from river import tree
from river import preprocessing
from river import ensemble, forest, compose

from bridget_utils import *
from orchestrator import *
from master_config import *
from classes import BetaUser
from bridget_main import HiC
from master_config import DATASETS

def load_and_partition_dutch_demo(ds_name):
    """
    Handling preprocessing and splitting
    
    ds_name(str): name of the dataset 
    """

    data= pd.read_csv(fr"./datasets/{ds_name}.csv")
    data= preprocess(data, drop_duplicates=True)

    mapping= { 
        'sex': {'male': 0, 'female': 1}
    }

    data= apply_map(data, mapping)

    data = data.sample(frac=1, random_state=42).reset_index(drop=True) 

    # setting up stratification
    class_0 = data[data['occupation'] == 0]
    class_1= data[data['occupation'] == 1]

    splits= {
        'calibration': (0.6, 0.8), 'mic': (0.8, 1.0),
        'warm_up_train': (0.0, 0.07), 'warm_up_test': (0.07, 0.1), 
        'hic_train': (0.1, 0.5), 'hic_test': (0.5, 0.6)
    }

    dfs= {name: stratif(start, end, class_0, class_1) for name, (start, end) in splits.items()}

    return data, dfs


if __name__ == "__main__":
    
    set_all_seeds(42)
    
    ds_name= 'dutch'
    device= torch.device("cpu")


    # 1. PARTITIONING

    # setting up the structures required for partitioning
    data, dfs= load_and_partition_dutch_demo(ds_name)
   
    target= DATASETS[ds_name]['target']
    categoricals= ['sex', 'prev_residence_place']
    numericals= [c for c in data if c not in categoricals and c != target]

    prepr_transf = (
        (compose.Select(*numericals) | preprocessing.MinMaxScaler()) +
        compose.Select(*categoricals)
    )

    # splitting X and y, setting up 
    X_w_train, y_w_train = x_y_split(dfs['warm_up_train'], target)
    X_w_test, y_w_test = x_y_split(dfs['warm_up_test'], target)
    X_hic_train, y_hic_train = x_y_split(dfs['hic_train'], target)
    X_hic_test, y_hic_test = x_y_split(dfs['hic_test'], target)
    X_val, y_val = x_y_split(dfs['calibration'], target)
    X_mic, y_mic = x_y_split(dfs['mic'], target)
  
    # the train function requires a dict, thus transform the warm up feature matrix accordingly
    X_w_dict= X_w_train.to_dict(orient='records')
    X_w_dict_test= X_w_test.to_dict(orient='records')

    df_batch_1 = (dfs['hic_train']).reset_index(drop=True)
    mic_data= dfs['mic'].reset_index(drop=True)
    df_warmup= pd.concat([dfs['warm_up_train'], dfs['warm_up_test']]).reset_index(drop=True)

    test_batch_1= X_hic_test.copy()
    test_batch_1[target]= y_hic_test


    # 2. INCREMENTAL LEARNERS MODEL SELECTION
    base = tree.HoeffdingTreeClassifier(grace_period=100)

    models= {
        'HoeffdingATC': tree.HoeffdingAdaptiveTreeClassifier(grace_period= 100, seed= 42), 
        "EFDT": tree.ExtremelyFastDecisionTreeClassifier(grace_period=100), 
        "AdaBoostCl": ensemble.AdaBoostClassifier(model= base, n_models= 15, seed= 42), 
        "ADWINBAGGING": ensemble.ADWINBaggingClassifier(model= base, n_models= 15, seed= 42), 
        "SRPCL": ensemble.SRPClassifier(model= base, n_models=15, seed= 42), 
        "ARF": forest.ARFClassifier(n_models= 15, grace_period= 100, max_features='sqrt', seed=42)   
        }

    # setting up logging directory
    prepr_dir= DATASETS[ds_name]['base_obj_paths']['preprocessors']
    os.makedirs(prepr_dir, exist_ok=True)

    model_dir= DATASETS[ds_name]['base_obj_paths']['incremental_learners']
    os.makedirs(model_dir, exist_ok=True)

    for model_name, model_obj in models.items():

        hic_instance= HiC(
                    RULE= FRANK_RULES['RULE'],
                    PAST= FRANK_RULES['PAST'], 
                    SKEPT=FRANK_RULES['SKEPT'], 
                    GROUP= FRANK_RULES['GROUP'], 
                    EVA=FRANK_RULES['EVA'], 
                    n_bins=FRANK_RULES['N_BINS'], 
                    n_var=FRANK_RULES['N_VAR'], 
                    maxc=FRANK_RULES['MAXC'], 
                    rule_att=DATASETS[ds_name]['rule_att'], 
                    rule_value=DATASETS[ds_name]['rule_value'], 
                    hic_model_name='placeholder', 
                    hic_model=model_obj,
                    start_performance= DATASETS[ds_name]['start_performance'],
                    allocated_budget= DATASETS[ds_name]['allocated_budget'][0],
                    skepticism_threshold= DATASETS[ds_name]['skepticism_threshold'],
                    performance_delta= DATASETS[ds_name]['performance_delta'],
                    dataset_name= ds_name,
                    user_name= 'placeholder',
                    batch1=df_batch_1, 
                    batch3=mic_data, 
                    batch1_test=test_batch_1, 
                    target=target, 
                    user_model='placeholder', 
                    protected=DATASETS[ds_name]['protected'], 
                    cats=categoricals, 
                    num=numericals,
                    preprocessor=prepr_transf,
                    training_iter= 0 
                    )

        hic_instance.train(X_w_dict, y_w_train, X_w_dict_test, y_w_test)

        base_preprocessor = hic_instance.preprocessor
        base_model = hic_instance.hic_model


        prepr_filename = f"{model_name}_preprocessor.pkl"
        model_filename = f"{model_name}_model.pkl"

        joblib.dump(base_preprocessor, os.path.join(prepr_dir, prepr_filename))
        joblib.dump(base_model, os.path.join(model_dir, model_filename))


    # 3. EXPERT CALIBRATION

    with open(fr"./experts_{ds_name}.yaml", "r") as f:
        config= yaml.safe_load(f)

    params_dict= config['experts']['groups']['w_dict']

    X_exp= X_hic_train.to_dict(orient='records')

    X_exp_scaled= []

    for x in X_exp:
        X_exp_scaled.append(prepr_transf.transform_one(x))

    X_exp_final = pd.DataFrame(X_exp_scaled)

    name= 'accurate_trusting'
    expert_conf= config['experts']['groups'][name]
    
    expert_object= BetaUser(
            belief_level= expert_conf['belief_value'],
            rethink_level= 0.8, 
            fairness= True,
            fpr= expert_conf['target_FPR'],
            fnr= expert_conf['target_FNR'],
            alpha= 0.9, #intra rater agreement as per OpenL2D's pipeline
            features_dict= params_dict,
            seed= expert_conf['group_seed']
            )
    res= expert_object.fit(X_exp_final, y_hic_train, tol= 0.001)

    save_dir= fr"./trained_experts/{ds_name}"
    os.makedirs(save_dir, exist_ok= True)
    file_path = os.path.join(save_dir, f"{name}.pkl")

    with open(file_path, 'wb') as f:
        pickle.dump(expert_object, f)
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



    # 4. DEFINING CONFIGURATION
 
    initial_ordering= [c for c in data if c != target]

    rules= FRANK_RULES

    # e.g.: Adaptive Random Forest is chosen
    base_params= {
        "cats": categoricals, #lst
        "num": numericals, #lss
        "warm_up_set": df_warmup.copy(),
        "batch1": df_batch_1.copy(),
        "batch1_test": test_batch_1.copy(),
        "batch3":dfs['mic'].copy(),
        "validation_set": dfs['calibration'].copy(), 
        "feature_order": initial_ordering.copy(), 
        "incremental_learner_name":"ARF",
        "n_cols": len(initial_ordering), 
        "mic_model_name": "Def_Net",
        "strat_1_name": "Selective_Pred",
        "strat_2_name": "Two_Step_Def",
    }


    # retrieving warmed up ARF and corresponding preprocessor
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


    
    # e.g. testing the Accurate, Trusting archetype
    current_user_name= "accurate_trusting"
    user_suffix= 'acc_t'

    # loading
    exp_path= fr"./trained_experts/{ds_name}/{current_user_name}.pkl"
    current_expert= joblib.load(exp_path)


    # update setup
    
    params = base_params.copy()  
    params.update({
        "user_name": current_user_name,
        "user_suffix": user_suffix
    })

    objects = base_objects.copy() 
    objects.update({
        "user_model": current_expert
    })



    # 5. HUMAN IN COMMAND PHASE
    
    run_hic(ds_name, params, objects)


    # 6. CALIBRATION PHASE

    # initializing rejector calibration after selecting a net architecture
    # eg chosen net: "large", 32/16

    net_layers= [32,16]
    hic_df_general_path= DATASETS[ds_name]['paths']['hic_df_save_path']
    net_general_path= DATASETS[ds_name]['paths']['def_net_save_path']

    # retrieving df and best net's paths
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
                            ) 


    # 7. MACHINE IN COMMAND PHASE


    # specifying user assets again since you may potentially 
    # want to run MIC without passing through HIC training or calibration again
    
    current_user_name= "accurate_trusting"
    user_suffix= 'acc_t'

    exp_path= fr"./trained_experts/{ds_name}/{current_user_name}.pkl"
    current_expert= joblib.load(exp_path)

    # update base configuration
    params = base_params.copy()  
    params.update({
        "user_name": current_user_name,
        "user_suffix": user_suffix,
        "run_selective_prediction": True, #e.g. 
        "run_two_step_deferral": True
    })

    objects = base_objects.copy() 
    objects.update({
        "user_model": current_expert
    })


    # e.g. assuming you chose the [32, 16] architecture
    def_net_layers= [32, 16]
    strat_1=None #initializing empty results dictionaries 
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
