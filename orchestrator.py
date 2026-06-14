"""
========================================
BRIDGET ORCHESTRATION PIPELINE 
========================================

This script serves as the main execution orchestrator for the framework.
It handles the three macro-processes independently, by executing each step in the pipeline according to the iterations
and data configurations/splits specified in the master_config.py file (which is thorougly referenced in the code below), 
while also managing the logging of the results at each step.

In particular, the file is structured to:

* Define DeferralNets validation and training using the data emerging from the Human in Command co-evolutionary labeling
* Instantiate Human in Command and Machine in Command sessions, called in the respective run_hic and run_mic macro functions
* Run the entire pipeline for a given dataset, user and model configurations, with minimal supervisor intervention 
   (DeferralNet selection MUST be performed before the execution of the run_mic function, using the results provided by the run_calibration functions)

The guide is referenced in the README file and in the examples provided in the repo.


"""

import numpy as np
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import json
import glob
import joblib
from torch.optim.lr_scheduler import StepLR

from classes import *
from bridget_main import MiC, HiC
from bridget_utils import *
from master_config import DATASETS, NET_CONFIGS, FRANK_RULES, R_NET_CONFIGS

set_all_seeds(42)


#========================================
#HUMAN IN COMMAND PHASE ASSETS 
#========================================



def net_calibration(ds_name,  
                    user_suffix, 
                    user_model,
                    iter, 
                    target, 
                    train_set, 
                    val_set, 
                    feature_order, 
                    net_layers, 
                    net_params, 
                    step_size,
                    gamma,
                    device,
                    log,
                    baseline= False):


    """
    Trains the Nets using the co-labeled data and the hyperparams configuration within the master_config.py file, 
    checkpoints model weights, plots
    
    Args:
        ds_name (str): used for logging and saving purposes, derived from the master_config.py file (eg. "dutch", "compas")
        user_suffix (str): identification of the current synthetic user for logging purposes (eg. "acc_t")
        user_model (object): synthetic user model
        iter (int): number of current iteration
        target (str): target label in string format
        train_set (pd.DataFrame): co-labeled data obtained after the Human in Command phase
        val_set (pd.DataFrame): validation set scaled using the preprocessor updated after HIC
        feature_order (list): list of feature names in str format (NO TARGET LABEL)
        net_layers (list): list identifying the NN structures (eg. [16,8], [32,16]) as per master_config.py
        net_params (dict): dict containing the training configuration (eg dropout_coeff, lr, weight_decay, smoothing) as per master_config.py

    """

    set_all_seeds(42)
    
    log.info(f"Entered Net Calibration | USER {user_suffix} | iter {iter}")




    # 1. Data Loader and Tensor Datasets creation for the Training data and the validation data
    X_cal, y_cal, train_loader = create_loader(train_set, feature_order, target, shuffle=False)

    _, _, val_loader = create_loader(val_set, feature_order, target, shuffle=False) #default size= 128, shuffle= False

    
    
    
    # 2. DeferralNet instantiation and retriaval of its necessary components from the master_config.py file

    net= DeferralNet(input_size=X_cal.shape[1],  #default val for the dropout coeff was set to 0 for the def net class
                            hidden_layer1= net_layers[0], 
                            hidden_layer2= net_layers[1], 
                            output_size=2,
                            dropout_coeff=0.0)
    net.to(device)

    optimizer = optim.Adam(params=net.parameters(),
                       lr=net_params['lr'],
                       weight_decay=net_params['weight_decay']
                    )
    
    weight_0, weight_1= compute_class_weights(y_cal)  # computing class weights for the CE loss
    weights = torch.tensor([weight_0, weight_1]).to(device)

    criterion= nn.CrossEntropyLoss(weight=weights, label_smoothing=net_params['label_smoothing']) #applying label smoothing if necessary
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) #defining scheduler for ADAM

    

    # 3. Defining saving directory for the trained nets and the plots according to the respective synthetic user and current BRIDGET iteration
    if baseline: #this one was for the baseline
        net_dir= DATASETS[ds_name]['baseline_paths']['trained_nets']
        model_name= f'{net_layers[0]}_{net_layers[1]}'

        save_dir= os.makedirs(net_dir, exist_ok=True)
        save_dir = os.path.join(net_dir, 
                                f"{user_suffix}_models",
                                f"{model_name}")
     
    else:
        net_dir= DATASETS[ds_name]['paths']['def_net_save_path']
        model_name= f'{net_layers[0]}_{net_layers[1]}'

        save_dir= os.makedirs(net_dir, exist_ok=True)
        save_dir = os.path.join(net_dir, 
                                f"iter_{iter}",
                                f"{user_suffix}_models",
                                f"{model_name}")

    
    # 4. Training and validating of the Deferral Nets structures (including Loss plots, confusion matrix and classification report)
    
    training_history, validation_history = net_trainer(net, optimizer,
                                                criterion, 
                                                device, 
                                                train_loader, 
                                                val_loader,
                                                scheduler, 
                                                iter, 
                                                model_name, 
                                                save_dir,
                                                log_interval=100, patience=8, max_epochs=20)
    
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    axes[0].plot(training_history['accuracy'], label='train')
    axes[0].plot(validation_history['accuracy'], label='val')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].plot(training_history['loss'], label='train')
    axes[1].plot(validation_history['loss'], label='val')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    fig.tight_layout()
    plot_path = os.path.join(save_dir, f"plots_{model_name}")
    os.makedirs(plot_path, exist_ok=True)
    loss_curves_p= os.path.join(plot_path, f"loss_curves.png")
    plt.savefig(loss_curves_p)
    plt.close()

    cm_report= plot_confusion_matrix(net, val_loader, device, plot_path)

    report_df = pd.DataFrame(cm_report).transpose()
    report_save_path = os.path.join(save_dir, f"report_{model_name}.parquet")
    report_df.to_parquet(report_save_path)


    # 5. Calibrating Tau threshold for the Selective Prediction deferral policy (ref Mozannar et al. 2020) on the validation data using the trained nets
    log.info("Calibrating TAU thresholds")

    choose_optimal_tau(ds_name=ds_name,
                       user_suffix=user_suffix, # da fuori
                       user_model= user_model,
                       net=net,
                       model_path=save_dir, #viene dall'alto
                       device=device, #probabilmente viene dall'alto 
                       layers=net_layers,#vengono dall'alto, always a lst btw
                       validation_set=val_set,
                       feat_order= feature_order,
                       target=target,
                       iteration= iter,
                       log=log,
                       min_coverage= 0.6
                        )

    

def hic_session(warm_up_set, hic_df_save_path,
                 target, 
                 protected, 
                 categoricals, 
                 numericals,
                 ds_name, 
                 rules, 
                 user_name, 
                 user_model, 
                 hic_iter, 
                 preprocessor, 
                 hic_model_name, 
                 hic_model, 
                 batch3, 
                 batch1, 
                 batch1_test,
                 allocated_budget,
                 start_performance, 
                 performance_delta, 
                 skept_thresholds,
                 rule_att,
                 rule_value,
                 log):
    
    """
    Instantiating the HiC class, running the start_HiC method of the HiC subclass. Then it logs and saves the results in the respective folders

    Although the list of args is extensive, the parameters are passed down by the run_hic function that requires the "params" and "objects" arguments.
    We refer to that to avoid repetitions


    Returns:
        hic_df (pd.DataFrame): structure containing the final predictions provided by the HIC phase, includes informative cols (e.g., "provider" of the final label of the row) 
        hic_inst (object): class instance, used to extract the benchmark result for the subsequent iteration
        evaluation_results (pd.DataFrame): DataFrame reporting the results pertaining to the fairness metrics, skepticism and XAI counters
        train_acc (list): list of accuracy values computed at each row
        train_f1 (list): list of f1 scores values computed at each row

    
    """
    hic_inst= HiC(
                    cats=categoricals,
                    num=numericals,
                    user_name=user_name,
                    training_iter=hic_iter,
                    hic_model_name=hic_model_name,
                    batch1=batch1,
                    batch3=batch3,
                    batch1_test=batch1_test,
                    user_model=user_model,
                    preprocessor=preprocessor,
                    hic_model=hic_model,                    
                    dataset_name=ds_name,
                    target=target,
                    protected=protected,
                    log=log,
                    allocated_budget=allocated_budget,
                    performance_delta=performance_delta,
                    skepticism_threshold=skept_thresholds,
                    RULE= rules['RULE'], 
                    PAST= rules['PAST'], 
                    SKEPT= rules['SKEPT'],
                    GROUP= rules['GROUP'],
                    EVA= rules['EVA'],
                    n_bins= rules['N_BINS'],
                    n_var= rules['N_VAR'],
                    maxc= rules['MAXC'],
                    rule_att= rule_att,
                    rule_value= rule_value,
                    start_performance=start_performance )
    
    hic_df, train_acc, train_f1 = hic_inst.start_HiC(warm_up_set)


    iter_dir = os.path.join(hic_df_save_path, f"iter_{hic_iter}", f"results_{user_name}")
    os.makedirs(iter_dir, exist_ok=True)

    file_path = os.path.join(iter_dir, f"hic_{hic_inst.name}.parquet")
    hic_df.to_parquet(file_path, index=False)

    log.info(f"HiC df saved to: {file_path}")


    evaluation_results = hic_inst.get_eval_report()

    eval_path = os.path.join(iter_dir,  f"hic_{hic_inst.name}_evaluation.parquet")
    evaluation_results.to_parquet(eval_path, index=False)

    log.info(f"HiC evaluation res saved to: {eval_path}")
    
    
    return hic_df, hic_inst, evaluation_results, train_acc, train_f1


def run_hic(ds_name, params, objects):

    """
    
    This function handles the entire HIC pipeline.
    It requires the params and objects structure.

    Args:

    * ds_name (str): dataset name, for logging purposes and necessary to build the saving directory
    * objects (dict): a dictionary which is built prior to BRIDGET's initialization. 
                After preparing the data and conducting the necessary preprocessing steps for each dataset, the dict is filled with

                * user_model (obj): straightforward, the user model frfom the BetaUser subclass (modulable, can include another class in the classes.py) 
                  that will call .predict during the co-evolutionary labeling
                * preprocessor (obj): River Preprocessor (the scaler), warmed up on the warm up set alongside the Incremental Learner.
                                      During the streaming co evolutionary process this scaler is continuously updated, until the iteration comes to and end.
                                      It is then checkpointed, used to scale the Validation and MIC Testing data and reloaded to the subsequent
                                      Human in Command iteration to learn again


                * incremental_learner (obj): River (although again, its modulable, any streaming model can be used) Incremental Model.
                                             Pre-trained on a separate warm-up set, it then enters the co-evolutionary training and its updated/re-trained from scratch
                                             according to the rules in bridget_main.py. Once the HIC phase ends, its checkpointed and frozen.
                                             Then, its reloaded at the subsequent HIC iteration to resume training and labeling.



    * params (dict): Due to the nature of the datasets and rules tested hereby, the master_config.py file contains 
              partial information only pertaining to the structures required in the HIC phase, organized in the DATASETS dictionary asset that is 
              imported at the top of the scripts.
              Params addresses the following requirements prior to deploying BRIDGET, and contextually with building the "objects" dictionary
    
              *warm_up_set (pd.DataFrame): DataFrame of data allocated towards pre-training the Incremental Learner
              *cats (list): list of categorical features in str format
              *num (list): list of numerical features in str format
              *incremental_learner_name (str): chosen saving name of the Incremental Learner (e.g., "ARF")
              *batch1 (pd.DataFrame): training data for the Human in Command phase
              *batch3 (pd.DataFrame): testing data for the Machine in Command phase
              *batch1_test (pd.DataFrame): testing data (diagnostics only) for the Human in Command phase
              *feature_order (list): list of the features' names in str format
              *user_suffix (str): used for building the saving structures (e.g., "acc_t")
    
    """

    # ========================================
    # 1. Creating the custom log for the current synthetic user and dataset
    # ========================================
    user_name=params['user_name']
    log_dir = DATASETS[ds_name]['paths']['hic_logs']
    user_log_path = os.path.join(log_dir, f"{user_name}.log")
    
    log = custom_log(params['user_name'], user_log_path)
    log.info(f"STARTING HIC RUN | User: {params['user_name']} | DS: {ds_name}")


    # ========================================
    # 2. Retrieving necessary assets from the params, objects dictionaries 
    # ========================================

    # (these contain a mix of fixed parameters listed beforehand in the master_config.py file, as well as custom params and configs for the 
    # current dataset that must be explicited after data preparation and splitting)
     
    current_preprocessor= objects['preprocessor']
    current_incremental_learner= objects['incremental_learner']

    warm_up_set=params['warm_up_set']
    categoricals=params['cats']
    numericals=params['num']
    
    
    hic_model_name=params['incremental_learner_name']
    batch1=params['batch1']
    batch3=params['batch3']
    batch1_test=params['batch1_test']
    
    user_model=objects['user_model']

    target=DATASETS[ds_name]['target']
    protected=DATASETS[ds_name]['protected']
    df_switch_save_path=DATASETS[ds_name]['paths']['hic_df_save_path']
    budgets=DATASETS[ds_name]['allocated_budget'] #since this is a list you'll need to get the proper one
    performance_delta=DATASETS[ds_name]['performance_delta']
    skept_thresholds=DATASETS[ds_name]['skepticism_threshold']
    batches_offsets= DATASETS[ds_name]['batches_offset']

    feature_order= params['feature_order']
    net_params=DATASETS[ds_name]['net_params']

    user_suffix= params['user_suffix']
    
    rule_att= DATASETS[ds_name]['rule_att']
    rule_value= DATASETS[ds_name]['rule_value']
    rules=FRANK_RULES

    

    # ========================================
    # 3. Instantiating hic_res dict to keep track of the current performance that will act as a benchmark (+/- performance_delta) for the subsequent iterations
    # ========================================
    hic_res= {
        'initial_state': {
            'performance_goal': DATASETS[ds_name]['start_performance'],
    }
    }
    



    # ========================================
    # 4. MAIN HUMAN IN COMMAND LOOP
    # ========================================
    # iterating through the 3 iterations defined in the conceptual pipelin. As the exact data splits are defined in the master_config.py,
    # if more than 3 iterations are needed, the for loop must be modified accordingly, as well as the data splits in master_config
    # no update is required on the logging functions or the directory definition since its modular and will handle it automatically

    for iteration in range(1,4):
        
        #4.A Defining starting performance for the current iter, referencing splits in master_config (Modular, no need to change anything)
        current_perf= hic_res['initial_state']['performance_goal']

        log.info(f"---  ITERATION {iteration}/3 ---")
        log.info(f"Current performance benchmark: {current_perf:.2f}")

        start_idx = batches_offsets[iteration-1]
        end_idx = batches_offsets[iteration]
        current_batch = batch1.iloc[start_idx:end_idx]


        # 4.b Running hic_session function
        hic_df, hic_inst, _, test_acc, test_f1 = hic_session(
            ds_name=ds_name,
            preprocessor=current_preprocessor,
            hic_model=current_incremental_learner,
            start_performance=current_perf,                                    
            warm_up_set=warm_up_set,
            categoricals=categoricals,
            numericals=numericals,
            user_name=user_name,
            hic_iter=iteration,
            hic_model_name=hic_model_name,
            batch1=current_batch,
            batch3=batch3,
            batch1_test=batch1_test,
        
            user_model=user_model,
            hic_df_save_path= df_switch_save_path,
            target=target,
            protected=protected,
            allocated_budget=budgets[iteration-1],
            performance_delta=performance_delta,
            skept_thresholds=skept_thresholds,
            rule_att= rule_att,
            rule_value= rule_value,
            rules=rules,
            log= log          
            
        )

        log.info(f" HiC Session completed. Acc on Train set: {np.mean(test_acc):.4f}, F1 Acc: {np.mean(test_f1):.4f}")
        

        # 4.C Scaling the validation set data AND the Machine in Command Testing data (which are then saved in the respective directories)
        # using the latest version of the RIVER preprocessor trained in HIC alongside the Incremental Learner
        
        log.info(" Scaling validation and MIC sets...")
        val_set, _ = scale_data(ds_name, iteration, params) # returnin



        # 4.D DeferralNets training and validation, and respective Tau threshold computation

        for arch_name in ["small", "medium", "large", "xl"]:
            layers = NET_CONFIGS[arch_name]['layers']
            step_size= NET_CONFIGS[arch_name]['step_size']
            gamma= NET_CONFIGS[arch_name]['gamma']
            log.info(f"Calibrating Def Net: {arch_name.upper()} (Layers: {layers})")

            net_calibration(ds_name= ds_name, 
                            user_suffix= user_suffix, 
                            user_model= user_model,
                            iter= iteration, 
                            target= target, 
                            train_set= hic_df, 
                            val_set= val_set, 
                            feature_order= feature_order, #WITHOUT LABEL
                            net_layers= layers, #string ("small" or "large")
                            net_params=net_params,
                            step_size= step_size,
                            gamma=gamma, 
                            log= log,
                            device= torch.device("cpu")
                            )

    
            
            log.info(f"Plots and conf matrix saved | ITER {iteration} | ARCHITECTURE {arch_name}")


        # 4.E Retrieving assets for the next iteration
        log.info(f"Loading updated objects for the next iter...")

        
        
        # PREPROCESSOR 
        
        prepr_dir= DATASETS[ds_name]['paths']['trained_preprocessor'] # prepr_dir
        prepr_path= os.path.join(
                prepr_dir, 
                f"iter_{iteration}", 
                f"results_{user_name}",
                f"User_{user_name}_{hic_model_name}_preprocessor.pkl"
                )
        trained_preprocessor = joblib.load(prepr_path)

        # INCREMENTAL MODEL
       
        model_dir= DATASETS[ds_name]['paths']['incremental_learner'] # prepr_dir
        model_path= os.path.join(
                model_dir, 
                f"iter_{iteration}", 
                f"results_{user_name}",
                f"User_{user_name}_{hic_model_name}_model.pkl"
                )
        
        trained_model = joblib.load(model_path)

        # CURRENT PERFORMANCE

        current_perf= (np.mean(hic_inst.machine_fea))*100
        log.info(f"Iteration {iteration} res | Machine EA: {current_perf:.2f}")

        #  UPDATE for the next iteration
        # Usually, you'd update this based on the 'train_acc' or 'eval_results' 
        # so the next HiC run knows where the human/machine left off
        hic_res['initial_state']['performance_goal'] = current_perf
    
        current_preprocessor= trained_preprocessor
        current_incremental_learner= trained_model

    log.info("ALL HIC ITERATIONS FINISHED SUCCESSFULLY!")




#========================================
#CALIBRATION PHASE ASSETS 
#========================================

def run_calibration(def_net_path, # DA PATH
                    df_switch, #INSERIRE MANUALMENTE # remember it should be the correct df
                    ds_name,
                    params,
                    device,
                    iteration,
                    def_net_layers, # you get it from outside since you loop outside,
                    batch_size=128, #DEF VALUE, CAN REMOVE
                    epochs= 20,
                    baseline= False
                    ):
    

    """
    This function handles the preliminary steps to deploy the Two-Stage Deferral Strategy within the Machine in Command phase.
    From the pipeline standpoint, this process happens after the supervisor has chosen the optimal DeferralNet architecture
    and before initializing Machine in Command
    
    Args:
        def_net_path (str): directory saving path as specified in master_config.py
        df_switch (pd.DataFrame): co-labeled data resulting from the corresponding iteration of Human in Command
        ds_name (str): dataset name in string format
        params (dict): 
        device (torch.device): 
        iteration (int):
        def_net_layers (list): net architecture chosen by the supervisor after model selection (e.g. [32,16])
        batch_size (int, default= 128): 
        epochs (int, default= 20): number of training epochs 
        baseline (bool)
    
    """
    #the idea within this function is simple: use the functions train r net and generate report of the anqi mao values :)

    # ========================================
    # 1. Creating saving directories using user name
    # ========================================
    user_name= params['user_name']

    if baseline:
        log_dir = DATASETS[ds_name]['baseline_paths']['2SD_baseline_log']
    else:
        log_dir = DATASETS[ds_name]['paths']['r_net_training_logs']

    user_log_path = os.path.join(log_dir, f"{user_name}.log")
    
    log = custom_log(params['user_name'], user_log_path)
    log.info(f"STARTING R-NETS TRAINING | User: {params['user_name']} | DS: {ds_name}")

    # ========================================
    # 2. Retrieving necessary assets, creating Tensor Dataset and Loader for the Training data
    # ========================================
    # these two refer to the Two-Stage Deferral Strategy (Mao et al., 2023) and load the configs from the master_config.py file
    # if there should arise the need to expand or restrict the inference cost (Beta) search interval, modify the corresponding
    # values in master_config, there is no need to specify it in this function
    r_conf= R_NET_CONFIGS   
    betas= r_conf.get('betas', [])

    target=DATASETS[ds_name].get('target')
    feat_order= params.get('feature_order')

    # creating loader for the training data
    X_cal, y_cal, _ = create_loader(df_switch, feat_order, target, shuffle= False)

    if baseline:
        log.info(f"--- Benchmark run calibration ---")
    else:
        log.info(f"--- Calibrating for iteration {iteration} ---")
    




    # ========================================
    # 3. Training Rejector Net on co-evolutionary data, looping over the Human Inference Cost (BETA), then checkpointing
    # ========================================
    # the nets calibrated will have the exact same architecture as specified in the 
    for beta in betas:
        log.info(f"--- Training r-net for Beta: {beta}")
        
        r_net= train_r_net(df_switch, 
                            ds_name,
                            user_name, 
                            device, 
                            alpha=r_conf.get('alpha'), 
                            beta= beta, 
                            feat_order=feat_order, #its required without the labels here
                            layers= r_conf.get('architecture'), 
                            dropout=r_conf.get('dropout'),
                            learning_rate=r_conf.get('lr'),
                            iter= iteration,
                            output_size= r_conf.get('output_size'),
                            log=log,
                            batch_size=batch_size,
                            epochs= epochs,
                            baseline= baseline
                            )
        

        # then the report to choose the anqi mao thresholds from is composed
        generate_thresh_report(
                        X_cal, 
                        y_cal,
                        r_net,
                        def_net_path, 
                        ds_name,
                        user_name,
                        device, 
                        iteration,
                        beta,
                        lower_thresh= r_conf.get('lower_thresh'), 
                        upper_thresh=r_conf.get('upper_thresh'), 
                        linspace_dimension=r_conf.get('linspace_dimension'),
                        def_net_layers= def_net_layers,
                        baseline= baseline)
        
    log.info(f"--- R-NET TRAINING FINISHED | User: {params['user_name']} | DS: {ds_name}")
    

def choose_optimal_tau(ds_name,
                       user_suffix, # da fuori
                       user_model,
                       net,
                       model_path, #viene dall'alto
                       device, #probabilmente viene dall'alto 
                       layers,#vengono dall'alto, always a lst btw
                       validation_set,
                       feat_order,
                       target,
                       iteration,
                       log,
                       min_coverage= 0.6
                        ):
    
   
    """
    Function that handles the optimal deferral threshold search for the Net architecture tested, then saves it to be reloaded in the 
    Machine in Command phase after model selection

    Args:
        All params spared for min_coverage are passed to this function within the run_calibration process. To avoid repetitions, refer to the 
        description above.

        min_coverage (float): Minimum proportion of rows that must be evaluated by the machine within the SelectivePrediction deferral policy context
                              that allows to assess the accuracy-coverage trade off.
                              Default value is 0.6.
        
    Returns:
        best_tau (float): Optimal Tau threshold scored within the linear space search interval for the specific net architecture, user and dataset tested
                          Used for logging purposes during the run_calibration function.
    
    """

    # ========================================
    # 1. Instatiating the Tensor Dataset loader for the validation set
    # ========================================

    X_val, y_val, _= create_loader(validation_set, feat_order, target, shuffle=False)


    # ========================================
    # 2. Conducting Linear Space search
    # ========================================

    best_tau, best_acc= calibrate_tau(ds_name=ds_name, 
                                      user_suffix=user_suffix,
                                      iteration=iteration,
                                      layers=layers,
                                      net= net, 
                                      user_model=user_model, 
                                      device=device, 
                                      X_val= X_val, 
                                      y_val=y_val, 
                                      log=log, 
                                      min_coverage= min_coverage)
    

    # ========================================
    # 3. Reporting best results, saving to corresponding directory
    # ========================================

    log.info(f" Best tau for {ds_name}, Iter {iteration}: {best_tau} (Acc: {best_acc})")

    model_key = f"{layers[0]}_{layers[1]}_{user_suffix}_model"
    
    if os.path.isfile(model_path):
        save_folder = os.path.dirname(model_path)
        net_path = model_path
    else:
        save_folder = model_path
        pattern = f"{layers[0]}_{layers[1]}_model_*.pt"
        search_query = os.path.join(model_path, pattern)
        matches = glob.glob(search_query)
        net_path = max(matches, key=os.path.getctime) if matches else model_path
        
    tau_json_path = os.path.join(save_folder, f'tau_threshold_{model_key}.json')

    tau_report = {
        "model_key": model_key,
        "tau_threshold": float(best_tau),
        "best_val_acc": float(best_acc),
        "net_path": net_path,
        "ds_name": ds_name,
        "iteration": iteration
    }

    with open(tau_json_path, 'w') as f:
        json.dump(tau_report, f, indent=4)

    return best_tau
    

#========================================
#MACHINE IN COMMAND PHASE ASSETS 
#========================================

def mic_session(ds_name, 
                target, #SOPPRA
                layers,
                def_net_path,
                def_net_name,
                device,
                df_switch, #just for xai 
                test_batch, # PARAMS MA HA IL SUO PATH
                training_iter,
                benchmark_performance,
                performance_delta,
                warm_up,
                belief_threshold,
                user_model,
                user_name,
                batch1,
                batch1_test,
                log,
                preprocessor,
                cats,
                num,
                protected,
                tau_threshold= None,
                r_net= None,
                human_defer_cost= None #anqi mao's beta
                ):
        

        """
        
        
        Args:
         
        """

        X_stream= torch.tensor(data= test_batch.drop(columns=[target]).values, dtype=torch.float32).to(device)
        y_stream= torch.tensor(data=test_batch[target].values, dtype= torch.long).to(device)

        mic_net= DeferralNet(input_size=X_stream.shape[1], 
                         hidden_layer1=layers[0], hidden_layer2=layers[1], 
                         output_size= NET_CONFIGS['output_size'],
                         dropout_coeff= 0.0)
        mic_net.to(device)
        
        mic_net.load_state_dict(torch.load(def_net_path, map_location=device))
        mic_net.eval()


        mic_inst= MiC(mic_model=mic_net, 
                        mic_model_name= def_net_name, 
                        dataset_name = ds_name , 
                        batch1= batch1,
                        batch1_test=batch1_test,
                        batch3= test_batch, 
                        training_iter= training_iter,
                        target=target,
                        benchmark_performance=benchmark_performance, 
                        performance_delta=performance_delta, 
                        warm_up=warm_up,
                        belief_threshold= belief_threshold,
                        user_model= user_model,
                        user_name= user_name,
                        preprocessor=preprocessor,
                        cats=cats,
                        num=num,
                        log=log,
                        device= device,
                        protected=protected,
                        tau_threshold=tau_threshold,
                        human_deferral_cost=human_defer_cost)
        
    
        if r_net:
            mic_df =mic_inst.start_MiC(X_stream, y_stream, df_switch, r_net=r_net, two_step_deferral=True)

        else:   
            mic_df =mic_inst.start_MiC(X_stream, y_stream, df_switch)
        
        return mic_df, mic_inst




def run_mic(ds_name,
            device, #probabilmente viene dall'alto 
            layers,# vengono dall'alto, always a lst btw, oppure da master config
            params,# da fuori
            objects, # OBJECTS
            iteration,
            strat_1_res=None,
            strat_2_res=None,
            baseline= False
            ):


    """ 
    
    Args:
    ds_name,
            device, #probabilmente viene dall'alto 
            layers,# vengono dall'alto, always a lst btw, oppure da master config
            params,# da fuori
            objects, # OBJECTS
            iteration,
            strat_1_res (bool, optional)
            strat_2_res (bool, optional)
            baseline (bool, optional)
    
    Returns:
    strat_1_res (dict):
    strat_2_res (dict):
    """
    ########## 
    # creating the log first
    user_name = params['user_name']

    if baseline:
        log_dir = DATASETS[ds_name]['baseline_paths']['2SD_baseline_log']
        user_log_path = os.path.join(log_dir, f"{user_name}.log")
        
        log = custom_log(params['user_name'], user_log_path)
        log.info(f"STARTING BENCHMARK RUN | User: {params['user_name']} | DS: {ds_name}")

    else:
        log_dir = DATASETS[ds_name]['paths']['mic_logs']
        user_log_path = os.path.join(log_dir, f"{user_name}.log")
        
        log = custom_log(params['user_name'], user_log_path)
        log.info(f"STARTING MIC RUN | User: {params['user_name']} | DS: {ds_name}")


    ########## 
    # Retrieving necessary params
    
    user_suff= params['user_suffix']
    i_learner_name= params['incremental_learner_name']
    cats=params['cats']
    num=params['num']
    strat_1_name= params['strat_1_name']
    strat_2_name= params['strat_2_name']
    def_net_name= params['mic_model_name']
    run_confidence= params['run_confidence']
    run_mao = params['run_mao']
    batch1= params['batch1']
    batch1_test=params['batch1_test']

    current_expert= objects['user_model']

    target= DATASETS[ds_name]['target']
    warm_up= DATASETS[ds_name]['warm_up']
    belief_threshold= DATASETS[ds_name]['belief_threshold']
    protected=DATASETS[ds_name]['protected']

    betas= R_NET_CONFIGS['betas']  


    ########### PRELIMINARY: CREATE STORING STRUCTURES
    # A. CONFIDENCE / STRAT 1 STORAGE
    
    if strat_1_res is None or not strat_1_res:
        strat_1_res= {
            'initial_state': {
                'benchmark': DATASETS[ds_name]['benchmark_performance'],
                'delta': DATASETS[ds_name]['performance_delta'],
                'drift': False
        }
        }
    if strat_2_res is None or not strat_2_res:
        strat_2_res = {}
        
        for beta in betas:

            beta_str= str(beta).replace('.', '')
            strat_2_res[f"beta_{beta_str}"] = {
                'initial_state': {
                    'benchmark': DATASETS[ds_name]['benchmark_performance'],
                    'delta': DATASETS[ds_name]['performance_delta']
                }
            }



    
    log.info(f"---  ITERATION {iteration}/3 ---")
    
    # 1. retrieving deferral net (DEFINE NET, THEN GET WEIGHTS)

    net_dir= DATASETS[ds_name]['paths']['def_net_save_path']
    net_dir=os.path.join(net_dir,
                f"iter_{iteration}",
                f"{user_suff}_models",
                f"{layers[0]}_{layers[1]}"
                )
    
    pattern = f"{layers[0]}_{layers[1]}_model_*.pt"
    search_query = os.path.join(net_dir, pattern)
    matches = glob.glob(search_query)

    if matches:
        def_net_path = max(matches, key=os.path.getctime)

    log.debug(f"RETRIEVING NET | User {user_name} | Iter {iteration} | NET {pattern}")

    def_net= DeferralNet(input_size=params['n_cols'], 
                    hidden_layer1=layers[0],
                    hidden_layer2=layers[1],
                    output_size=NET_CONFIGS['output_size'],
                    dropout_coeff=0.0)
    
    #loading weights: REMEMBER TO RE-INSTANTIATE BEFORE RECALLING THE PATH
    def_net.load_state_dict(torch.load(def_net_path, map_location=device))  
    def_net.to(device)
    def_net.eval()
    
    # 2. retrieving pre processor

    if baseline:
        current_preprocessor= objects['scaler']
    
    else: 
        prepr_dir= DATASETS[ds_name ]['paths']['trained_preprocessor']
        prepr_path = os.path.join(
                    prepr_dir, 
                    f"iter_{iteration}", 
                    f"results_{user_name}", 
                    f"User_{user_name}_{i_learner_name}_preprocessor.pkl"
        )

        current_preprocessor = joblib.load(prepr_path)

    # 3. Retrieving df switch

    df_switch_base_path= DATASETS[ds_name]['paths']['hic_df_save_path']

    ds_path= os.path.join(df_switch_base_path,
                            f"iter_{iteration}",
                            f"results_{user_name}",
                            f"hic_{user_name}.parquet")
    
    df_switch= pd.read_parquet(ds_path)

    # 4. Retrieving corresponding scaled batch3 

    batch3_base_path= DATASETS[ds_name]['paths']['scaled_mic_batch']
    batch3_path= os.path.join(batch3_base_path,
                                f"iter_{iteration}",
                                f"{user_suff}_scaled_batch.parquet",
                                )

    batch3= pd.read_parquet(batch3_path)

    # 5. loading corresponding tau threshold

    tau_coeff_base= DATASETS[ds_name]['paths']['tau_calibration_res']
    tau_path= os.path.join(tau_coeff_base,
                           f"iter_{iteration}",
                           f"{user_suff}_models",
                           f"{layers[0]}_{layers[1]}",
                           f"tau_threshold_{layers[0]}_{layers[1]}_{user_suff}_model.json"
                           )
    with open(tau_path, 'r') as f:
        tau_data = json.load(f)

    current_tau = tau_data['tau_threshold']
    log.info(f"Loaded Tau: {current_tau}, iter {iteration}")
    
    # 6. building dir for saving the res at each iteration

    mic_save_dir = DATASETS[ds_name]['paths']['mic_df_save_path']

    mic_save_path = os.path.join(
                    mic_save_dir, 
                    f"iter_{iteration}",
                    f"results_{user_suff}",
                    "strats"
                    )
    os.makedirs(mic_save_path, exist_ok=True)

    ############ DEFERRAL STRAT 1: CONFIDENCE BASED
    if run_confidence:
        log.info(f"STARTING STRAT {strat_1_name} | USER {user_name} | Iter {iteration}")


        initial_state = strat_1_res['initial_state']
        start_perf_conf = initial_state["benchmark"]
        start_delta = initial_state['delta']
        log.info(f"Iteration {iteration} starting with benchmark: {start_perf_conf:.2f}, delta: {start_delta:.2f}")
        #log.info(f"Current benchmark: {start_perf_conf:.2f}, current delta: {start_delta:.2f}")

        mic_df, mic_inst= mic_session(
                                ds_name=ds_name,
                                def_net_name=def_net_name, 
                                target=target, 
                                layers=layers,
                                def_net_path=def_net_path,
                                device=device,
                                df_switch= df_switch, #just for xai 
                                test_batch=batch3, # PARAMS MA HA IL SUO PATH
                                training_iter= iteration,
                                batch1= batch1,
                                batch1_test=batch1_test,
                                benchmark_performance= start_perf_conf,
                                performance_delta= start_delta,
                                warm_up= warm_up,
                                belief_threshold= belief_threshold,
                                user_model= current_expert,
                                user_name= user_name,
                                preprocessor=current_preprocessor,
                                cats=cats,
                                num=num,
                                log=log,
                                protected=protected,
                                tau_threshold=current_tau,
                                r_net= None,
                                human_defer_cost= None #anqi mao's beta
                            )
        

        # updating performance for next iter

        drift_detected = len(mic_df) < len(batch3)
        new_delta= DATASETS[ds_name]['stricter_delta'] if drift_detected else DATASETS[ds_name]['performance_delta']
        

        strat_1_res[iteration] = {
            'benchmark': float(np.mean(mic_inst.fea_mic)) * 100,
            'delta': new_delta,
            'drift': drift_detected
        }

        strat_1_res['initial_state'] = strat_1_res[iteration]

        log.info(f"Finished Iteration {iteration}, strat {strat_1_name} | Machine EA: {strat_1_res[iteration]['benchmark']:.2f} | Delta: {new_delta}")
        
        # saving res
        s1_path = os.path.join(mic_save_path, f"{strat_1_name}.parquet") #metrics
        mic_df_path = os.path.join(mic_save_path, f"{strat_1_name}_data.parquet") # df
        
        pd.DataFrame([strat_1_res[iteration]]).to_parquet(s1_path, index=False)
        mic_df.to_parquet(mic_df_path, index=True)

        log.info(f"Saved Strat 1 results to: {s1_path}")

    else:
        log.info(f"Iter {iteration}, skipping {strat_1_name} strategy...")


    ############ DEFERRAL STRAT 2: ANQI MAO TWO STAGE DEFERRAL
    if run_mao:        

        log.info(f"STARTING STRAT {strat_2_name} | USER {user_name} | Iter {iteration}")
        

        for beta in betas:
            beta_str= str(beta).replace('.', '')
            beta_key = f"beta_{beta_str}"

            initial_state_mao = strat_2_res[beta_key]['initial_state']  

            start_perf_mao = initial_state_mao["benchmark"]
            start_delta_mao= initial_state_mao['delta']

            log.info(f" Current Beta {beta} | Start Perf: {start_perf_mao:.2f}")

            # 0. retrieving def net
            net_dir= DATASETS[ds_name]['paths']['def_net_save_path']
            net_path= os.path.join(net_dir,
                        f"iter_{iteration}",
                        f"{user_suff}_models",
                        f"{layers[0]}_{layers[1]}"
                        )
            pattern = f"{layers[0]}_{layers[1]}_model_*.pt"
            search_query = os.path.join(net_path, pattern)
            matches = glob.glob(search_query)

            if matches:
                def_net_path = max(matches, key=os.path.getctime)
            
            log.debug(f"RETRIEVING NET | User {user_name} | Iter {iteration}")

            # 1. retrieving r-net
            r_net_dir= DATASETS[ds_name]['paths']["r_net_path"]
            r_net_path= os.path.join(r_net_dir,
                            f"iter_{iteration}",
                            f"beta_{beta_str}",
                            f"r_net_{user_name}.pth"
                            )
            
            r_net= DeferralNet(input_size= params['n_cols'],
                                hidden_layer1=R_NET_CONFIGS['architecture'][0],
                                hidden_layer2=R_NET_CONFIGS['architecture'][1],
                                output_size=R_NET_CONFIGS['output_size'],
                                dropout_coeff=R_NET_CONFIGS['dropout'])
            
            r_net.load_state_dict(torch.load(r_net_path, map_location=device))
            r_net.to(device)
            r_net.eval()


            log.debug(f"RETRIEVING R-NET | PATH {r_net_path}")

           
            # 3. starting MIC SESSION
            mic_df, mic_inst= mic_session(
                                ds_name=ds_name, 
                                def_net_name=def_net_name,
                                target=target, #SOPPRA
                                layers=layers,
                                def_net_path=def_net_path,
                                device=device,
                                df_switch= df_switch, #just for xai 
                                test_batch=batch3, # PARAMS MA HA IL SUO PATH
                                batch1= batch1,
                                batch1_test=batch1_test,
                                training_iter= iteration,
                                benchmark_performance= start_perf_mao,
                                performance_delta= start_delta_mao,
                                warm_up= warm_up,
                                belief_threshold= belief_threshold,
                                user_model= current_expert,
                                user_name= user_name,
                                preprocessor=current_preprocessor,
                                cats=cats,
                                num=num,
                                log=log,
                                protected=protected,
                                tau_threshold=None,
                                r_net= r_net,
                                human_defer_cost=beta
                            )
            


            strat_2_res[beta_key][iteration] = {
                'benchmark': float(np.mean(mic_inst.fea_mic)) * 100,
                'delta': DATASETS[ds_name]['performance_delta']
            }

            strat_2_res[beta_key]['initial_state'] = strat_2_res[beta_key][iteration]


            log.info(f"Beta {beta_str} | Iteration {iteration} | Machine EA: {strat_2_res[beta_key][iteration]['benchmark']:.2f}")

            # saving res
            s2_path = os.path.join(mic_save_path, f"{strat_2_name}_beta_{beta_str}.parquet")
            current_beta_data = strat_2_res[beta_key][iteration]

            mic_df_s2_path = os.path.join(mic_save_path, f"{strat_2_name}_beta_{beta_str}_data.parquet")#df
            
            pd.DataFrame([current_beta_data]).to_parquet(s2_path, index=False)
            mic_df.to_parquet(mic_df_s2_path, index=True)

            log.info(f"Saved Strat 2, Beta {beta} results to: {s2_path}")

    else:
        log.info(f"Iter {iteration}, skipping {strat_2_name} Deferral strategy...")
    


    return strat_1_res, strat_2_res
                 
                

 
    
    
   
    
    




