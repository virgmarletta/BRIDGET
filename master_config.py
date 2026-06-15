"""
========================================
BRIDGET EXPERIMENTAL CONFIGURATIONS
========================================


This module defines the global scheme, hyperparameters configuration, and path structures relative to the datasets used to
validate BRIDGET (Dutch Census, COMPAS, Adult).
It is organized as follows:
    * DATASETS (dict): General archive containing starting conditions, specific rule values, saving paths realtive to the datasets
    * NET_CONFIGS (dict): Architectural structures of the DNNs predictors engaging in the Machine in Command phase
    * R_NET_CONFIGS (dict): Architectural structure and hyperparameter grid for the rejector acting within the context of 
                            the Two-Stage Deferral Strategy in Machine in Command.
    * FRANK_RULES (dict): Collection of boolean flags enabling the 4 interaction rules of the Human in Command phase
    (adapted from the FRANK algorithm, original source code available at [https://github.com/FedericoMz/Frank/tree/main])
"""



DATASETS = {
    "dutch": {
         
        "target": "occupation", #prediction task
        "yaml_prefix": "experts_dutch", # identifies yaml file containing the values necessary for expert calibration
    
        "protected": ['sex', 'age', 'citizenship'], # protected attributes
        
        # Ideal Rule Check setup
        "rule_att" : 'edu_level', 
        "rule_value" : 0.1,
        
        "paths": { 
                "dataset_path": r"./datasets/dutch.csv", 
                "user_obj_path": r"./trained_experts/dutch", 
                "trained_preprocessor": r"./HIC_res/dutch", # PREPROCESSOR TRAINED BY THE HIC PHASE 
                "incremental_learner": r"./HIC_res/dutch", #INCREMENTAL LEARNER TRAINED BY THE HIC PHASE 

                "hic_df_save_path": r"./processed_data/dutch/hic_switch_ds", 
                "hic_previous_performance": r"./HIC_res/dutch", # PREVIOUS PERFORMANCE (HIC)
                "def_net_save_path": r"./nets/dutch",   #SAVE DIRECTORY FOR TRAINED NETS AND TAU THRESHOLDS
                "mic_previous_performance": r"./MIC_res/dutch", # PREVIOUS PERFORMANCE(MIC)
                "mic_df_save_path": r"./processed_data/dutch/mic_result_ds", # DF PRODUCED BY MIC PHASE
                
                "r_net_path": r"./r_nets/dutch", #R-NETS FOR TWO-STAGE DEFERRAL 
                "two_stage_thresholds": r"./r_nets_results/dutch",
                "validation_data_save_path": r"./processed_data/dutch/validation_data",
                "scaled_mic_batch": r"./processed_data/dutch/scaled_batch3",
                "hic_logs": r"./logs/dutch/hic",
                "r_net_training_logs": r"./logs/dutch/r_net_training",
                "mic_logs": r"./logs/dutch/mic",
                "tau_calibration_res": r"./nets/dutch"
                
        },

        # directory of the warmed up scaler and incremental learners, loaded at the initial BRIDGET deployment
        "base_obj_paths": {
            'preprocessors': r"./HIC_res/dutch/base_objects/preprocessors",
            'incremental_learners': r"./HIC_res/dutch/base_objects/incremental_learners"

        }, 
        
        # separate handles for future benchmarking
        "baseline_paths": {
            "trained_experts": r'./baseline/experts/dutch',
            "trained_scalers": r'./baseline/scalers/dutch',
            "train_df_labeled_by_experts": r"./baseline/mic/labeled_ds/dutch/train", 
            "test_df_for_mic": r"./baseline/mic/labeled_ds/dutch/test",  
            "validation_set_labeled": r"./baseline/mic/labeled_ds/dutch/val",

            # two stage deferral strat
            "calibrated_r_nets":  r"./baseline/mic/r_nets/dutch",
            "trained_nets": r"./baseline/mic/nets/dutch",
            "two_stage_thresh_baseline": r"./baseline/mic/thresholds/dutch",

            # benchmarking strats
            "benchmark_res": r"./baseline/mic/dutch",        
            "logs": r"./baseline/logs/dutch"
        },
        
        
        
        # Human in Command phase setup
        "allocated_budget": [2459, 2459, 2458],
        "batches_offset": [0, 2459, 4918, 7376],
        "skepticism_threshold": 0.5,
        "start_performance": 63, #a default value for the first iteration
        

        # Deferral Nets calibration
        "net_params": {
            "lr": 0.001,
            "weight_decay": 1e-2,
            "label_smoothing": 0.00, 
            
        },
        
        # Machine in Command phase setup
        "benchmark_performance": 63, #a default value for the first iteration
        'performance_delta': 0.05,
        'stricter_delta': 0.03,
        'warm_up': 738, #20% of 3688
        'belief_threshold':0.7,
        
    
    },


    "compas": { 
        "target": "did_recid",  #prediction task
        "yaml_prefix": "experts_compas", # identifies yaml file containing the values necessary for expert calibration

        "protected": ['race', 'age', 'sex'], # protected attributes
        
        # Ideal Rule Check setup
        "rule_att" : 'priors_count', 
        "rule_value": -0.7,
                

        "paths": { 
                "dataset_path": r"./datasets/compas.csv", 
                "user_obj_path": r"./trained_experts/compas", 
                "trained_preprocessor": r"./HIC_res/compas", # PREPROCESSOR TRAINED BY THE HIC PHASE 
                "incremental_learner": r"./HIC_res/compas", #INCREMENTAL LEARNER TRAINED BY THE HIC PHASE 

                "hic_df_save_path": r"./processed_data/compas/hic_switch_ds", #DF SWITCH (PRODUCED BY HIC PHASE)
                "hic_previous_performance": r"./HIC_res/compas", # PREVIOUS FEA(HIC)
                "def_net_save_path": r"./nets/compas",   #SAVE DIR FOR TRAINED NETS + TAU THRESHOLDS
                "mic_previous_performance": r"./MIC_res/compas", # PREVIOUS FEA(MIC)
                "mic_df_save_path": r"./processed_data/compas/mic_result_ds", # DF PRODUCED BY MIC PHASE
                "r_net_path": r"./r_nets/compas", #R-NETS FOR ANQI MAO DEFERRAL 
                "two_stage_thresholds": r"./r_nets_results/compas",
                "validation_data_save_path": r"./processed_data/compas/validation_data",
                "scaled_mic_batch": r"./processed_data/compas/scaled_batch3",
                "hic_logs": r"./logs/compas/hic",
                "r_net_training_logs": r"./logs/compas/r_net_training",
                "mic_logs": r"./logs/compas/mic",
                "tau_calibration_res": r"./nets/compas"
                
        },

         # directory of the warmed up scaler and incremental learners, loaded at the initial BRIDGET deployment
        "base_obj_paths": {
            'preprocessors': r"./HIC_res/compas/base_objects/preprocessors",
            'incremental_learners': r"./HIC_res/compas/base_objects/incremental_learners"

        }, 
        
         # separate handles for future benchmarking
        "baseline_paths": {
            "trained_experts": r'./baseline/experts/compas',
            "trained_scalers": r'./baseline/scalers/compas',
            "train_df_labeled_by_experts": r"./baseline/mic/labeled_ds/compas/train", 
            "test_df_for_mic": r"./baseline/mic/labeled_ds/compas/test",  
            "validation_set_labeled": r"./baseline/mic/labeled_ds/compas/val",

            # two stage deferral strat
            "calibrated_r_nets":  r"./baseline/mic/r_nets/compas",
            "trained_nets": r"./baseline/mic/nets/compas",
            "two_stage_thresh_baseline": r"./baseline/mic/thresholds/compas",

            # benchmarking strats
            "benchmark_res": r"./baseline/mic/compas",        
            "logs": r"./baseline/logs/compas"
        },


        # Human in Command phase setup
        "allocated_budget": [722, 722, 723],
        "batches_offset": [0, 722, 1444, 2167],
        "skepticism_threshold": 0.35,
        "start_performance": 63, #a default value for the first iteration  
        

        # Deferral Nets calibration
        "net_params": {
            "lr": 0.001,
            "weight_decay": 1e-3,
            "label_smoothing": 0.00,
    
          },

        # Machine in Command phase setup
        "benchmark_performance": 63, #a default value for the first iteration
        'performance_delta': 0.05,
        'stricter_delta': 0.03,
        'warm_up': 217, # 20% of 1085
        'belief_threshold':0.7,
    },

    
    "adult": {
        "target": "class",  #prediction task
        "yaml_prefix": "experts_adult", # identifies yaml file containing the values necessary for expert calibration

        "protected":['race', 'age', 'sex'], # protected attributes

        # Ideal Rule Check setup
        "rule_att" : 'capital-gain', 
        "rule_value" : 0.9, 

        "paths": { 
                "dataset_path": r"./datasets/adult_clean.csv", 
                "user_obj_path": r"./trained_experts/adult", 
                "trained_preprocessor": r"./HIC_res/adult", # PREPROCESSOR TRAINED BY THE HIC PHASE 
                "incremental_learner": r"./HIC_res/adult", #INCREMENTAL LEARNER TRAINED BY THE HIC PHASE 

                "hic_df_save_path": r"./processed_data/adult/hic_switch_ds", #DF SWITCH (PRODUCED BY HIC PHASE)
                "hic_previous_performance": r"./HIC_res/adult", # PREVIOUS FEA(HIC)
                "def_net_save_path": r"./nets/adult",   #SAVE DIR FOR TRAINED NETS + TAU THRESHOLDS
                "mic_previous_performance": r"./MIC_res/adult", # PREVIOUS FEA(MIC)
                "mic_df_save_path": r"./processed_data/adult/mic_result_ds", # DF PRODUCED BY MIC PHASE
                "r_net_path": r"./r_nets/adult", #R-NETS FOR ANQI MAO DEFERRAL 
                "two_stage_thresholds": r"./r_nets_results/adult",
                "validation_data_save_path": r"./processed_data/adult/validation_data",
                "scaled_mic_batch": r"./processed_data/adult/scaled_batch3",
                "hic_logs": r"./logs/adult/hic",
                "r_net_training_logs": r"./logs/adult/r_net_training",
                "mic_logs": r"./logs/adult/mic",
                "tau_calibration_res": r"./nets/adult"
                
        },

         # directory of the warmed up scaler and incremental learners, loaded at the initial BRIDGET deployment
        "base_obj_paths": {
            'preprocessors': r"./HIC_res/adult/base_objects/preprocessors",
            'incremental_learners': r"./HIC_res/adult/base_objects/incremental_learners"

        }, 
        
        # separate handles for future benchmarking
        "baseline_paths": {
            "trained_experts": r'./baseline/experts/adult',
            "trained_scalers": r'./baseline/scalers/adult',
            "train_df_labeled_by_experts": r"./baseline/mic/labeled_ds/adult/train", 
            "test_df_for_mic": r"./baseline/mic/labeled_ds/adult/test",  
            "validation_set_labeled": r"./baseline/mic/labeled_ds/adult/val",

            # two stage deferral strat
            "calibrated_r_nets":  r"./baseline/mic/r_nets/adult",
            "trained_nets": r"./baseline/mic/nets/adult",
            "two_stage_thresh_baseline": r"./baseline/mic/thresholds/adult",

            # benchmarking strats
            "benchmark_res": r"./baseline/mic/adult",        
            "logs": r"./baseline/logs/adult"
        },
        
    

        # Human in Command phase setup
        "allocated_budget": [3584, 3584, 3584], # 1/3 of the total
        "batches_offset": [0, 3584, 7168, 10752],
        "skepticism_threshold": 0.2,
        "start_performance": 63,  #a default value for the first iteration
        

        # Deferral Nets calibration
        "net_params": {
            "lr": 0.001,
            "weight_decay": 1e-2,
            "label_smoothing": 0.00,
    
          },

        # Machine in Command phase setup
        "benchmark_performance": 63, #a default value for the first iteration
        'performance_delta': 0.05,
        'stricter_delta': 0.03,
        'warm_up': 1075, #warm up given for the MiC portion, obtained as 20% of the test set (thus 20% of 5376)
        'belief_threshold':0.7,
    }
 }




NET_CONFIGS = {
     # for saving purposes
    "small": {
        "layers":[16, 8],
        "step_size": 15,
        "gamma":0.8
        },

    "medium": {
        "layers":[32, 16],
        "step_size": 10,
        "gamma":0.8
    },

    "large": {
        "layers":[64, 32],
        "step_size": 8,
        "gamma":0.8
    },

    "xl": {
        "layers":[128, 64],
        "step_size": 8,
        "gamma":0.8
    },

    "patience": 3,
    "max_epochs": 20,
    "output_size": 2
    
}

R_NET_CONFIGS = {
    "architecture": [16, 8],
    "lr": 1e-3,
    "dropout": 0.2,
    "alpha": 1.0,
    "betas": [0.1, 0.3, 0.5, 0.7, 0.9], # initial wider inference cost grid
    "linspace_dimension": 300, #linspace search parameters to print deferral thresholds and performances (informative only)
    "lower_thresh":0.0,
    "upper_thresh": 0.8,
    "output_size": 1, # output neuron of the rejector, probability of defer
    "defer_rate_low": 0.14,
    "defer_rate_upp": 0.25
   
}

FRANK_RULES= {
        "RULE" : False,
        "PAST" : False,
        "SKEPT" : True,
        "GROUP" : True,
        "EVA":  True
}


