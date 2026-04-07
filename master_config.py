
DATASETS = {
    "dutch": {
         
        "target": "occupation",
        "yaml_prefix": "experts_dutch",
        
        # ds info
        "protected": ['sex', 'age', 'citizenship'],
        "rule_att" : 'edu_level', #random rule
        "rule_value" : 0.1,
        
        "paths": { 
                "dataset_path": r"./datasets/dutch.csv", #DS 
                "user_obj_path": r"./trained_experts/dutch", # USER 
                "trained_preprocessor": r"./HIC_res/dutch", # PREPROCESSOR TRAINED BY THE HIC PHASE 
                "incremental_learner": r"./HIC_res/dutch", #INCREMENTAL LEARNER TRAINED BY THE HIC PHASE 

                "hic_df_save_path": r"./processed_data/dutch/hic_switch_ds", #DF SWITCH (PRODUCED BY HIC PHASE)
                "hic_previous_performance": r"./HIC_res/dutch", # PREVIOUS FEA(HIC)
                "def_net_save_path": r"./nets/dutch",   #SAVE DIR FOR TRAINED NETS + TAU THRESHOLDS
                "mic_previous_performance": r"./MIC_res/dutch", # PREVIOUS FEA(MIC)
                "mic_df_save_path": r"./processed_data/dutch/mic_result_ds", # DF PRODUCED BY MIC PHASE
                "r_net_path": r"./r_nets/dutch", #R-NETS FOR ANQI MAO DEFERRAL 
                "anqi_mao_thresholds": r"./r_nets_results/dutch",
                "validation_data_save_path": r"./processed_data/dutch/validation_data",
                "scaled_mic_batch": r"./processed_data/dutch/scaled_batch3",
                "hic_logs": r"./logs/dutch/hic",
                "r_net_training_logs": r"./logs/dutch/r_net_training",
                "mic_logs": r"./logs/dutch/mic",
                "tau_calibration_res": r"./nets/dutch"
                
        },

        "base_obj_paths": {
            'preprocessors': r"./HIC_res/dutch/base_objects/preprocessors",
            'incremental_learners': r"./HIC_res/dutch/base_objects/incremental_learners"

        }, 
        
        "baseline_paths": {
            "trained_experts": r'./baseline/experts/dutch',
            "trained_scalers": r'./baseline/scalers/dutch',
            "train_df_labeled_by_experts": r"./baseline/mic/labeled_ds/dutch/train", #this one is universal across all baseline strats
            "test_df_for_mic": r"./baseline/mic/labeled_ds/dutch/test",  #this one is universal across all baseline strats
            "validation_set_labeled": r"./baseline/mic/labeled_ds/dutch/val",

            # two stage deferral strat
            "calibrated_r_nets":  r"./baseline/mic/r_nets/dutch",
            "trained_nets": r"./baseline/mic/nets/dutch",
            "anqi_mao_thresh_baseline": r"./baseline/mic/thresholds/dutch",

            # benchmarking strats
            "benchmark_res": r"./baseline/mic/dutch",        
            "logs": r"./baseline/logs/dutch"
        },
        
        
        
        # HUMAN IN COMMAND PARAMS
        "allocated_budget": [2459, 2459, 2458],
        "batches_offset": [0, 2459, 4918, 7376],
        "skepticism_threshold": 0.5,
        "start_performance": 63,
        

        #NN calibration
        "net_params": {
            "lr": 0.001,
            "weight_decay": 1e-2,
            "label_smoothing": 0.00, 
            
        },
        
        # MACHINE IN COMMAND PARAMS
        "benchmark_performance": 65,
        'performance_delta': 0.05,
        'stricter_delta': 0.03,
        'warm_up': 922, #25% of 3688
        'belief_threshold':0.7,
        
    
    },


    "compas": { 
        "target": "did_recid",
        "yaml_prefix": "experts_compas",

        # ds info
        "protected": ['race', 'age', 'sex'],
        "rule_att" : 'priors_count', #random rule
        "rule_value": -0.7,
                

        "paths": { 
                "dataset_path": r"./datasets/compas.csv", #DS 
                "user_obj_path": r"./trained_experts/compas", # USER 
                "trained_preprocessor": r"./HIC_res/compas", # PREPROCESSOR TRAINED BY THE HIC PHASE 
                "incremental_learner": r"./HIC_res/compas", #INCREMENTAL LEARNER TRAINED BY THE HIC PHASE 

                "hic_df_save_path": r"./processed_data/compas/hic_switch_ds", #DF SWITCH (PRODUCED BY HIC PHASE)
                "hic_previous_performance": r"./HIC_res/compas", # PREVIOUS FEA(HIC)
                "def_net_save_path": r"./nets/compas",   #SAVE DIR FOR TRAINED NETS + TAU THRESHOLDS
                "mic_previous_performance": r"./MIC_res/compas", # PREVIOUS FEA(MIC)
                "mic_df_save_path": r"./processed_data/compas/mic_result_ds", # DF PRODUCED BY MIC PHASE
                "r_net_path": r"./r_nets/compas", #R-NETS FOR ANQI MAO DEFERRAL 
                "anqi_mao_thresholds": r"./r_nets_results/compas",
                "validation_data_save_path": r"./processed_data/compas/validation_data",
                "scaled_mic_batch": r"./processed_data/compas/scaled_batch3",
                "hic_logs": r"./logs/compas/hic",
                "r_net_training_logs": r"./logs/compas/r_net_training",
                "mic_logs": r"./logs/compas/mic",
                "tau_calibration_res": r"./nets/compas"
                
        },

        "base_obj_paths": {
            'preprocessors': r"./HIC_res/compas/base_objects/preprocessors",
            'incremental_learners': r"./HIC_res/compas/base_objects/incremental_learners"

        }, 
        
        "baseline_paths": {
            "trained_experts": r'./baseline/experts/compas',
            "trained_scalers": r'./baseline/scalers/compas',
            "train_df_labeled_by_experts": r"./baseline/mic/labeled_ds/compas/train", #this one is universal across all baseline strats
            "test_df_for_mic": r"./baseline/mic/labeled_ds/compas/test",  #this one is universal across all baseline strats
            "validation_set_labeled": r"./baseline/mic/labeled_ds/compas/val",

            # two stage deferral strat
            "calibrated_r_nets":  r"./baseline/mic/r_nets/compas",
            "trained_nets": r"./baseline/mic/nets/compas",
            "anqi_mao_thresh_baseline": r"./baseline/mic/thresholds/compas",

            # benchmarking strats
            "benchmark_res": r"./baseline/mic/compas",        
            "logs": r"./baseline/logs/compas"
        },
        
        


        # hic phase config
        "allocated_budget": [722, 722, 723],
        "batches_offset": [0, 722, 1444, 2167],
        "skepticism_threshold": 0.2,
        "start_performance": 63,  #just a default value for the first iteration
        

        #NN calibration
        "net_params": {
            "lr": 0.001,
            "weight_decay": 1e-3,
            "label_smoothing": 0.00,
    
          },

        # MACHINE IN COMMAND PARAMS
        "benchmark_performance": 63,
        'performance_delta': 0.05,
        'stricter_delta': 0.03,
        'warm_up': 271,
        'belief_threshold':0.7,
    },

    
    "adult": {
        "target": "class",
        "yaml_prefix": "experts_adult",

        # ds info
        "rule_att" : 'capital-gain', 
        "rule_value" : 0.9, # viene scalata anyways
        "protected":['race', 'age', 'sex'],
        

        "paths": { 
                "dataset_path": r"./datasets/adult_clean.csv", #DS 
                "user_obj_path": r"./trained_experts/adult", # USER 
                "trained_preprocessor": r"./HIC_res/adult", # PREPROCESSOR TRAINED BY THE HIC PHASE 
                "incremental_learner": r"./HIC_res/adult", #INCREMENTAL LEARNER TRAINED BY THE HIC PHASE 

                "hic_df_save_path": r"./processed_data/adult/hic_switch_ds", #DF SWITCH (PRODUCED BY HIC PHASE)
                "hic_previous_performance": r"./HIC_res/adult", # PREVIOUS FEA(HIC)
                "def_net_save_path": r"./nets/adult",   #SAVE DIR FOR TRAINED NETS + TAU THRESHOLDS
                "mic_previous_performance": r"./MIC_res/adult", # PREVIOUS FEA(MIC)
                "mic_df_save_path": r"./processed_data/adult/mic_result_ds", # DF PRODUCED BY MIC PHASE
                "r_net_path": r"./r_nets/adult", #R-NETS FOR ANQI MAO DEFERRAL 
                "anqi_mao_thresholds": r"./r_nets_results/adult",
                "validation_data_save_path": r"./processed_data/adult/validation_data",
                "scaled_mic_batch": r"./processed_data/adult/scaled_batch3",
                "hic_logs": r"./logs/adult/hic",
                "r_net_training_logs": r"./logs/adult/r_net_training",
                "mic_logs": r"./logs/adult/mic",
                "tau_calibration_res": r"./nets/adult"
                
        },

        "base_obj_paths": {
            'preprocessors': r"./HIC_res/adult/base_objects/preprocessors",
            'incremental_learners': r"./HIC_res/adult/base_objects/incremental_learners"

        }, 
        
        "baseline_paths": {
            "trained_experts": r'./baseline/experts/adult',
            "trained_scalers": r'./baseline/scalers/adult',
            "train_df_labeled_by_experts": r"./baseline/mic/labeled_ds/adult/train", #this one is universal across all baseline strats
            "test_df_for_mic": r"./baseline/mic/labeled_ds/adult/test",  #this one is universal across all baseline strats
            "validation_set_labeled": r"./baseline/mic/labeled_ds/adult/val",

            # two stage deferral strat
            "calibrated_r_nets":  r"./baseline/mic/r_nets/adult",
            "trained_nets": r"./baseline/mic/nets/adult",
            "anqi_mao_thresh_baseline": r"./baseline/mic/thresholds/adult",

            # benchmarking strats
            "benchmark_res": r"./baseline/mic/adult",        
            "logs": r"./baseline/logs/adult"
        },
        
    

        # hic phase config
        "allocated_budget": [3584, 3584, 3584], # 1/3 of the total
        "batches_offset": [0, 3584, 7168, 10752],
        "skepticism_threshold": 0.2,
        "start_performance": 63,  #just a default value for the first iteration
        

        #NN calibration
        "net_params": {
            "lr": 0.001,
            "weight_decay": 1e-2,
            "label_smoothing": 0.00,
    
          },

        # MACHINE IN COMMAND PARAMS
        "benchmark_performance": 63,
        'performance_delta': 0.05,
        'stricter_delta': 0.03,
        'warm_up': 1344, #warm up given for the MiC portion, obtained as 25% of the test set (thus 25% of 5376)
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
    "output_size": 2,
    #"benchmark_outputs": 3
    #"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

R_NET_CONFIGS = {
    "architecture": [16, 8],
    "lr": 1e-3,
    "dropout": 0.2,
    "alpha": 1.0,
    "betas": [0.1, 0.3, 0.5, 0.7, 0.9],
    "linspace_dimension": 300,
    "lower_thresh":0.0,
    "upper_thresh": 0.8,
    "output_size": 1,
    "defer_rate_low": 0.14,
    "defer_rate_upp": 0.25
   
}

FRANK_RULES= {
        "RULE" : False,
        "PAST" : True,
        "SKEPT" : True,
        "GROUP" : True,
        "EVA":  True,
        "N_BINS": 10,
        "N_VAR" : 3,
        "MAXC" : 5
}


