## BENCHMARKING UTILS
import os 
import sys
import torch
import glob
import pandas as pd
import numpy as np
import sklearn.metrics
import copy
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR

current_dir = os.getcwd()
repo_path = os.path.join(current_dir, "hdms-essai24", "L2Dcode")
sys.path.append(repo_path)

# 2. Imports (Matching the sidebar exactly)
from lce_surrogate import LceSurrogate
from basemethod import *

from realizable_surrogate import RealizableSurrogate
from baselines.compare_confidence import CompareConfidence
from baselines.differentiable_triage import DifferentiableTriage
from baselines.selective_prediction import SelectivePrediction
from baselines.one_v_all import OVASurrogate
from baselines.mix_of_exps import MixtureOfExperts

from metrics import *
from utils import *
from basenet import BenchmarkNet
from master_config import DATASETS, NET_CONFIGS
from bridget_utils import custom_log, create_loader
from functools import partial
import helpers.metrics
import sklearn.metrics
import numpy as np


def load_net(input_size, layers, classes, device):
    #classes being output size + 1
    net= BenchmarkNet(input_size=input_size, 
                    hidden_layer1=layers[0],
                    hidden_layer2=layers[1],
                    output_size=classes,
                    dropout_coeff=0.0
    )

    for param in net.parameters():
        param.requires_grad = True
    
    """
    #doing the same thing the f does in the og ipynb (forcing classes+1) at the output layer
    in_features = net.linear_relu_stack[6].in_features
    net.linear_relu_stack[6] = nn.Linear(in_features, classes).to(device)
    
    with torch.no_grad():
        net.linear_relu_stack[6].bias[-1].fill_(0.1)
    """
    net.to(device)
    net.train()
    return net



def get_l2d_results_and_metrics(method, 
                                input_size,
                                layers,
                                n_classes, 
                                test_data,
                                val_data,
                                train_data,
                                device, 
                                ds_name,
                                target,
                                total_epochs,
                                lr,
                                step_size,
                                weight_decay,
                                gamma, 
                                features,
                                human_preds
                                ):
  
    """method (str): L2D algorithm to use. Either of: {'MixtureOfExperts', 'CrossEntropy', 'OneVsAll', 'Realizable', 'CompareConfidence', 'DifferentiableTriage', 'SelectivePrediction'}
        model_fn: name of function to initialize the nn model
        n_classes (int): cardinality of label space
    """

    print('Computing L2D algorithm with method: ', method)
    print('Retrieving Neural Network(s) models')

    # initialize the one or two nn
    if method in ['MixtureOfExperts', 'CrossEntropy', 'OneVsAll', 'Realizable']:
        model_1 = load_net(input_size, layers, n_classes+1, device)
    elif method == 'SelectivePrediction':
        model_1 = load_net(input_size, layers, n_classes, device)
    elif method in ['CompareConfidence','DifferentiableTriage']:
        model_1 = load_net(input_size, layers, n_classes, device) # classifier
        model_2 = load_net(input_size, layers, 2, device) # rejector
    else:
        print('Unknown method: ', method)
        return

    print('Initializing the L2D algorithm')
    if method == 'MixtureOfExperts':
        l2d_algo = MixtureOfExperts(model_1, device)
    elif method == 'CrossEntropy':
        l2d_algo = LceSurrogate(1, 300, model_1, device)
    elif method == 'OneVsAll':
        l2d_algo = OVASurrogate(1, 300, model_1, device)
    elif method == 'Realizable':
        l2d_algo = RealizableSurrogate(1, 300, model_1, device, learnable_threshold_rej=True)
    elif method == 'CompareConfidence':
        l2d_algo = CompareConfidence(model_1, model_2, device)
    elif method == 'DifferentiableTriage':
        l2d_algo = DifferentiableTriage(model_1, model_2, device, 0.000, "human_error")
    elif method == 'SelectivePrediction':
        l2d_algo = SelectivePrediction(model_1, device)
    else:
        print('Unknown method: ', method)
        return

    # retrieving test loader
    _, _, test_loader= create_loader(test_data, target=target, features=features, human_preds= human_preds, batch_size=32)
    _, _, train_loader= create_loader(train_data, target=target, features=features, human_preds= human_preds, batch_size=32)
    _, _, val_loader= create_loader(val_data, target=target, features=features, human_preds= human_preds, batch_size=32)

    print('Fitting the model')
    log_path= os.path.join(DATASETS[ds_name]['baseline_paths']['logs'], 
                           f"fitting",
                           f"{method}.log")
    log= custom_log(_, log_path)

    optimizer_ptr = partial(AdamW, lr=lr, weight_decay=weight_decay)
    scheduler_ptr= partial(StepLR, step_size=step_size, gamma=gamma)
   
    # now, inside this fit method we gotta: save the net and save the dataset at each iter so we can monitor where it crashes
    # so since all of them call basemethods, include it there
    l2d_algo.fit_hyperparam(
        train_loader,
        val_loader,
        test_loader,
        epochs=total_epochs,
        optimizer=optimizer_ptr,
        scheduler=scheduler_ptr,
        lr=lr,
        log= log,
        verbose=False,
        test_interval=5,
    )

    print('Computing predictions on the test set and corresponding metrics:')
    
    # compute prediction on the test dataset
    l2d_test = l2d_algo.test(test_loader, log=log)

    # compute deferral metrics (but its already computed above)
    l2d_def_metrics = compute_deferral_metrics(l2d_test) # no log currently, its inside the test loader

    # compute classifier metrics
    l2d_clf_metrics = compute_classification_metrics(l2d_test, ds_name=ds_name, method=method)

    # compute_deferral_metrics(data_test_modified) on different coverage levels, first element of list is compute_deferral_metrics(data_test)
    l2d_coverage_metrics = compute_coverage_v_acc_curve(l2d_test, method=method, ds_name=ds_name)

    return l2d_algo, l2d_test, l2d_def_metrics, l2d_clf_metrics, l2d_coverage_metrics



def run_benchmark(ds_name, users_lst, method, layers, architecture, device, params):
        bench_res= {}

        log_dir = DATASETS[ds_name]['baseline_paths']['logs']
        target=DATASETS[ds_name]['target'] 
        features=params['feature_order']
        net_input_size= params['n_cols']
        human_preds= params['human_preds_col_name']
        weight_decay=DATASETS[ds_name]['net_params']['weight_decay']
        step_size=NET_CONFIGS[architecture]['step_size']
        gamma=NET_CONFIGS[architecture]['gamma']
        lr= DATASETS[ds_name]['net_params']['lr']
        
    
        
        for name in users_lst:
            log_name = os.path.join(log_dir, f"{name}.log")
            bench_log = custom_log(name, log_name)

            bench_res[name] = {method: {}}

            #################### RETRIEVING OBJECTS
            bench_log.info(f"STARTING BENCHMARK RUN | METHOD: {method}")

            #we're gonna train the net from scratch inside the fit method
            # so no need to retrieve it
            
            """"
            # 1. NET PATHS

            net_path= DATASETS[ds_name]['baseline_paths']['trained_nets']
            net_path= os.path.join(net_path,
                            f"{name}_models",
                            f"{layers[0]}_{layers[1]}")
            pattern = f"{layers[0]}_{layers[1]}_model_*.pt"
            search_query = os.path.join(net_path, pattern)
            matches = glob.glob(search_query)

            if matches:
                    net_path = max(matches, key=os.path.getctime)
            """

            # 2. TEST DATA
            test_data_path= DATASETS[ds_name]['baseline_paths']['test_df_for_mic']
            test_data_path= os.path.join(test_data_path, 
                                    f"test_baseline_{name}.parquet")
            test_d= pd.read_parquet(test_data_path)

            bench_log.info("Retrieved necessary assets (net, test, val, train)")


            # 3. TRAIN DATA

            train_data_path= DATASETS[ds_name]['baseline_paths']['train_df_labeled_by_experts']
            train_data_path= os.path.join(train_data_path, 
                                    f"train_baseline_{name}.parquet")
            train_d= pd.read_parquet(train_data_path)

            # 4. VAL DATA
            val_data_path= DATASETS[ds_name]['baseline_paths']['validation_set_labeled']
            val_data_path= os.path.join(val_data_path, 
                                    f"val_baseline_{name}.parquet")
            val_d= pd.read_parquet(val_data_path)


            # 5. BENCHMARKING 
            _, _, deferral_metrics, clf_metrics, coverage_metrics= get_l2d_results_and_metrics(method= method, 
                                    input_size= net_input_size,
                                    layers=layers,
                                    n_classes=NET_CONFIGS['output_size']+1, 
                                    step_size= step_size,
                                    gamma=gamma,
                                    test_data= test_d,
                                    train_data= train_d,
                                    val_data= val_d,
                                    ds_name=ds_name,
                                    weight_decay=weight_decay,
                                    lr= lr,
                                    total_epochs=500,
                                    device= device, 
                                    target=target,
                                    features=features,
                                    human_preds=human_preds
                                    )
            
     
            bench_res[name][method]['deferral_metrics']= deferral_metrics
            bench_res[name][method]['clf_metrics']= clf_metrics
            bench_res[name][method]['coverage_metrics']= coverage_metrics

            # SAVING RESULTS

            base_dir = DATASETS[ds_name]['baseline_paths']['benchmark_res']
            folder = os.path.join(base_dir, method)
            os.makedirs(folder, exist_ok=True)
        
            file_save_path = os.path.join(folder, f"res_{name}.parquet")
            
            results_df = pd.DataFrame([bench_res[name][method]])
            results_df.to_parquet(file_save_path, index=False)
            
            bench_log.info(f"BENCHMARKING WITH {method} FINISHED! | Res in {file_save_path}")