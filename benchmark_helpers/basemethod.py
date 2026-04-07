from abc import ABC, abstractmethod
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import argparse
import os
import random
import shutil
import time
import torch.utils.data as data
import sys
import pickle
import logging
from tqdm import tqdm
sys.path.append("..")
from utils import *
from metrics import *


class BaseMethod(ABC):
    """Abstract method for learning to defer methods"""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """this function should fit the model and be enough to evaluate the model"""
        pass

    def fit_hyperparam(self, *args, **kwargs):
        """This is an optional method that fits and optimizes hyperparameters over a validation set"""
        return self.fit(*args, **kwargs)

    @abstractmethod
    def test(self, dataloader):
        """this function should return a dict with the following keys:
        'defers': deferred binary predictions
        'preds':  classifier predictions
        'labels': labels
        'hum_preds': human predictions
        'rej_score': a real score for the rejector, the higher the more likely to be rejected
        'class_probs': probability of the classifier for each class (can be scores as well)
        """
        pass


class BaseSurrogateMethod(BaseMethod):
    """Abstract method for learning to defer methods based on a surrogate model"""

    def __init__(self, alpha, plotting_interval, model, device, learnable_threshold_rej = False):
        '''
        alpha: hyperparameter for surrogate loss 
        plotting_interval (int): used for plotting model training in fit_epoch
        model (pytorch model): model used for surrogate
        device: cuda device or cpu
        learnable_threshold_rej (bool): whether to learn a treshold on the reject score (applicable to RealizableSurrogate only)
        '''
        self.alpha = alpha
        self.plotting_interval = plotting_interval
        self.model = model
        self.device = device
        self.threshold_rej = 0.7 #it was 0
        self.learnable_threshold_rej = learnable_threshold_rej 

    @abstractmethod
    def surrogate_loss_function(self, outputs, hum_preds, data_y):
        """surrogate loss function"""
        pass



    def fit_epoch(self, dataloader, optimizer, log, save_dir=None,verbose=False, epoch=1):
        """
        Fit the model for one epoch
        model: model to be trained
        dataloader: dataloader
        optimizer: optimizer
        verbose: print loss
        epoch: epoch number
        """
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        self.model.train()

        log.info(f"Starting fit.epoch")
        for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            hum_preds = hum_preds.to(self.device)
            outputs = self.model(data_x)
            loss = self.surrogate_loss_function(outputs, hum_preds, data_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if torch.isnan(loss):
                print("Nan loss")
                log.warning(f"NAN LOSS")
                if save_dir:
                    torch.save(self.model.state_dict(), os.path.join(save_dir, "crash_model.pt"))
                break

            if verbose and batch % self.plotting_interval == 0:
                log.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )
        if save_dir is not None:
            checkpoint_path = os.path.join(save_dir, 
                                           f"benchnet_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses.avg,
            }, checkpoint_path)
            log.info(f"Saved model checkpoint to {checkpoint_path}")

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        log,
        save_data_dir= None,
        method_name= None,
        scheduler=None,
        verbose=True,
        test_interval=5,
    ):
        
        log.info("Starting .fit method")
        optimizer = optimizer(self.model.parameters(), lr=lr)

        try:
            scheduler = scheduler(optimizer) 
        except TypeError:
            scheduler = scheduler(optimizer, len(dataloader_train) * epochs)
        best_acc = 0
        # store current model dict
        best_model = copy.deepcopy(self.model.state_dict())

        net_dir = None
        if save_data_dir is not None:
            net_dir = os.path.join(save_data_dir, "benchmark_nets", str(method_name))
            os.makedirs(net_dir, exist_ok=True)

        for epoch in tqdm(range(epochs)):
            self.fit_epoch(dataloader_train, optimizer, log, net_dir, verbose, epoch)
            if epoch % test_interval == 0 and epoch > 1 :
                
                if self.learnable_threshold_rej:
                    log.info("fitting rejection threshold")
                    self.fit_treshold_rej(dataloader_val)

                data_test = self.test(dataloader_val) 
                
                if save_data_dir is not None:
                    debug_path = os.path.join(save_data_dir, f"DEBUG_{method_name}_val_pre_metrics.parquet")
                    data_test.to_parquet(debug_path)
                    log.info(f"Saved debug data to {debug_path}")
                
                val_metrics = compute_deferral_metrics(data_test)

                # dictionary with metric:accuracy value
                log.info(f"Validation metrics at epoch {epoch}: {val_metrics}")

                if val_metrics["system_acc"] >= best_acc:
                    # setting new system acc benchmark accuracy 
                    best_acc = val_metrics["system_acc"]
                    best_model = copy.deepcopy(self.model.state_dict())

                if verbose:
                    logging.info(compute_deferral_metrics(data_test))

            if scheduler is not None:
                scheduler.step()
                log.info(f"Updated scheduler for learning rate, epoch {epoch}, lr={scheduler.get_last_lr()[0]}")
        self.model.load_state_dict(best_model)

        if self.learnable_threshold_rej:
            self.fit_treshold_rej(dataloader_val)

        log.info(f"Best validation system acc: {best_acc}")
        
        final_test = self.test(dataloader_test, log=log)
       
        final_test_metrics= compute_deferral_metrics(final_test)


        # save results
        if save_data_dir is not None:
            os.makedirs(save_data_dir, exist_ok=True)

            test_path= os.path.join(save_data_dir,
                                            f"benchmark_results", 
                                            f"{method_name}_test_data.parquet")
            test_df= pd.DataFrame(final_test)
            test_df.to_parquet(test_path, index=False)       

            metrics_path= os.path.join(save_data_dir, 
                                            f"benchmark_results",
                                            f"{method_name}_deferral_metrics.parquet")
            metrics_df=pd.DataFrame([final_test_metrics])
            metrics_df.to_parquet(metrics_path, index=False)

            validation_path= os.path.join(save_data_dir, 
                                            f"benchmark_results",
                                            f"{method_name}_validation_data.parquet")
            validation_df=pd.DataFrame([final_test_metrics])
            validation_df.to_parquet(validation_path, index=False)

        log.info(f"Final test metrics saved to: {metrics_path}, test data saved to: {test_path}, validation data saved to: {validation_path}")
        return final_test_metrics



    def fit_treshold_rej(self, dataloader): #only for realizable surrogate 
        
        data_test = self.test(dataloader)
        rej_scores = np.unique(data_test["rej_score"])
        
        # sort by rejection score
        # get the 100 quantiles for rejection scores
        rej_scores_quantiles = np.quantile(rej_scores, np.linspace(0, 1, 100))
        # for each quantile, get the coverage and accuracy by getting a new deferral decision
        all_metrics = []
        best_treshold = 0
        best_accuracy = 0
        for q in rej_scores_quantiles:
            # get deferral decision
            defers = (data_test["rej_score"] > q).astype(int)
            copy_data = copy.deepcopy(data_test)
            copy_data["defers"] = defers
            # compute metrics
            metrics = compute_deferral_metrics(copy_data)
            if metrics['system_acc'] > best_accuracy:
                best_accuracy = metrics['system_acc']
                best_treshold = q
        self.threshold_rej = best_treshold
        
    
    def test(self, dataloader, log=None):
        """
        Test the model
        dataloader: dataloader
        """
        defers_all = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability

        # once you initialize all needed structures, set model to eval and begin iterating over data loader
        self.model.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds) in enumerate(dataloader): 
                # this means the data loader used must return the hum preds separately
                data_x = data_x.to(self.device) # moving to device
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device) 

                outputs = self.model(data_x) # obtain logits
                outputs_class = F.softmax(outputs[:, :-1], dim=1) # apply softmax to classes only
                outputs = F.softmax(outputs, dim=1) # apply softmax to all 
                _, predicted = torch.max(outputs.data, 1)
                max_probs, predicted_class = torch.max(outputs.data[:, :-1], 1)
                predictions_all.extend(predicted_class.cpu().numpy())
                
                defer_scores = [ outputs.data[i][-1].item() - outputs.data[i][predicted_class[i]].item() for i in range(len(outputs.data))]
                
                # TEMPORARY DEBUG PRINT
                defer_binary = [int(defer_score > self.threshold_rej) for defer_score in defer_scores]
                #defer_binary = [int(defer_score >= self.threshold_rej) for defer_score in defer_scores]
                if log is not None:
                    
                    log.info(f"DEBUG: Batch Max Score: {max(defer_scores):.4f}, threshold: {self.threshold_rej}, items_deferred: {sum(defer_binary)}/128")
               
                defers_all.extend(defer_binary)
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())

                for i in range(len(outputs.data)):
                    rej_score_all.append(
                        outputs.data[i][-1].item()
                        - outputs.data[i][predicted_class[i]].item()
                    )
                class_probs_all.extend(outputs_class.cpu().numpy())

        # convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        rej_score_all = np.array(rej_score_all)
        class_probs_all = np.array(class_probs_all)
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "hum_preds": hum_preds_all,
            "preds": predictions_all,
            "rej_score": rej_score_all,
            "class_probs": class_probs_all,
        }

        if log is not None:
            log.info(f"DEBUG: Final Data 'defers' shape: {data['defers'].shape}, sum: {data['defers'].sum()}")
        return data
        
