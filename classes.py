### --- This is the first test. Objective: create classes for the Users to be employed in both HiC and MiC
from bridget_utils import *
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
import copy
import math
from pyexpat import model
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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


# -- Di seguito manteniamo la classe principale User di FRANK, che rappresenta le caratteristiche (in senso di attributi) dell'User
# -- In base al comportamento che l'user deve seguire nella predizione, usiamo sottoclassi come l'implementazione classica di FRANK
def sigmoid(x):
    # Applichiamo i limiti -15 e 15 a tutto il vettore in un colpo solo
    # Questo sostituisce gli "if" che creano il problema
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))


class User:
        def __init__(self, belief_level, rethink_level, fairness):

            self.belief_level= belief_level  # belief level si intende quello della f sotto o quello fisso che definisce l'archetipo?
            self.rethink_level= rethink_level
            self.fairness= fairness

        
        def believe(self, seed=None):
            rng= random.Random(seed) if seed is not None else random
            if self.belief_level > 1:
                return None

            else:
                return rng.choices(population=[True, False], 
                              weights=[self.belief_level, 1-self.belief_level], k=1)[0]
        

        def rethink(self):
            return random.choices(population=[True, False], 
                              weights=[self.rethink_level, 1-self.rethink_level], k=1)[0]
        
        def fairness_percentage(self):
            return self.fairness
        
# -- Sottoclasse BetaUser (in Open L2A calcoliamo i betas)
# -- In sostanza dobbiamo dare la predizione dell'user usando la logica di IDN di Open L2A
# -- basandosi sulla funzione
# -- ho mantenuto separati i target FPR e FNR
# -- perchè la generazione della probabilità d'errore dell'esperto è calcolata separatamente per le due metriche
# -- utilizzando la configurazione di L2D, abbiamo bisogno delle funzioni ausiliare sigmoid and invert probabilities perchè la logica di predizione li richiede


class BetaUser(User):
    
    def __init__(self, 
                 belief_level, rethink_level, fairness, 
                 fpr, fnr,
                 alpha,  # - alpha being the intra-rater agreement
                 features_dict,
                 seed
    ):
        super().__init__(belief_level, rethink_level, fairness)

        self.fpr= fpr
        self.fnr= fnr
        self.alpha= alpha
        self.features_dict= features_dict
        self.seed = seed
     

        # - Then, as per L2D, here are the params set by the fit function

        self.fnr_beta= None
        self.fpr_beta= None
        self.w= None
        #self.error_prob= pd.DataFrame(-1, index=np.arange(1000000), columns=['p_of_fp', 'p_of_fn'])


    # -- SEZIONE 1: Campionamento dei pesi a partire dal features dict, sulla base di mean e std 
        # -- Prepariamo una funzione per il campionamento a partire dalla configurazione yaml con la distribuzione delle features per generare i pesi
  
    def sample_weights(self, X):

        if self.features_dict is None:
            raise ValueError("Features_dict not yet defined")
        
        else:
            np.random.seed(self.seed)
            self.feature_names = X.columns.tolist()
            self.w = np.zeros(X.shape[1])
            for feature in self.features_dict.keys():
                self.w[X.columns.get_loc(feature)] = np.random.normal(
                    loc = self.features_dict[feature][0], scale = self.features_dict[feature][1]
                    )


    
    def search_bounds( self, X, y, error, beta_target):
        
        lower_b, upper_b = -200, 200
       
        for i in range(5):

            #1. False Positives
            if error == 'fp':
                self.fpr_beta = lower_b
                self.calc_probs_fp(X, y)
                mean_lower = float(self.error_prob.loc[y == 0, 'p_of_fp'].mean())

                self.fpr_beta = upper_b
                self.calc_probs_fp(X,y)
                mean_upper = float(self.error_prob.loc[y == 0, 'p_of_fp'].mean()) 


            #2. False Negatives
            elif error == 'fn': # fn research here
                self.fnr_beta= lower_b
                self.calc_probs_fn(X, y)
                mean_lower = float(self.error_prob.loc[y == 1, 'p_of_fn'].mean())

                self.fnr_beta= upper_b
                self.calc_probs_fn(X,y)
                mean_upper = float(self.error_prob.loc[y == 1, 'p_of_fn'].mean()) 


            if (mean_lower - beta_target) * (mean_upper - beta_target) < 0:
                return lower_b, upper_b, mean_lower, mean_upper

            lower_b -= 200
            upper_b += 200
 



    def fit(self, X, y, tol):
        self.sample_weights(X)

        self.error_prob= pd.DataFrame(np.nan, index=X.index, columns=['p_of_fp', 'p_of_fn'])
        

        # 1. Fit per FPR
        fpr_l, fpr_u, fp_mean_lower, fp_mean_upper = self.search_bounds(X, y, 'fp', self.fpr)

        self.fpr_beta= (fpr_l + fpr_u) / 2 
        self.calc_probs_fp(X,y)
        fp_mean= self.error_prob.loc[y == 0, 'p_of_fp'].mean() 

        fpr_iters= 0
        while np.abs(fp_mean - self.fpr) > tol:
            if (fp_mean_lower - self.fpr) * (fp_mean - self.fpr) < 0:
                fpr_u =self.fpr_beta
                fp_mean_upper= fp_mean
            else:
                fpr_l= self.fpr_beta
                fp_mean_lower= fp_mean
            
            self.fpr_beta= (fpr_u + fpr_l)/2
            self.calc_probs_fp(X,y)
            fp_mean= self.error_prob.loc[y == 0, 'p_of_fp'].mean() 
            fpr_iters += 1 ## conteggio solo da mettere nel dict finale



        # 2. Fit di FNR
        fnr_l, fnr_u, fn_mean_lower, fn_mean_upper = self.search_bounds(X, y, 'fn', self.fnr)

        self.fnr_beta= (fnr_l + fnr_u) / 2 
        self.calc_probs_fn(X,y)
        fn_mean= self.error_prob.loc[y == 1, 'p_of_fn'].mean() 


        fnr_iters= 0
        while np.abs(fn_mean - self.fnr) > tol:
            if (fn_mean_lower - self.fnr) * (fn_mean - self.fnr) < 0:
                fnr_u= self.fnr_beta
                fn_mean_upper= fn_mean
            else:
                fnr_l= self.fnr_beta
                fn_mean_lower= fn_mean

            self.fnr_beta= (fnr_u + fnr_l)/2
            self.calc_probs_fn(X,y)
            fn_mean= self.error_prob.loc[y == 1, 'p_of_fn'].mean() 
            fnr_iters+= 1
        

        fit_res = {
            "fpr iters number": fpr_iters,
            "calibrated_fpr_beta": self.fpr_beta,
            "target_fpr": self.fpr,
            "achieved_fpr": fp_mean,

            "fnr iters number": fnr_iters,
            "calibrated_fnr_beta": self.fnr_beta,
            "target_fnr": self.fnr,
            "achieved_fnr": fn_mean,

            "feature_weights": dict(zip(X.columns, self.w)),
        }

        return fit_res
   


    
    def calc_probs_fp(self, X, y, **kwargs):  # kwargs not used (compatibility purposes)
        if self.w is None:
            raise ValueError('Synthetic expert must be .fit() to the data.')
        
        weights = self.w
        dot_prod= X.dot(weights)
        weights_norm= np.linalg.norm(weights)

        term = dot_prod/weights_norm
        
        z_dec = self.fpr_beta + (self.alpha * term)
        probability_of_fp = sigmoid(z_dec)

        mask = (y == 0)
    
        self.error_prob.loc[mask, 'p_of_fp'] = probability_of_fp[mask]

        """
        probability_of_fp = (y == 0) * (
            self.fpr_beta + (self.alpha*(X * weights/(np.linalg.norm(weights))).sum(axis=1)
            )).apply(sigmoid)
        self.error_prob.loc[X.index,'p_of_fp'] = probability_of_fp
        """


    def calc_probs_fn(self, X, y, **kwargs):  # kwargs not used (compatibility purposes)
        if self.w is None:
            raise ValueError('Synthetic expert must be .fit() to the data.')
       
        weights = self.w
        weights_norm= np.linalg.norm(weights)

        dot_prod = X.dot(weights)
        term = dot_prod / weights_norm
        #print(f"DEBUG - Proiezione media: {term.mean():.4f}, DevStd: {term.std():.4f}")

        z_dec= self.fnr_beta - (self.alpha*term)
        probability_of_fn= z_dec.apply(sigmoid)

        mask = (y==1)

        self.error_prob.loc[mask, 'p_of_fn'] = probability_of_fn[mask]



        """
        probability_of_fn = (y == 1) * (
            self.fnr_beta - (self.alpha*(X * weights/(np.linalg.norm(weights))).sum(axis=1)
            )).apply(sigmoid)
    
        self.error_prob.loc[X.index,'p_of_fn'] = probability_of_fn
        """






    def predict(self, record, ground, iter, **kwargs):  # kwargs not used (compatibility purposes)
        
    # -- per adattare la logica di Open L2D alla predizione sequenziale di FRANK bisognava mantenere record e ground da FRANK
    # -- ma cambiare tutta la struttura perchè utilizzava operazioni su vettori
    # -- in sostanza bisognava riscrivere la formula di FPR e FNR 


    # .. edit: siccome non l'ho messo prima
    # come input record deve essere una npy array perchè chiaramente sotto abbiamo np.dot (!!! IMPORTANTISSIMO RICORDA)
    # weights era già preprocessato sopra come npy vector quindi RICORDATI !!! di controllare il formato di record

        if self.w is None:
            raise ValueError('Synthetic expert must be .fit() to the data.')
        
        if isinstance(record, dict):
            record_arr = np.array([record[f] for f in self.feature_names])

        else:
            if torch.is_tensor(record):
                record_arr = record.detach().cpu().numpy()
            else:
                record_arr = np.array(record)

        if record_arr.shape[0] != self.w.shape[0]:
            print(f"Theres a mismatch, feature dict has shape {self.w.shape[0]}, record has {record.shape[0]}")

        
        weights = self.w
        weights_norm= np.linalg.norm(weights) 

        ground_arr= np.array([ground]) # serve esclusivamente per invert labels with probabilities, perchè richiede esplicitamente un arr

        norm_feature_term= np.dot(record_arr, weights)/weights_norm

        if ground == 0:
            fpr_argument= self.fpr_beta + (self.alpha*(norm_feature_term))
            probability_of_fp= sigmoid(fpr_argument)
            probability_of_fn= 0
        
        else:
            fnr_argument= self.fnr_beta - (self.alpha*(norm_feature_term))
            probability_of_fn= sigmoid(fnr_argument)
            probability_of_fp= 0
        
        
        probability_of_error = probability_of_fn + probability_of_fp

        decisions = invert_labels_with_probabilities(
            labels_arr=ground_arr,
            p_arr=probability_of_error,
            seed=self.seed+iter
        )

        return int(decisions[0]) # oppure bool(decisions[0]) se label è True / False





class DeferralNet(nn.Module):

    def __init__(self, input_size, hidden_layer1, hidden_layer2, output_size, dropout_coeff=None):
        super(DeferralNet, self).__init__()

        self.softmax = nn.Softmax(dim=1)


        #self.flatten =nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_layer1),
            nn.ReLU(),
            nn.Dropout(dropout_coeff),

            nn.Linear(hidden_layer1, hidden_layer2),
            nn.ReLU(),
            nn.Dropout(dropout_coeff),
            nn.Linear(hidden_layer2, output_size)
        )
        
    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

    def predict_proba_nn(self, x, device= None):
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(x): # non si sa mai
                x = torch.tensor(x, dtype=torch.float32).to(device)
            
            if x.ndim == 1:
                x = x.unsqueeze(0)
            x= x.to(device)
            logits = self.forward(x)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, x, device=None):
        probas= self.predict_proba_nn(x, device)
        return np.argmax(probas, axis=1)




class RiverModelWrapper(BaseEstimator):
    def __init__(self, river_model, target_column, feature_names=None):
        self.river_model = river_model
        self.target_column = target_column
        self.feature_names = feature_names

    def predict_one(self, x):
        return self.river_model.predict_one(x)
    
    def predict_proba_one(self, x):
        return self.river_model.predict_proba_one(x)
    
    def learn_one(self, x, y):
        self.river_model.learn_one(x, y)

    def fit(self, X, Y):
        if isinstance(X, np.ndarray):
            for x, y in zip(X, Y):
                x = {name: value for name, value in zip(self.feature_names, x)}
                self.river_model.learn_one(x, y)
                
        elif isinstance(X, pd.DataFrame):
            X = X.to_dict(orient='records')
            for x, y in zip(X, Y):
                self.river_model.learn_one(x, y)
        else:
            self.river_model.learn_one(X, Y)
        
        return self
   
    def predict(self, X):
       
        if isinstance(X, pd.DataFrame):
            X = X.to_dict(orient='records')
            preds = [self.river_model.predict_one(x) for x in X]
            return np.array(preds).astype(int)
        
        elif isinstance(X, np.ndarray):
            X = X.squeeze()
           
            
            if X.ndim == 1:
                x_d = [{name: value for name, value in zip(self.feature_names, X)}]
                
                x_dict = {name: value for name, value in zip(self.feature_names, X)}
                pred = self.river_model.predict_one(x_dict)
                # Fondamentale: restituiamo un array di un solo elemento intero

                return np.array([int(pred)])
            else:
                preds = []
                for row in X:
                    x_dict = {name: value for name, value in zip(self.feature_names, row)}
                    preds.append(self.river_model.predict_one(x_dict))
        
                return np.array(preds).astype(int)
        else:
            # Caso singolo dizionario o altro
            pred = self.river_model.predict_one(X)
            return np.array([int(pred)])
    
    def predict_proba(self, X):
      
        probas = []
        
        if isinstance(X, pd.DataFrame): 
            X = X.to_dict(orient='records')
            for x in X:
                
                try:
                   
                    prediction_proba = self.river_model.predict_proba_one(x)
                    probas.append([prediction_proba.get(False, 0), prediction_proba.get(True, 0)])
                except:
                    probas.append([0, 0])
                #print(probas)
                
        elif isinstance(X, np.ndarray):
            
            
            if X.ndim == 1:
                x_d = [{name: value for name, value in zip(self.feature_names, X)}]
                
            else:
                x_d = []
                for row in X:
                    x_d.append({name: value for name, value in zip(self.feature_names, row)})
            
            for x in x_d:
                prediction_proba = self.river_model.predict_proba_one(x)
                try:
                    probas.append([prediction_proba.get(False, 0), prediction_proba.get(True, 0)])
                except:
                    probas.append([0, 0])
        else:
            raise TypeError("Formato di input non supportato: utilizzare DataFrame o array NumPy.")
       
        return np.array(probas)




class PyTorchWrapper(BaseEstimator):  # esclusivamente per la XAI in MiC
    def __init__(self, model, target, features_names):
        self.model = model
        self.target= target
        self.features_names= features_names

    def predict(self, X):
        self.model.eval() 
         
        if isinstance(X, dict):
            X = np.array([[X[f] for f in self.features_names]])

        if isinstance(X, pd.DataFrame):
            X= X[self.features_names].values
        
        X_t = torch.tensor(X, dtype= torch.float32)
        device = next(self.model.parameters()).device
    
        # 3. Move the data to that device (The missing step!)
        X_t = X_t.to(device)

        with torch.no_grad():
            outputs = self.model(X_t)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds


    def predict_proba(self, X):
        self.model.eval() 
        if isinstance(X, dict):
            
            X = np.array([[X[f] for f in self.features_names]])

        if isinstance(X, pd.DataFrame):
             X= X[self.features_names].values
        
        X_t = torch.tensor(X, dtype= torch.float32)

        with torch.no_grad():
            outputs = self.model(X_t)
            probas= torch.softmax(outputs, dim= 1).numpy()

        return probas
    
    
    def predict_one(self, x):

        # ok ragioniamo, io gli passo x come dizionario dal main body
        # gs invece quando lo applico mi passa una numpy array
        self.model.eval() 

        if isinstance(x, dict):
            x_input = np.array([[x[f] for f in self.features_names]], dtype=np.float32)
            # se è un dict come succede, lo trasformiamo in array con dimesione (1, 8) per compas
        else:
            x_input = np.array(x, dtype=np.float32).reshape(-1, 8)  
            # se non è un dict può essere tensore o già numpy array che mi arriva da GS
            # con il reshape se fatto in maniera giusta dovrei ottenere di nuovo (1,8) ma tutto dipende da come è fatto quel -1
            # perchè siccome gli passo come parametro di generazione 200, 200 sono le righe, quindi non può essere 1 il primo termine,
            # ma dovrebbe essere 200

        with torch.no_grad():
            res = self.model(torch.from_numpy(x_input))
        return int(torch.argmax(res, dim=1).item())





class Baseline:

    def __init__(self, 
                 user, #user is already calibrated
                 label,
                 train_set,
                 test_set,
                 val_set,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 x_val,
                 y_val,
                 model= None,
                 strat= None):
        
        ## note, remember that train set and test set must be transformed to tensors anyways

        self.model= model
        self.user= user
        self.label= label
        self.strat= strat

        self.train_set= train_set
        self.test_set= test_set
        self.val_set= val_set
        self.features_names= [c for c in self.train_set if c != self.label]

        self.x_train= x_train
        self.y_train= y_train

        self.x_test= x_test
        self.y_test= y_test

        self.x_val= x_val
        self.y_val= y_val

    

    def fit_expert(self, scaler):
    
        train_scaled= scaler.fit_transform(self.x_train)
        test_scaled= scaler.transform(self.x_test)
        val_scaled= scaler.transform(self.x_val)

        train= pd.DataFrame(train_scaled, columns= self.features_names)
        test= pd.DataFrame(test_scaled, columns= self.features_names)
        val= pd.DataFrame(val_scaled, columns= self.features_names)

        train_user_preds= []
        test_user_preds= []
        val_user_preds= []


        for i in tqdm(range(len(train))):
            x= train.iloc[i]
            y_gt= self.y_train.iloc[i]
            
            user_pred= self.user.predict(x, y_gt, i)
            train_user_preds.append(user_pred)
        

        for i in tqdm(range(len(test))):
            x= test.iloc[i]
            y_gt= self.y_test.iloc[i]
            
            user_pred = self.user.predict(x, y_gt, i)
           
            test_user_preds.append(user_pred)
        
        for i in tqdm(range(len(val))):
            x= val.iloc[i]
            y_gt= self.y_val.iloc[i]
            
            user_pred = self.user.predict(x, y_gt, i)
           
            val_user_preds.append(user_pred)
        
        train_df = self.train_set.copy()
        test_df = self.test_set.copy()
        val_df = self.val_set.copy()

        train_df[self.features_names] = train
        test_df[self.features_names] = test
        val_df[self.features_names] = val

        train_df['expert prediction'] = train_user_preds
        test_df['expert prediction'] = test_user_preds
        val_df['expert prediction'] = val_user_preds

        return train_df, test_df, val_df
    
    
    def fit_model(self, train_loader):
        pass


