"""
========================================
BRIDGET DECISION MAKING ENGINE  
========================================

This module implements the infrastructure of BRIDGET's decision making logic using OOP, with a parent class BRIDGET and two subclasses 
representing the two operational phases that handle their decision logic independently from each other.


Assumptions and Scope
----------------------------------------
..note:: BRIDGET establishes a collaboration with one machine (which is an Incremental Model during the Human in Command streaming phase, and a 
      traditional batch learner in the subsequent Machine in Command) and ONE human. Although operationally it is possible to re-structure the 
      code to a working standard for the multiple expert setting, the intended usage as the first prototype required first and foremost the validation
      of the underlying empirical assumptions.

      Additionally, stress testing the algorithm with a heterogeneous pool of experts (with varying degrees of trust in the machine and expertise, 
      explicited by their False Positive and False Negative Rate) highlighted a direct relation between the severity of the 
      skepticism threshold and the success of the subsequent Deferral Policies caused by the interaction with inaccurate and not trusting users.

      It is necessary, before proceding with expanding the structure towards multiple users, to determine whether systemic degradation at the later iterations
      can be mitigated by adopting alternative skepticism schemes or by systemically raising the threshold over time

      
..note:: The structure is model agnostic. 
        However, Human in Command requires an Incremental Model, while Machine in Command requires two batch learners (predictor/rejector pair) 
        
        For the record, during the experimental validation the following models were used:
        * Synthetic Users : modified OpenL2D generation scheme (reasoning and formulation provided in the full text)
        * Human in Command: Adaptive Random Forest (River)
        * Machine in Command: DNN (the predictor and rejector have different architectures and hyperparams)

        
Architecture
----------------------------------------
    * **BRIDGET: initializes the logging processes and the internal structures necessary in both phases. Handles logging when drift is detected

    * **HIC: adapts the FRANK (Mazzoni et al.) algorithm and integrates it with BRIDGET's refinements listed in the foundational paper, 
       as well as the modifications that arose during my thesis project in collaboration with my supervisors.

       In line with FRANK, 4 decision rules are used to regulate the HAIC co-evolutionary training. 
       In particular, the SLC (Skeptical Learning Check, line) directly exploits the Skeptical Learning paradigm using a set skepticism threshold
       that is compared with the current skepticism level of the machine based on the empirical accuracies and confidence of both agents.

       During SLC, a 1-NN (modulable) record of the same and opposite label is provided to the user to monitor consistency
       

    * **MIC: implements the Machine in Command labeling logic via the Selective Prediction (Mozannar et al., 2020) and 
       Two-Stage Learning to Defer (Mao et al., 2023)

       Counterfactuals are provided using the GrowingSpheres algorithm (plausibility, proximity and sparsity are contextually evaluated and stored)

"""


from river import metrics
import random
import numpy as np
import time
from tqdm import tqdm
from classes import PyTorchWrapper
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
import pickle
import os 
from collections import Counter
from river import metrics
from bridget_utils import *




class BRIDGET:

    """
    Main orchestration class for the BRIDGET framework. Manages data lifecycle, data structure and phase transition.
    
    Args:
        dataset_name (str): identifier for the dataset tested
        user_name (str): identifier for the user archetype
        batch1 (DataFrame): training data for the Human in Command phase, to be co-labeled through the interaction between agents
        batch3 (DataFrame): data allocated towards the Machine in Command phase, subject to deferral
        batch1_test (DataFrame): data portion used to assess the performance of the Human in Command phase through Accuracy, F1 score 
        target (str): name of the target label (defined in master_config.py, DATASETS)
        user_model (object, BetaUser): simulated human expert engine, modeled using a reduced version of OpenL2D
        protected (list): list of protected attributes features (defined in master_config.py, DATASETS)
        cats (list); list of names of the categorical features
        num (list): list of names of the numerical features
        preprocessor (object, optional): River scaler
        training_iter (int): number of current training iteration, used for logging and checkpointing purposes
        log (object, optional): custom log, see bridget_utils.py for the definition

    Methods:
        __init__: initializes data components and structures needed in the subsequent phases
        switch_phase: checkpoints and compiles logs upon conceptual drift

    """
    def __init__(self, 
                 dataset_name, 
                 user_name, 
                 batch1, 
                 batch3, 
                 batch1_test, 
                 target, 
                 user_model, 
                 protected, 
                 cats, 
                 num,
                 preprocessor=None,
                 training_iter=1,
                 log= None):
        
        self.dataset_name= dataset_name
        self.name= user_name
        self.user_model = user_model 
        self.preprocessor= preprocessor
        self.training_iter = training_iter
        self.log= log

        # defining core data components
        self.df_batch1 = batch1 
        self.df_batch3= batch3
        
        self.target = target 
        self.feature_names = [c for c in self.df_batch1 if c!=self.target]
        self.attr_list = list(batch1.columns) 
        self.protected= protected  
        self.protected_values=batch1[protected[0]].unique()

        self.cats = cats
        self.num= num
        
        # slicing over batch1 and the corresponding test section
        self.X = self.df_batch1[self.feature_names]
        self.X = list(self.X.to_dict(orient='index').values())
        self.Y = list(self.df_batch1[target])
        self.Y = [int(y) for y in self.Y] 
        
        self.X_test1= batch1_test[self.feature_names]  
        self.X_test1= list(self.X_test1.to_dict(orient= 'index').values())
        self.Y_test1= list(batch1_test[target])
        self.Y_test1 = [int(y) for y in self.Y_test1]

        self.train_check = False #boolean check that becomes TRUE once the incremental learner is trained
              
        
        # initializing storing structure to calculate agent-wise empirical accuracies for the classes
        self.stats = dict()
        self.stats[False] = dict()
        self.stats[True] = dict()
        for e in ['user', 'machine']:
            self.stats[False][e] = dict()
            self.stats[True][e] = dict()
            self.stats[False][e]['tried'] = 0
            self.stats[True][e]['tried'] = 0
            self.stats[False][e]['got'] = 0
            self.stats[True][e]['got'] = 0
           
            if e == 'user':
                self.stats[False][e]['conf'] = 1
                self.stats[True][e]['conf'] = 1
            else:
                self.stats[False][e]['conf'] = 0
                self.stats[True][e]['conf'] = 0

       

    def switch_phase(self, drift_detected, current_phase, current_log, strat=None):
        """
        Checkpoints the log structure in case of drift detection, saving it in a specific folder

        Attributes:
            drift_detected (bool): Boolean value resulting from the drift detection function called within the phases
            current_phase (str): phase identifier, for logging purposes
            current_log (DataFrame): runtime DataFrame of the current phase. Contains feature matrix X, ground truths vector, provider flag, label provided by agent and machine confidence (predict proba)
            strat (str, optional): identifier of current deferral policy evaluated   
        
        """
        if drift_detected:

            self.log.warning(f"DRIFT DETECTED | Phase: {current_phase} | Strat: {strat}")           

            if current_phase == 'HiC':
                phase_info = f"Current desired performance: {self.desired_performance:.4f} | Last 5 Machine EA: {self.machine_fea[-5:]}"
        
            else: 
                phase_info = f"Current belief_threshold: {self.belief_threshold} | Last 5 Machine EA: {self.fea_mic[-5:]}"

            self.log.info(f"DRIFT METRICS: {phase_info}")
            dir = os.path.join(f"processed_data", 
                               self.dataset_name, 
                               "drift_checkpoints",
                               f"{current_phase}", 
                               f"iter_{self.training_iter}")
            
            os.makedirs(dir, exist_ok=True)

            if current_phase == 'MiC' and self.human_deferral_cost is not None:
                file_path = os.path.join(dir, f"{current_phase}_{self.name}_{strat}_{self.human_deferral_cost}.parquet")
            else:
                file_path = os.path.join(dir, f"{current_phase}_{self.name}_{strat}.parquet")
                
            current_log.to_parquet(file_path, index= False)

            self.log.info(f"Drift df saved to: {file_path}")
            return True
        
        return False
            


class HiC(BRIDGET):

    """
    Class for the Human-in-Command phase.
    Handles pre-training the Incremental Learner, co-labeling logic under 4 interaction rules and checkpointing/logging

    Attributes:
        RULE (bool): Boolean flag enabling the Ideal Rule Check (defined in master_config.py, FRANK_RULES)
        PAST (bool): Boolean flag enabling the Individual Fairness Check (defined in master_config.py, FRANK_RULES)
        SKEPT (bool): Boolean flag enabling the Skeptical Learning Check (defined in master_config.py, FRANK_RULES)
        GROUP (bool) Boolean flag enabling the Group Fairness Check (defined in master_config.py, FRANK_RULES)
        EVA (bool): Boolean flag enabling evaluation on the holdout HIC Test set (defined in master_config.py, FRANK_RULES)
        rule_att (str): Name of the feature subject to Ideal Rule Check condition (defined in master_config.py, DATASETS)
        rule_value (int): Value of the rule enforced by Ideal Rule Check, relative to attribute rule_att (defined in master_config.py, DATASETS)
        hic_model_name (str): Name of the chosen Incremental Learner (saving purposes)
        hic_model (obj): River Incremental Learner model (either warmed up or cold start)
        start_performance (int): Baseline Accuracy performance, used to modulate the benchmark within the exit conditions (STARTING value defined in master_config.py, DATASETS)
        performance_delta (float): Delta applied to the start_performance to create the desired performance target
        allocated_budget (int): Maximum interaction budget allocated by the human. Represents their max fatigue level. 
        skepticism_threshold (float): Activation threshold of the Skeptical Learning Check. When the skepticism calculated
                                      for the instance evaluated exceeds this threshold, the machine actively challenges the 
                                      user's authority.
                                      (defined in master_config.py, DATASETS)
        
        **kwargs derived from BRIDGET class

    """
    def __init__(self,
                 RULE, PAST, SKEPT, GROUP, EVA, 
                 rule_att, rule_value, 
                 hic_model_name, hic_model,
                 start_performance=60, #default value, can be used in the first iteration only if the Incremental Learner was cold started
                 allocated_budget = 200,  
                 skepticism_threshold= 0.6,
                 performance_delta= 0.05,
                 **kwargs):  # budget (representing more like fatigue) of the user, works like an exit condition
                
        super().__init__(**kwargs)
         

        self.RULE = RULE
        self.PAST = PAST
        self.SKEPT = SKEPT
        self.GROUP = GROUP
        self.EVA = EVA

        self.rule_att = rule_att
        self.rule_value = rule_value
       
        self.start_performance= start_performance 
        self.allocated_budget= allocated_budget
    
        self.skepticism_threshold= skepticism_threshold
        self.desired_performance= self.start_performance + (self.start_performance*performance_delta)
        self.spent_budget= 0
         
        self.hic_model= hic_model
        self.hic_model_name= hic_model_name
        self.machine_fea= []
        self.user_fea= []
        self.hic_fea= []
        self.initial_model = pickle.loads(pickle.dumps(self.hic_model))

        self.hic_evaluation_results = []
        self.hic_acc = metrics.Accuracy()
        self.hic_F1 = metrics.F1()

        #various counters for testing/debugging purposes
        self.rules_count = 0
        self.past_count = 0
        self.ok_count = 0
        self.no_count = 0
        self.xai_check= 0
        self.xai_ok = 0
        self.xai_no = 0
        self.skept_count = 0
        self.agree_count = 0
        self.disagree_count = 0
        self.fairness_records = [len(self.X) - 1]
        for i in range(0, 100, 10)[1:]: 
            # defines the proportion of data after which the Group Fairness Check is enabled
            # 10 means it requires 10% of the data, it can be changed without any issue
            self.fairness_records.append(percentage(i, len(self.X)))

        self.retrain_count= 0 # tracking the number of times the Incremental Learnes is trained from scratch during HIC
       


    def train(self, x_warm_up, y_warm_up, x_test_warm_up, y_test_warm_up):
        """
        Executes pre-training for the Incremental Learners before deploying BRIDGET, given the data portions allocated.
        This method is called within BRIDGET's Human in Command phase when the model must be re-trained.
        
        Args:
            x_warm_up (list of dicts): Input data used for pre-training the learner
            y_warm_up (array-like): Target label vector matching corresponding training set
            x_test_warm_up (list of dicts): Validation set to assess Accuracy, F1 Score, Predicted Class distribution for model selection
            y_test_warm_up (array-like): Target label vector matching corresponding testing set
        """

        accuracy = metrics.Accuracy()
        f1= metrics.F1()
        predictions = []
    
        
        for x, y in zip(x_warm_up, y_warm_up):
            y = int(y)

            if self.preprocessor is not None:
                self.preprocessor.learn_one(x)
                x= self.preprocessor.transform_one(x)
        
            self.hic_model.learn_one(x, y)
    

        for x, y in zip(x_test_warm_up, y_test_warm_up):
            y = int(y)
            
            if self.preprocessor is not None:
                x = self.preprocessor.transform_one(x)

            y_pred = self.hic_model.predict_one(x)
            
            accuracy.update(y, y_pred)
            f1.update(y, y_pred)
            predictions.append(y_pred)
    
        print(f"{accuracy}")
        print(f"{f1}")
        print(f"Distribution of predictions: {Counter(predictions)}")

        self.train_check = True
        print(f"{self.hic_model} trained")
        return self 



    def get_eval_report(self):
        """
        Compiles evaluation metrics for the HIC phase, including fairness metrics, counters related to the interaction rules
        used to assess the correct functioning of the framework
        """
        names= ['human_fairness', 'human_acc', 'systemic', 'frank_fairness', 'frank_acc',
        'rules_count', 'past_count', 'ok_count', 'no_count', 
        'xai_ok', 'xai_no', 'skept_count', 'agree_count', 'disagree_count'
        ]

        report = pd.DataFrame(self.hic_evaluation_results, columns=names)
        return report

    def start_HiC(self, warm_up_set):
            """
            Orchestrates and executes the Human in Command phase logic.
            Sequentially processess input instances, computes the predictions of both agents and assigns the final label
            according to the four interaction rules.

            Args:
                warm_up_set (DataFrame): Subset that was used to pre-train the learner, included in the procedure as a building block of the 
                K-NN neighborhood population

            Returns:
                final_log (DataFrame): Co-labeled streaming dataset, containing original feature values, final provider flag,
                agent-provided labels and model confidence
                accuracy_score (list): List tracking the Accuracy performance obtained on the holdout batch1_test data split 
                f1_score (list): List tracking the F1-Score obtained on the holdout batch1_test data split 
            """
            self.processed= dict()
            

            machine_predictions = []
            machine_conf_lvls= []
            accuracy_score = []
            f1_score = []
            
            # FEA structures
            skepticisms = []
            fea_num_machine= 0.0
            fea_num_user= 0.0

            fea_den_machine= 0.0
            fea_den_user= 0.0

            fea_system_num =0.0
            fea_system_den =0.0


            # XAI Structures

            proximities_KNN_SAME=[]
            proximities_KNN_OPP= []
            times_KNN= []
            sparsities_KNN_SAME= []
            sparsities_KNN_OPP= []

            similar_nn = []
            opposite_nn = []


            self.X = convert_dict_list_to_float32(self.X)

            for i in tqdm(range(len(self.X))):      

                relabel = False #when this is set to True, Re-Labelling is triggered for the Incremental Learner

                x = self.X[i]
                y = int(self.Y[i])
 
                
                if self.preprocessor is not None: # if a RIVER scaler is provided it learns and transforms the features in row x
                    self.preprocessor.learn_one(x)
                    x= self.preprocessor.transform_one(x) 

                
                record = tuple(list(x.values()))
                user_truth = int(self.user_model.predict(record, y, i))

                machine_prediction = int(self.hic_model.predict_one(x))            
                machine_predictions.append(machine_prediction)

                if record in self.processed: #duplicated records are handled firsthand
                    self.processed[record]['times'] += 1

                    self.log.info(f"Record, row {i} already processed...")
                    old_decision = int(self.processed[record]['decision'])                    

                    if user_truth == old_decision:
                        
                        self.log.info(f"And you are consistent! Decision accepted, class {old_decision}, record {i}.")
                        decision = old_decision
                    
                    else:
                
                        self.log.info(f"Inconsistent, record {i}. You previously said: {old_decision}, Want to change old decision?")                        
                                       
                        confirm= random.choices(population=[False, True], weights=[0.8, 0.2], k=1)[0]

                        if confirm == False:
                            decision = old_decision
                            
                        else:
                            decision = user_truth
                            relabel = True
                    
                    self.stats[user_truth]['user']['tried'] += 1
                    self.stats[machine_prediction]['machine']['tried'] += 1
            
                    if decision == user_truth:
                        self.stats[user_truth]['user']['got'] += 1

                    if decision == machine_prediction: 
                        self.stats[machine_prediction]['machine']['got'] += 1
                    
                    
                
                else:  # LOGIC FOR UNSEEN RECORDS
                   
                    self.processed[record] = dict()
                    self.processed[record]['notes'] = []
                    self.processed[record]['vs'] = None
                    self.processed[record]['ideal'] = None
                    self.processed[record]['times'] = 1

                    try:
                        pred_proba = self.hic_model.predict_proba_one(x)[machine_prediction]

                    except:
                        pred_proba = 0

                    try:
                        user_proba = self.hic_model.predict_proba_one(x)[user_truth]
                    except:

                        self.log.info("Still unlearned...")
                        user_proba = 1
                    

                    user_confidence = self.stats[user_truth]['user']['conf']
                    mach_confidence = self.stats[machine_prediction]['machine']['conf']

                    self.stats[machine_prediction]['machine']['tried'] += 1
                    self.stats[user_truth]['user']['tried'] += 1
                                        
                    machine_conf_lvls.append(pred_proba)
                    
                    # FEA COMPUTATION 
                    
                    if machine_prediction == y:
                        fea_num_machine += 1 

                    if user_truth == y:
                        fea_num_user += 1

                    fea_den_machine += 1
                    fea_den_user += 1

                    mach_fea= fea_num_machine / fea_den_machine if fea_den_machine > 0 else 0.5
                    user_fea= fea_num_user/ fea_den_user if fea_den_user > 0 else 1.0

                    self.user_fea.append(user_fea)
                    self.machine_fea.append(mach_fea)
                            

                    # - UPDATING LOG - meglio alla fine 
                    
                    self.processed[record]['dict_form'] = x
                    self.processed[record]['user'] = user_truth
                    self.processed[record]['machine'] = machine_prediction
                    self.processed[record]['ground_truth'] = y
                    self.processed[record]['proba_model'] = pred_proba

                    self.spent_budget += 1


                    # - LABELING LOGIC
                    
                    #Is record covered by Ideal Rule Check?
                    ideal_value = ideal_record_test(x, self.rule_att, self.rule_value) 
                    
                    #Is record covered by Individual Fairness Check?
                    vs_records, vs_decision = get_value_swap_records(x, self.processed,
                                                                    self.protected, self.attr_list) 

                    if user_truth == machine_prediction:
                        skepticism = 0
                    else:
                        skepticism = mach_confidence * pred_proba - user_confidence * user_proba
                    skepticisms.append({str(i):skepticism})
                    #print(f"Current skepticism: {skepticism}")

                    #User is not consistent w.r.t. Ideal Rule
                    if ideal_value is not None and user_truth != ideal_value and self.RULE: 
                        self.rules_count += 1
                        decision = ideal_value
                        self.processed[record]['ideal'] = False
                        if machine_prediction == ideal_value:
                            provider= 'M'
                            self.stats[machine_prediction]['machine']['got'] += 1

                    #User is consistent w.r.t. Ideal Rule
                    elif ideal_value is not None and user_truth == ideal_value and self.RULE:
                        decision = ideal_value
                        provider= 'H'  
                        self.processed[record]['ideal'] = True
                        if machine_prediction == ideal_value:
                            self.stats[machine_prediction]['machine']['got'] += 1


                    #IRC not triggered. User not consistent w.r.t. Individual Fairnesss
                    elif vs_decision is not None and user_truth != vs_decision and self.PAST: 
                        self.log.info(f"Record {i}, user not consistent w Individual Fairness")
                        self.processed[record]['vs'] = True
                        self.past_count += 1
                        for rec in vs_records:
                            self.processed[rec]['vs'] = True

                        confirm = random.choices(population=[False, True], weights=[0.8, 0.2], k=1)[0]
                        
                        if confirm in [0, "0", False]:
                            decision = vs_decision
                            if machine_prediction == vs_decision:
                                provider= 'M'
                                self.stats[machine_prediction]['machine']['got'] += 1
                                self.log.info(f"Not confirmed, assigned to M record {i}")

                        elif confirm in [1, "1", True]:
                            decision = user_truth
                            provider= 'H'
                            self.log.info(f"Confirmed, assigned to H record {i}")
                            self.stats[user_truth]['user']['got'] += 1
                            if machine_prediction == user_truth:
                                self.stats[machine_prediction]['machine']['got'] += 1
                            for rec in vs_records:
                                self.processed[rec]['decision'] = user_truth
                            relabel = True


                     #IRC not triggered. User consistent w.r.t. Individual Fairnesss
                    elif vs_decision is not None and user_truth == vs_decision and self.PAST:
                        self.processed[record]['vs'] = True
                        for rec in vs_records:
                            self.processed[rec]['vs'] = True
                        decision = vs_decision
                        provider= 'H'
                        if machine_prediction == vs_decision:
                            self.stats[machine_prediction]['machine']['got'] += 1
                    
                    else: #Other conditions not triggered. Skeptical Learning Check
                        
                        if user_truth != machine_prediction and self.SKEPT:
                            self.log.info(f"Skepticality at record {i}, coeff: {skepticism}")

                            if skepticism > self.skepticism_threshold:
                                #print("High skepticism, asking for XAI...")
                                self.skept_count += 1
                                self.xai_check = 0
                                confirm = self.user_model.believe() 
                                #print(f"User belief: {confirm}")
                                #confirm = None
                                
                                if confirm in [0, "0", False]: 
                                    self.log.info(f"XAI requested at record {i}")
                                    start_time = time.time()
                                    xai_log = prepr_log_for_xai(warm_up_set, self.processed, self.attr_list, self.target)

                                    nearest_ex, nearest_opp, sparsity_ex, sparsity_opp = get_neighbors(x, 
                                                                            y, 
                                                                            self.target,
                                                                            xai_log, 
                                                                            relevance_window= 50, 
                                                                            n_neighbors= 2
                                                                            )
                                    
                                    time_KNN= time.time()-start_time

                                    similar_nn.append(nearest_ex)
                                    opposite_nn.append(nearest_opp)

                                    # XAI Evaluation Metrics: (adapted to the 1-NN, hence plausibility is not computed)
                    
                                    #1. Proximity: distance between x and df, same class

                                    distance_x_nn_ex= calculate_distances(x, nearest_ex) 
                                    proximity_nn_ex= distance_x_nn_ex[0][1]
                                    

                                    #2. Proximity: distance between x and df, opposite class
                                    distance_x_nn_opp= calculate_distances(x, nearest_opp) 
                                    proximity_nn_opp= distance_x_nn_opp[0][1]

                                    proximities_KNN_SAME.append({str(record):proximity_nn_ex})
                                    proximities_KNN_OPP.append({str(record):proximity_nn_opp})

                                    times_KNN.append(time_KNN)
                                    
                                    #3. Sparsity
                                    sparsities_KNN_SAME.append({str(record):sparsity_ex})
                                    sparsities_KNN_OPP.append({str(record):sparsity_opp})
                                    
                                    
                                    n_examples= len(nearest_ex) + len(nearest_opp)
                                    
                                    if n_examples > 0:

                                        for e in nearest_ex.values:
                                            gt_col = e[-1] 
                                            rec_feats= e[:-1]

                                            user_opinion = self.user_model.predict(rec_feats, gt_col, i)
                                            if user_opinion == machine_prediction:
                                                self.xai_check += 1
                                                self.xai_ok += 1
                                            else:
                                                self.xai_no += 1

                                        for e in nearest_opp.values:
                                            gt_col = e[-1] 
                                            rec_feats= e[:-1]

                                            user_opinion = self.user_model.predict(rec_feats,gt_col, i)
                                            if user_opinion != machine_prediction:
                                                self.xai_check += 1
                                                self.xai_ok += 1
                                            else:
                                                self.xai_no += 1

                                        if (self.xai_check / n_examples) > 0.5:
                                            confirm = True
                                        else:
                                            confirm = False           

                                if confirm in [0, "0", False]:
                                    self.log.info(f"Unconfirmed explaination, record {i} given to user")
                                    self.no_count += 1
                                    decision = user_truth
                                    provider= 'H'
                                    self.stats[user_truth]['user']['got'] += 1
                                    
                                                            
                                else:
                                    self.log.info(f"Explaination satisfactory, record {i} given to machine")
                                    self.ok_count += 1
                                    decision = machine_prediction
                                    provider= 'M'
                                    self.stats[machine_prediction]['machine']['got'] += 1

                                    
                            else:
                                self.log.info(f"Skepticality less than threshold, record {i} given to user")
                                self.disagree_count += 1
                                decision = user_truth
                                provider= 'H'
                                self.stats[user_truth]['user']['got'] += 1
                                
                        else:
                            self.log.info(f"Both agreed, record {i} given to user")
                            self.agree_count += 1
                            decision = user_truth
                            provider='H'
                            self.stats[user_truth]['user']['got'] += 1
                            self.stats[machine_prediction]['machine']['got'] += 1
                            
                            

                    if decision == y:
                        fea_system_num += 1

                    fea_system_den += 1

                    hic_fea= fea_system_num / fea_system_den if fea_system_den > 0 else 0.5
                    self.hic_fea.append(hic_fea)

                    
                    
                    #Once the final decision has been taken, 
                    # the model is updated. Internal data structures are also updated
                    
                    self.processed[record]['decision'] = int(decision) 
                    self.processed[record]['provider_flag'] = provider

                    self.hic_model.learn_one(x, decision)
                                        
                    
                   
                    try:
                        self.hic_acc.update(decision,machine_prediction)
                        self.hic_F1.update(decision,machine_prediction)
                    
                    except:
                        print('err', x, decision)
                        self.hic_model = pickle.loads(pickle.dumps(self.initial_model))

                        for data in self.processed.values():
                            self.retrain_count += 1
                            self.hic_model.learn_one(data['dict_form'], data['decision']) 


                    
                    
                
                try:
                    self.stats[user_truth]['user']['conf'] = self.stats[user_truth]['user']['got'] / self.stats[user_truth]['user']['tried']
                except:
                    self.stats[user_truth]['user']['conf'] = 1

                try:
                    self.stats[machine_prediction]['machine']['conf'] = self.stats[machine_prediction]['machine']['got'] / self.stats[machine_prediction]['machine']['tried']
                except:
                    self.stats[machine_prediction]['machine']['conf'] = 0



                if relabel == True:

                    self.hic_model = pickle.loads(pickle.dumps(self.initial_model))
                    self.log.info(f"Retraining HiC model at record {i}")
                    for proc in (self.processed.keys()):
                                               
                        x_relabel = self.processed[proc]['dict_form']
                        y_relabel = self.processed[proc]['decision']
                        self.retrain_count += 1
                        self.hic_model.learn_one(x_relabel, y_relabel)
                        
                
                            
                if i in self.fairness_records and self.GROUP:
                    DN, PP, _ = get_fairness(self.hic_model, self.protected, self.processed, self.protected_values)
                    fairnes_relabel = DN[:round(len(DN) * 0.25)] + PP[:round(len(PP) * 0.25)]
                    for e in fairnes_relabel:
                        self.processed[e[1]]['decision'] = 1 - self.processed[e[1]]['decision']
                    
                    self.hic_model = pickle.loads(pickle.dumps(self.initial_model))
                    self.log.info(f"Retraining HiC model at record {i}, Group Fairness")
                    
                    for proc in (self.processed.keys()):
                        x_relabel = self.processed[proc]['dict_form']
                        y_relabel = self.processed[proc]['decision']
                        self.retrain_count += 1
                        self.hic_model.learn_one(x_relabel, y_relabel)
                        
                
                if self.EVA:
                    human_fairness, human_acc, systemic = evaluation_human(self.processed, self.protected, self.Y,
                                                                        self.attr_list)
                
                    frank_fairness, frank_acc, frank_f1,_ = evaluation_frank(self.X_test1, self.Y_test1, self.hic_model, self.protected, self.preprocessor)
                    accuracy_score.append(frank_acc)
                    f1_score.append(frank_f1)
                    
                    self.hic_evaluation_results.append([human_fairness, human_acc, systemic, frank_fairness, frank_acc,
                                                    self.rules_count, self.past_count,
                                                    self.ok_count, self.no_count,
                                                    self.xai_ok, self.xai_no,
                                                    self.skept_count, self.agree_count, self.disagree_count
                                                    ])
                


                hic_drift= exit_HiC(self.allocated_budget, self.spent_budget, self.machine_fea, self.desired_performance)

                if hic_drift:
                    #print(f"Drift here! Record index {record}")
                    dir = os.path.join("HIC_res", f"{self.dataset_name}",f"iter_{self.training_iter}", f"results_{self.name}")
                   
                    hic_pref = f"User_{self.name}_{self.hic_model_name}_"

                    metrics_knn = {
                    "time_steps": times_KNN,
                    "sparsity_SAME": sparsities_KNN_SAME,
                    "sparsity_OPP": sparsities_KNN_OPP,
                    "proximity_SAME": proximities_KNN_SAME,
                    "proximity_OPP": proximities_KNN_OPP,
                    "method": "KNN",  
                    "dataset": self.dataset_name
                    }


                    skept = {
                        "skept":skepticisms
                    }

                    hic_d = {
                    "model.pkl": self.hic_model,
                    "preprocessor.pkl":self.preprocessor,
                    "Accuracy.txt": accuracy_score,
                    "F1.txt": f1_score,
                    "Machine_Confidence.txt": machine_conf_lvls,
                    "KNN_metrics.json": metrics_knn,
                    "skept.json": skept,
                    "HiC_Acc_machine.txt": self.machine_fea,
                    "HiC_Acc_user.txt": self.user_fea,
                    "HiC_System_ACC.txt": self.hic_fea,
                    "HiC_stats.txt": self.stats
                   
                    }
                    
                    save_data(dir, hic_pref, hic_d)
                    
                    _, df_drift_log = get_percentage_and_df(None, self.processed,self.target, self.feature_names) 

                    super().switch_phase(hic_drift, current_phase= 'HiC', current_log=df_drift_log)
                    return df_drift_log, accuracy_score, f1_score
                

                    
                
                    

                if i ==  (len(self.X)-1):
                    
                    dir = os.path.join("HIC_res", 
                                       f"{self.dataset_name}", 
                                       f"iter_{self.training_iter}", 
                                       f"results_{self.name}")

                    hic_pref = f"User_{self.name}_{self.hic_model_name}_"


                    metrics_knn = {
                    "time_steps": times_KNN,
                    "sparsity_SAME": sparsities_KNN_SAME,
                    "sparsity_OPP": sparsities_KNN_OPP,
                    "proximity_SAME": proximities_KNN_SAME,
                    "proximity_OPP": proximities_KNN_OPP,
                    "method": "KNN",  
                    "dataset": self.dataset_name
                    }


                    skept = {
                        "skept":skepticisms
                    }

                    hic_d = {
                    "model.pkl": self.hic_model,
                    "preprocessor.pkl":self.preprocessor,
                    "Accuracy.txt": accuracy_score,
                    "F1.txt": f1_score,
                    "Machine_Confidence.txt": machine_conf_lvls,
                    "KNN_metrics.json": metrics_knn,
                    "skept.json": skept,
                    "HiC_EA_machine.txt": self.machine_fea,
                    "HiC_EA_user.txt": self.user_fea,
                    "HiC_System_ACC.txt": self.hic_fea,
                    "HiC_stats.txt": self.stats
                    
                    }
                    
                    save_data(dir, hic_pref, hic_d)



            _, df_final_hic = get_percentage_and_df(None, self.processed,self.target, self.feature_names) 
            
            return df_final_hic, accuracy_score, f1_score

class MiC(BRIDGET):
    """
    Orchestrates and executes the Machine in Command phase.
    Includes the Selective Prediction and Two-Stage Deferral strategies.
    Generates counterfactuals using the GrowingSpheres algorithm.

    Attributes:
        mic_model (object, DeferralNet): Pre-trained Neural Network classifier
        mic_model_name (str): Identifier for the DeferralNet model
        benchmark_performance (float): Baseline performance obtained during the latest Machine in Command iteration
        warm_up (int): Minimum number of instances that must be labeled by the machine (NO DEFERRAL), before BRIDGET starts assessing conceptual drifts
        device (torch.device): Either CPU/CUDA
        performance_delta (float): Margin threshold applied to the benchmark_performance to identify the current target
        belief_threshold (float): Threshold identifying low-belief instances, for reporting the healthiness level of the machine
        tau_threshold (float, optional): Deferral threshold optimizing the coverage/accuracy trade off for the Selective Prediction deferral strategy
        human_deferral_cost (float, optional): Current configuration of inference cost incurred by the human when predicting contextually to the Two-Stage deferral policy
        **kwargs
    """

    def __init__(self,
                 mic_model, 
                 mic_model_name,
                 benchmark_performance,
                 warm_up,
                 device,
                 performance_delta= 0.05,
                 belief_threshold= 0.6,
                 tau_threshold= None,
                 human_deferral_cost= None,  
                 **kwargs
                 ):
        
        
        super().__init__(**kwargs)

        self.human_deferral_cost= str(human_deferral_cost) # cast to str to create the saving directory
        self.mic_model= mic_model  
        self.mic_model_name= mic_model_name
        self.benchmark_performance=benchmark_performance
        self.device= device
        self.performance_delta= performance_delta

        self.warm_up= warm_up

        self.belief_threshold= belief_threshold
        self.tau= tau_threshold
      
        # deriving desired performance for the current iteration based on the benchmark of the precedent iteration 
        self.performance_thresh= (self.benchmark_performance - (self.benchmark_performance * self.performance_delta)) /100

        
        self.system_acc= 0.0
        self.model_acc_all= 0.0
        self.model_acc_undeferred=0.0
        self.model_acc_deferred= 0.0

        self.mic_preds= []  
        self.low_belief_count= 0
        self.deferred_decisions= 0
        self.undeferred_decisions= 0

        self.fea_mic= []
        self.fea_net= []
        self.mic_user_fea= []

    def start_MiC(self, x_stream, y_stream, df_switch, r_net=None, two_step_deferral= None): 
        """
        Executes the Machine in Command phase: instances are triaged according to the deferral policies included within the framework

        Args:
            x_stream (Tensor): Feature matrix from batch3, transformed to tensor 
            y_stream (Tensor):Llabel vector from batch3, transformed to tensor 
            df_switch (DataFrame): Co-labeled data resulting from HIC phase
            r_net (object, optional): Trained cost-sensitive rejector network for Two-Stage Deferral policy
            two_step_deferral (bool, optional): Boolean flag signaling whether Two-Stage deferral is deployed in the current iteration
        
        Returns:
            final_log (DataFrame): Final Data Frame storing the original feature values, final provider flag,
            labels provisionally given by both agents and model confidence
        """
        # initializing logs and necessary storing structures
        self.processed= dict()
        self.mic_results= dict()

        fea_mic_num = 0
        fea_mic_den = 0

        fea_net_num= 0
        fea_net_den= 0

        fea_user_num= 0
        fea_user_den= 0

        # counters for accuracy measures
        net_correct_on_undeferred= 0
        net_correct_on_deferred= 0

        mach_confidence= [] 
        mach_predictions= [] 

        # GROWING SPHERES structures
        times_GS= []
        sparsities_GS = []
        proximities_GS = [] 
        plausabilities_GS = []

        ## Main Loop
    
        for record, (x_rec, y_gt) in enumerate(tqdm(zip(x_stream, y_stream), total=len(x_stream))):
                
            self.processed[record]= dict() 
            self.mic_results[record]= dict()
            x = self.df_batch3[self.feature_names].iloc[record].to_dict()
           
            y_gt= int(y_gt.item())

            self.processed[record]['dict_form'] = x
            self.processed[record]['ground_truth'] = y_gt
            
            # obtaining Net predictions
            with torch.no_grad():

                if x_rec.ndim == 1:
                    x_rec = x_rec.unsqueeze(0)

                outputs= self.mic_model(x_rec) # logit scores

                net_output= torch.argmax(outputs, dim=1).item() # obtaining label
            
                mach_predictions.append(net_output)

            # computing predicted probability 
            try:
                probas= self.mic_model.predict_proba_nn(x_rec, device=self.device)
                max_conf = float(np.max(probas))
                mach_confidence.append(max_conf)

            except Exception as e:
                self.log.warning(f"Error: {e}") 
                max_conf = 0.0
            
            self.stats[net_output]['machine']['tried']+=1
    

            ## - DEFERRAL LOGIC 
            
            if two_step_deferral: ## Two Stage Deferral Strategy
                 
                deferral_proba= p_defer(x_rec, r_net)
                p_val = deferral_proba.item()
             
                if p_val >= 0.5: 
                    
                    x_input_user = x_rec.squeeze().cpu().numpy()    
                    user_pred = self.user_model.predict(x_input_user, y_gt, record)
                    decision = user_pred
                    provider= 'H'

                    if net_output == y_gt:
                        net_correct_on_deferred += 1

                    self.deferred_decisions +=1
                    self.stats[user_pred]['user']['tried'] +=1
                    self.stats[user_pred]['user']['got'] += 1

               
                
                else:
                    provider= 'M'
                    decision = int(net_output)
                    
                    x_input_user = x_rec.squeeze().cpu().numpy()
                    user_pred = self.user_model.predict(x_input_user, y_gt, record)

                    if net_output == y_gt:
                        net_correct_on_undeferred+=1

                    self.stats[net_output]['machine']['got']+=1 
                    self.undeferred_decisions+=1
                    


            else:  # Selective Prediction
                 
                # 1. No Deferral, these records are reliable according to the results obtained
                
                if max_conf >= self.tau:

                    decision = int(net_output)
                    x_input_user = x_rec.squeeze().cpu().numpy()
                    user_pred= self.user_model.predict(x_input_user, y_gt, record) 
    
                    provider= 'M'

                    if net_output == y_gt:
                        net_correct_on_undeferred+=1
                        
                    self.stats[net_output]['machine']['got']+=1 # this is relative to the class the net has predicted
                    self.undeferred_decisions+=1

                # 2. Deferral to user: the model was unrealiable here, show counterexamples
                else:
                    
                    x_input_user = x_rec.squeeze().cpu().numpy()
                                
                    user_pred = self.user_model.predict(x_input_user, y_gt, record)
                   
                    decision = user_pred
                    provider = 'H'

                    if net_output == y_gt:
                        net_correct_on_deferred += 1

                    self.deferred_decisions +=1
                    self.stats[user_pred]['user']['tried'] +=1
                    self.stats[user_pred]['user']['got'] += 1

               
            self.mic_preds.append(decision) # updating the structure



            ### Explanation logic: GROWING SPHERES

            xai_log= prepr_log_for_xai(df_switch, self.processed, self.attr_list, self.target)
            torch_wrapper= PyTorchWrapper(self.mic_model,self.target, self.feature_names)

            cfs, time_GS, sparsity_GS= get_GS_cfe(xai_log, x, torch_wrapper, self.cats, self.target)

            try:

                cf_df = pd.DataFrame([cfs]).drop(columns=[self.target], errors='ignore')
                x_df = pd.DataFrame([x]).drop(columns=[self.target], errors='ignore')
                        
                cf_df = cf_df[x_df.columns]
                        
                d = cdist(cf_df, x_df, metric='euclidean').flatten()[0] # computing distances
                proximity_GS = d

            except Exception as e:
                self.log.warning(f"Error during distance computation: {e}")
                proximity_GS = np.nan

            distance_cf_hist_GS = calculate_distances(cfs, xai_log, feature_ranges=None)  
            plausability_GS = distance_cf_hist_GS[0][1]
                    
            # logging
            proximities_GS.append({str(record):proximity_GS})
            times_GS.append(time_GS)
            sparsities_GS.append({str(record):sparsity_GS})
            plausabilities_GS.append({str(record): plausability_GS})
            

            ## - EMPIRICAL ACC COMPUTATION + AGENT-BASED ACCURACY
            
            # MiC System Accuracy
            if decision == y_gt:
                fea_mic_num += 1 

            fea_mic_den += 1

            fea_mic_model= fea_mic_num / fea_mic_den if fea_mic_den > 0 else 0.5
              
            self.fea_mic.append(fea_mic_model)
            

            # Deferral Net Accuracy
            if net_output == y_gt:
                fea_net_num += 1 

            fea_net_den += 1

            fea_net= fea_net_num / fea_net_den if fea_net_den > 0 else 0.5
                
            self.fea_net.append(fea_net)

    
            # User Accuracy
            if user_pred == y_gt:
                fea_user_num += 1
            
            fea_user_den += 1
            fea_user= fea_user_num / fea_user_den if fea_user_den > 0 else 0.5
                
            self.mic_user_fea.append(fea_user)
            
            
            # Empirical Accuracy computation (measuring EA values for all classes and both agents)
            try:
                self.stats[user_pred]['user']['conf'] = self.stats[user_pred]['user']['got'] / self.stats[user_pred]['user']['tried']
            except:
                self.stats[user_pred]['user']['conf'] = 1

            try:
                self.stats[net_output]['machine']['conf'] = self.stats[net_output]['machine']['got'] / self.stats[net_output]['machine']['tried']
            except:
                self.stats[net_output]['machine']['conf'] = 0
                        
        
            # updating log

            self.processed[record]['user'] = user_pred
            self.processed[record]['machine'] = net_output
            self.processed[record]['proba_model']= max_conf
            self.processed[record]['decision'] = decision   
            self.processed[record]['provider_flag'] = provider         

            # assessing concept drift and flagging low belief instances

            if torch.is_tensor(max_conf):
                    max_conf = max_conf.item()

            belief= max_conf * fea_mic_model
            
            if belief <= self.belief_threshold:
                self.low_belief_count += 1

            mic_drift= exit_MiC(fea_vals=self.fea_mic,
                                desired_performance=self.performance_thresh,
                                undeferred_decisions=self.undeferred_decisions,
                                warm_up= self.warm_up
            )
            
            
            if mic_drift:
                #print(f"Drift here! Record index {record}")
                self.log.warning(f"Drift here! Record index {record}")

                y_mic = np.array(self.mic_preds)
                y_mach = np.array(mach_predictions)
                y_true = self.df_batch3[self.target].iloc[:len(y_mic)].to_numpy()
                
                self.system_acc= (y_mic == y_true).mean()
                self.model_acc_all= (y_mach == y_true).mean()

                self.model_acc_undeferred= net_correct_on_undeferred/self.undeferred_decisions if self.undeferred_decisions > 0 else 0.5
                self.model_acc_deferred= net_correct_on_deferred/self.deferred_decisions if self.deferred_decisions > 0 else 0.5

                self.mic_results[record]['system_accuracy'] = self.system_acc
                self.mic_results[record]['machine_overall_accuracy']= self.model_acc_all
                self.mic_results[record]['machine_acc_on_undeferred']= self.model_acc_undeferred
                self.mic_results[record]['machine_acc_on_deferred']= self.model_acc_deferred
            
                res_df= pd.DataFrame(self.mic_results).transpose()
                
                
                # checkpointing if drift happened
                strat = "Two_Step" if two_step_deferral else "Confidence"
                dir = os.path.join("MIC_res", 
                                   f"{self.dataset_name}", 
                                   f"iter_{self.training_iter}", 
                                   f"{self.name}_{strat}"
                                   )
                
                if two_step_deferral:
                    dir= os.path.join(dir,
                                           f"beta_{self.human_deferral_cost}"
                                           )
                    mic_pref= f"MIC_DRIFT_{self.mic_model_name}_"

                else:
                    mic_pref= f"MIC_DRIFT_{self.mic_model_name}_"
                
                os.makedirs(dir, exist_ok=True)
                res_df.to_parquet(os.path.join(dir, f"{mic_pref}Performances.parquet"))

                metrics_gs = {
                            "time_steps": times_GS,
                            "sparsity": sparsities_GS,
                            "proximity": proximities_GS,
                            "plausability":plausabilities_GS,
                            "method": "GS",  
                            "dataset": self.name
                            }
                
                mic_res = {
                    "model.pkl": self.mic_model,
                    "Model_Confidence.txt": mach_confidence,
                    "MiC_stats.txt": self.stats,
                    "System_Accuracy.txt": self.fea_mic,
                    "Model_Accuracy.txt": self.fea_net,
                    "User_Accuracy.txt": self.mic_user_fea,
                    "GS_metrics.json": metrics_gs                
                    }
                
                save_data(dir, mic_pref, mic_res)

                _, df_log= get_percentage_and_df(None, self.processed, self.target, self.feature_names) 

                super().switch_phase(mic_drift, current_phase= 'MiC', current_log= df_log, strat= strat)

                return df_log
           
            
            # when drift is not happening update accuracy metrics and evaluat
            y_mic = np.array(self.mic_preds)
            y_mach = np.array(mach_predictions)
            y_true = self.df_batch3[self.target].iloc[:len(y_mic)].to_numpy()
                    
            self.system_acc= (y_mic == y_true).mean()
            self.model_acc_all= (y_mach == y_true).mean()        
            self.model_acc_undeferred= net_correct_on_undeferred/self.undeferred_decisions if self.undeferred_decisions > 0 else 0.5
            self.model_acc_deferred= net_correct_on_deferred/self.deferred_decisions if self.deferred_decisions  > 0 else 0.5
            
            self.mic_results[record]['system_accuracy'] = self.system_acc
            self.mic_results[record]['machine_overall_accuracy']= self.model_acc_all
            self.mic_results[record]['machine_acc_on_undeferred']= self.model_acc_undeferred
            self.mic_results[record]['machine_acc_on_deferred']= self.model_acc_deferred

            

        ## Main Loop exit with no drift

        strat = "Two_Step" if two_step_deferral else "Confidence"
        dir = os.path.join("MIC_res", 
                                   f"{self.dataset_name}", 
                                   f"iter_{self.training_iter}", 
                                   f"{self.name}_{strat}"
                                   ) 
      
        if two_step_deferral:
            dir= os.path.join(dir,
                            f"beta_{self.human_deferral_cost}"
                            )
            mic_pref= f"{self.mic_model_name}_"
        else:
            mic_pref= f"{self.mic_model_name}_"
        
        os.makedirs(dir, exist_ok=True)
        res_df= pd.DataFrame(self.mic_results).transpose()
        res_df.to_parquet(os.path.join(dir, f"{mic_pref}Performances.parquet"))

        metrics_gs = {
                        "time_steps": times_GS,
                        "sparsity": sparsities_GS,
                        "proximity": proximities_GS,
                        "plausability":plausabilities_GS,
                        "method": "GS",  
                        "dataset": self.name
                    }
                
        mic_res = { "model.pkl": self.mic_model,
                    "Performances.txt": self.mic_results, 
                    "Model_Confidence.txt": mach_confidence,
                    "MiC_stats.txt": self.stats,
                    "System_Accuracy.txt": self.fea_mic,
                    "Model_Accuracy.txt": self.fea_net,
                    "User_Accuracy.txt": self.mic_user_fea,
                    "GS_metrics.json": metrics_gs
                    }
        
        save_data(dir, mic_pref, mic_res)


        _, df_log_final = get_percentage_and_df(None, self.processed, self.target, self.feature_names)

        return df_log_final
    








 