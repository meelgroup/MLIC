

# Contact: Bishwamittra Ghosh [email: bghosh@u.nus.edu]

import numpy as np
import pandas as pd
import warnings
import math
import random
from tqdm import tqdm
from time import time
# warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import classification_report, accuracy_score





# from pyrulelearn
import pyrulelearn.utils
import pyrulelearn.cplex_wrap
import pyrulelearn.maxsat_wrap



class imli():
    def __init__(self, num_clause=5, data_fidelity=1, weight_feature=1, threshold_literal=-1, threshold_clause=-1,
                 solver="open-wbo", rule_type="CNF", batchsize=400,
                 work_dir=".", timeout=100, verbose=False):
        '''

        :param numBatch: no of Batchs of training dataset
        :param numClause: no of clause in the formula
        :param dataFidelity: weight corresponding to accuracy
        :param weightFeature: weight corresponding to selected features
        :param solver: specify the (name of the) bin of the solver; bin must be in the path
        :param ruleType: type of rule {CNF,DNF}
        :param workDir: working directory
        :param verbose: True for debug

        --- more are added later

        '''

        # assert 0 <= batchsize and batchsize <= 1
        assert isinstance(batchsize, int)
        assert isinstance(data_fidelity, int)
        assert isinstance(weight_feature, int)
        assert isinstance(num_clause, int)
        assert isinstance(threshold_clause, int)
        assert isinstance(threshold_clause, int)


        
        self.numClause = num_clause
        self.dataFidelity = data_fidelity
        self.weightFeature = weight_feature
        self.solver = solver
        self.ruleType = rule_type
        self.workDir = work_dir
        self.verbose = verbose
        self._selectedFeatureIndex = []
        self.timeOut = timeout
        self.memlimit = 1000*16
        self.learn_threshold_literal = False
        self.learn_threshold_clause = False
        self.threshold_literal = threshold_literal
        self.threshold_clause = threshold_clause
        self.batchsize = batchsize
        self._solver_time = 0
        self._prediction_time = 0
        self._wcnf_generation_time = 0
        self._demo_time = 0

        
        

        

        if(self.ruleType == "relaxed_CNF"):
            self.solver = "cplex"  # this is the default solver for learning rules in relaxed_CNFs
        
    
    def __repr__(self):
        print("\n\nIMLI:->")
        return '\n'.join(" - %s: %s" % (item, value) for (item, value) in vars(self).items() if "_" not in item)

    def _get_selected_column_index(self):
        return_list = [[] for i in range(self.numClause)]
        
        for elem in self._selectedFeatureIndex:
            new_index = int(elem)-1
            return_list[int(new_index/self.numFeatures)].append(new_index % self.numFeatures)
        return return_list

    def get_selected_column_index(self):
        temp = self._get_selected_column_index()
        result = []
        for index_list in temp:
            each_level_index = []
            for index in index_list:
                phase = 1
                actual_feature_len = int(self.numFeatures/2)
                if(index >= actual_feature_len):
                    index = index - actual_feature_len
                    phase = -1
                each_level_index.append((index, phase))
            result.append(each_level_index)
        
        return result
                



    def get_num_of_iterations(self):
        return self.iterations

    def get_num_of_clause(self):
        return self.numClause

    def get_weight_feature(self):
        return self.weightFeature

    def get_rule_type(self):
        return self.ruleType
    

    def get_work_dir(self):
        return self.workDir

    def get_weight_data_fidelity(self):
        return self.dataFidelity

    def get_solver(self):
        return self.solver

    def get_threshold_literal(self):
        return self.threshold_literal_learned
        
    def get_threshold_clause(self):
        return self.threshold_clause_learned

    def _fit_relaxed_CNF_old(self, XTrain, yTrain):

        
        if (self.threshold_clause == -1):
            self.learn_threshold_clause = True
        if (self.threshold_literal == -1):
            self.learn_threshold_literal = True

        self.iterations = int(math.ceil(XTrain.shape[0]/self.batchsize))    
        self.trainingSize = len(XTrain)
        self._assignList = []
        
        # define weight (use usual regularization, nothing)
        # self.weight_feature = (1-self.lamda)/(self.level*len(self.column_names))
        # self.weight_datafidelity = self.lamda/(self.trainingSize)

        # reorder X, y based on target class, when sampling is allowed
        # if(self.sampling):
        XTrain_pos = []
        yTrain_pos = []
        XTrain_neg = []
        yTrain_neg = []
        for i in range(self.trainingSize):
            if(yTrain[i] == 1):
                XTrain_pos.append(XTrain[i])
                yTrain_pos.append(yTrain[i])
            else:
                XTrain_neg.append(XTrain[i])
                yTrain_neg.append(yTrain[i])

        Xtrain = XTrain_pos + XTrain_neg
        ytrain = yTrain_pos + yTrain_neg

        for i in range(self.iterations):
            if(self.verbose):
                print("\n\n")
                print("sampling-based minibatch method called")

                print("iteration", i+1)
            XTrain_sampled, yTrain_sampled = pyrulelearn.utils._generateSamples(self, XTrain, yTrain)

            assert len(XTrain[0]) == len(XTrain_sampled[0])

            pyrulelearn.cplex_wrap._call_cplex(self, np.array(XTrain_sampled), np.array(yTrain_sampled))

    
    def _fit_relaxed_CNF(self, XTrain, yTrain):

        
        if (self.threshold_clause == -1):
            self.learn_threshold_clause = True
        if (self.threshold_literal == -1):
            self.learn_threshold_literal = True

        self.iterations = int(math.ceil(XTrain.shape[0]/self.batchsize))    
        
        best_loss = self.dataFidelity * XTrain.shape[0] + self.numFeatures * self.weightFeature * self.numClause
        self._assignList = []
        best_loss_attribute = None
        num_outer_idx = 2
        for outer_idx in range(num_outer_idx):

            # time check
            if(time() - self._fit_start_time > self.timeOut):
                continue
            

            XTrains, yTrains = pyrulelearn.utils._numpy_partition(self, XTrain, yTrain)
            batch_order = None
            random_shuffle_batch = False
            if(random_shuffle_batch):
                batch_order = random.sample(range(self.iterations), self.iterations)
            else:
                batch_order = range(self.iterations)

            for each_batch in tqdm(batch_order, disable = not self.verbose):
                # time check
                if(time() - self._fit_start_time > self.timeOut):
                    continue
                
                if(self.verbose):
                    print("\nTraining started for batch: ", each_batch+1)
                pyrulelearn.cplex_wrap._call_cplex(self, XTrains[each_batch], yTrains[each_batch])


                # performance
                yhat = self.predict(XTrain)
                acc = accuracy_score(yTrain, yhat)
                def _loss(acc, num_sample, rule_size):
                    return (1-acc) * self.dataFidelity * num_sample + rule_size * self.weightFeature
                loss = _loss(acc, XTrain.shape[0], len(self._selectedFeatureIndex))
                if(loss <= best_loss):
                    # print()
                    # print(acc, len(self._selectedFeatureIndex))
                    # print(loss)
                    best_loss = loss
                    best_loss_attribute = (self._xhat, self._selectedFeatureIndex, self._assignList, self.threshold_literal_learned, self.threshold_clause_learned)

                else:
                    if(best_loss_attribute is not None):
                        (self._xhat, self._selectedFeatureIndex, self._assignList, self.threshold_literal_learned, self.threshold_clause_learned) = best_loss_attribute

                
            if(self.iterations == 1):
                # When iteration = 1, training accuracy is optimized. So there is no point to iterate again
                break

        
        assert best_loss_attribute is not None
        self._xhat, self._selectedFeatureIndex, self._assignList, self.threshold_literal_learned, self.threshold_clause_learned = best_loss_attribute 
        # print("Finally", self.threshold_literal_learned, self.threshold_clause_learned)
        # self._learn_parameter() # Not required for relaxed_CNF
        return 

    
    def _fit_decision_sets(self, XTrain, yTrain):

        """
            The idea is to learn a decision sets using an iterative appraoch. 
            A decision set is a set of rule, class label pair where each rule is a conjunction of Boolean predicates (i.e., 
            single clause DNF)
            In a decision set, each rule is independent (similar to If-then statements).
            The classification of new input is decided based on following three rules:
                1. If the input satisfies only one rule, it is predicted the corresponding class label
                2. If the input satisfies more than one rules, we consider a voting function. One simple voting function
                   is to consider the majority class label out of all satisfied rules.
                3. When the input does not satisfy any rule, there is a default class at the end of the decision sets.


            In the iterative approach, we want to learn a rule for a target class label in each iteration. 
            Once a rule is learned, we will separate the training set into covered and uncovered. 

            Covered set: These samples satisfy the rule. A sample can either be correctly covered or incorrectly covered. 
                We consider following two cases:

                    A. For a correctly covered sample, we want to specify constraints such that no other rule covers this sample. 
                        Because, in the best case, another rule(s) with same class label may cover this sample, which is desired.
                        In the worst case, a rule(s) with different class label can cover this, which is not desired!!!!

                        In both cases, the overlap between rules increases, but we want to decrease overlap.

                    B. For an incorrectly covered sample, we want it to be correctly covered by another rule(s) and ask for
                        the voting function to finally (!!) output the correct class. 

                        In this case, we want to increase overlap between rules but carefully.

            Uncovered set: These samples do not satisy the rule. Hence we initiate another iteration to learn a new rule that 
                will hopefully cover more samples. 


            As we have defined covered and uncovered samples, we next define how we choose the target class. 

            Target class: this is a drawback(?) of IMLI that can learn a DNF formula for a fixed target class label.
            For now, we choose the majority class in the uncovered samples as the target class.

            We next discuss modifying the class labels of already covered samples, which constitutes the most critical contribution
            of this algorithm. 

            *************************
            ** Critical discussion **
            *************************

            As clear from point A and B, we have opposing goal of both increasing and decreasing overlap in order to increase
            the overall training accuracy

            ## First we tackle A (correctly covered). Let assume the target class for the current iteration is t (other choice is 1-t for binary classification)
                For a correctly covered sample with original class t, we modify the class as 1-t because we want to 
                decrease overlap
                And for a correctly covered sample with original class 1-t, no modification is required

                Similar argument applies when target class is 1-t

            ## To tackle B (incorrectly covered), no modification of class labels is required. Because this samples are incorectly 
                covered at least once. So we want to learn new rules that can cover them correctly. 
               

            
            
        """
        num_outer_idx = 2

        # know which class is majority
        majority = np.argmax(np.bincount(yTrain))
        all_classes = np.unique(yTrain) # all classes in y

        # sample size is variable. Therefore we will always maintain this sample size as maximum sample size for all iterations
        sample_size = self.batchsize
        

        # Use MaxSAT-based rule learner
        ruleType_orig = self.ruleType
        self.ruleType = "DNF"
        # self.timeOut = int(self.timeOut/(num_outer_idx * self.numClause))
        self.timeOut = float(self.timeOut/self.numClause)
        k = self.numClause
        self.numClause = 1
        self.clause_target = []
        xhat_computed = []
        verbose = self.verbose
        self.verbose = False

        XTrain_covered = np.zeros(shape=(0,XTrain.shape[1]), dtype=bool)
        yTrain_covered = np.zeros(shape=(0,), dtype=bool)
        
        time_statistics = []
        # iteratively learn a DNF clause for 1, ..., k
        for idx in range(k):
            self._fit_start_time = time()
            
            # Trivial termination when there is no sample to classify
            if(len(yTrain) == 0):
                break

            yTrain_orig = yTrain.copy()

            if(verbose):
                print("\n\n\n")
                print(idx)
                print("total samples:", len(yTrain))
                print("positive samples:", yTrain.sum())

            
            # decide target class, at this point, the problem reduces to binary classification
            target_class = np.argmax(np.bincount(yTrain))
            self.clause_target.append(target_class)
            yTrain = (yTrain == target_class).astype(bool)
            yTrain_working = np.concatenate((yTrain, np.zeros(shape=yTrain_covered.shape, dtype=bool)))
            XTrain_working = np.concatenate((XTrain, XTrain_covered))

            if(verbose):
                print("\nTarget class:", target_class)
                print("Including covered samples")
                print("total samples:", len(yTrain_working))
                print("target samples:", int(yTrain_working.sum()))
                print("Time left:", self.timeOut - time() + self._fit_start_time)

                
            
            
            


            self.iterations = max(2**math.floor(math.log2(len(XTrain_working)/sample_size)),1)
            if(verbose):
                print("Iterations:", self.iterations)
            
            
            
            best_loss = self.dataFidelity * XTrain.shape[0] + self.numFeatures * self.weightFeature
            best_loss_attribute = None
            self._assignList = []
            for outer_idx in range(num_outer_idx):

                # time check
                if(time() - self._fit_start_time > self.timeOut):
                    continue


                """
                    Two heuristics: 
                    1. random shuffle on batch (typically better performing)
                    2. without randomness
                """
                XTrains, yTrains = pyrulelearn.utils._numpy_partition(self, XTrain_working, yTrain_working)
                
                batch_order = None
                random_shuffle_batch = False
                if(random_shuffle_batch):
                    batch_order = random.sample(range(self.iterations), self.iterations)
                else:
                    batch_order = range(self.iterations)

                for each_batch in tqdm(batch_order, disable = not verbose):
                    
                    # time check
                    if(time() - self._fit_start_time > self.timeOut):
                        continue
                    

                    if(self.verbose):
                        print("\nTraining started for batch: ", each_batch+1)

                    pyrulelearn.maxsat_wrap._learnModel(self, XTrains[each_batch], yTrains[each_batch], isTest=False)
                    

                    # performance
                    self._learn_parameter()
                    yhat = self.predict(XTrain)
                    acc = accuracy_score(yTrain, yhat)
                    def _loss(acc, num_sample, rule_size):
                        return (1-acc) * self.dataFidelity * num_sample + rule_size * self.weightFeature
                    loss = _loss(acc, XTrain.shape[0], len(self._selectedFeatureIndex))
                    if(loss <= best_loss):
                        # print()
                        # print(acc, len(self._selectedFeatureIndex), self.dataFidelity, self.weightFeature, XTrain.shape[0])
                        # print(loss)    
                        best_loss = loss
                        best_loss_attribute = (self._xhat, self._selectedFeatureIndex, self._assignList)
                    else:
                        if(best_loss_attribute is not None):
                            self._assignList = best_loss_attribute[2]
                    
                    # best_loss_attribute = (self._xhat, self._selectedFeatureIndex, self._assignList)


                    
                if(self.iterations == 1):
                    # When iteration = 1, training accuracy is optimized. So there is no point to iterate again
                    break
                    
                # print()
            if(verbose):
                print("Max loss:", best_loss)
            assert best_loss_attribute is not None

            

            self._xhat, self._selectedFeatureIndex, self._assignList = best_loss_attribute 
            # print("Best:", best_loss)
                    

            self._learn_parameter()

            # If learned rule is empty, it can be discarded
            if(self._xhat[0].sum() == 0):
                if(len(self.clause_target) > 0):
                    self.clause_target = self.clause_target[:-1]
                break
            else:
                xhat_computed.append(self._xhat[0])

            yhat = self.predict(XTrain)
            
            """
            Decision sets is a list of independent itemsets ( or list of DNF clauses).
            If yhat matches both the clause_target and ytrain_orig, then the sample is  covered by the DNF clause
            and is also perfectly classified. So the rest of the samples are considered in the next iteration.
            """
            
            # Find incorrectly covered or uncovered samples
            mask = (yhat == 0) | (yhat != yTrain)

            # include covered samples
            XTrain_covered = np.concatenate((XTrain_covered, XTrain[~mask]))
            yTrain_covered = np.concatenate((yTrain_covered, yTrain_orig[~mask]))

            # extract uncovered and incorrectly covered samples
            XTrain = XTrain[mask]
            yTrain = yTrain_orig[mask]

            
            if(verbose):
                print("Coverage:", len(yTrain_orig[~mask]) , "samples")
                print("Of which, positive samples in original:", yTrain_orig[~mask].sum())
            
            # If no sample is removed, next iteration will generate same hypothesi, hence the process is terminated
            if(len(yTrain_orig[~mask]) == 0):
                break

        
        
        """
        Default rule
        """
        xhat_computed.append(np.zeros(self.numFeatures))
        reach_once = False
        for each_class in all_classes:
            if(each_class not in self.clause_target):
                self.clause_target.append(each_class)
                reach_once = True
                break
        if(not reach_once):
            self.clause_target.append(majority)


        # Get back to initial values
        self.numClause = len(self.clause_target)
        self._xhat = xhat_computed
        self.ruleType = ruleType_orig
        self.verbose = verbose


        # parameters learned for rule
        self.threshold_literal_learned = [selected_columns.sum() for selected_columns in self._xhat]
        self.threshold_clause_learned = None

        # print(self.clause_target)
        # print(self._xhat)
        # print(self.threshold_clause_learned, self.threshold_literal_learned)

    def _fit_CNF_DNF_recursive(self, XTrain, yTrain):

        num_outer_idx = 2
        # sample size is variable. Therefore we will always maintain this sample size as maximum sample size for all iterations
        sample_size = self.batchsize
        

        # Use MaxSAT-based rule learner
        ruleType_orig = self.ruleType
        # self.ruleType = "DNF"
        # self.timeOut = int(self.timeOut/(num_outer_idx * self.numClause))
        self.timeOut = float(self.timeOut/self.numClause)
        k = self.numClause
        self.numClause = 1
        self.clause_target = []
        xhat_computed = []
        selectedFeatureIndex_computed = []
        verbose = self.verbose
        self.verbose = False
        
        
        # iteratively learn a DNF clause for 1, ..., k iterations
        for idx in range(k):
            self._fit_start_time = time()
                

            
            # yTrain_orig = yTrain.copy()
            
            if(verbose):
                print("\n\n\n")
                print(idx)
                print("total samples:", len(yTrain))
                print("Time left:", self.timeOut - time() + self._fit_start_time)




            
            # # decide target class, at this point, the problem reduces to binary classification
            # target_class = np.argmax(np.bincount(yTrain))
            # self.clause_target.append(target_class)
            # yTrain = (yTrain == target_class).astype(int)
            

            self.iterations = max(2**math.floor(math.log2(len(XTrain)/sample_size)),1)
            if(verbose):
                print("Iterations:", self.iterations)

            
            
            best_loss = self.dataFidelity * XTrain.shape[0] + self.numFeatures * self.weightFeature
            best_loss_attribute = None
            self._assignList = []
            for outer_idx in range(num_outer_idx):

                # time check
                if(time() - self._fit_start_time > self.timeOut):
                    continue
                

                """
                    Two heuristics: 
                    1. random shuffle on batch (typically better performing)
                    2. without randomness
                """
                XTrains, yTrains = pyrulelearn.utils._numpy_partition(self, XTrain, yTrain)
                batch_order = None
                random_shuffle_batch = False
                if(random_shuffle_batch):
                    batch_order = random.sample(range(self.iterations), self.iterations)
                else:
                    batch_order = range(self.iterations)

                for each_batch in tqdm(batch_order, disable = not verbose):
                    # time check
                    if(time() - self._fit_start_time > self.timeOut):
                        continue
                    
                    if(self.verbose):
                        print("\nTraining started for batch: ", each_batch+1)
                    pyrulelearn.maxsat_wrap._learnModel(self, XTrains[each_batch], yTrains[each_batch], isTest=False)

                    # performance
                    self._learn_parameter()
                    yhat = self.predict(XTrain)
                    acc = accuracy_score(yTrain, yhat)
                    def _loss(acc, num_sample, rule_size):
                        return (1-acc) * self.dataFidelity * num_sample + rule_size * self.weightFeature
                    loss = _loss(acc, XTrain.shape[0], len(self._selectedFeatureIndex))
                    if(loss <= best_loss):
                        # print()
                        # print(acc, len(self._selectedFeatureIndex))
                        # print(loss)
                        best_loss = loss
                        best_loss_attribute = (self._xhat, self._selectedFeatureIndex, self._assignList)
                    else:
                        if(best_loss_attribute is not None):
                            self._assignList = best_loss_attribute[2]

                if(self.iterations == 1):
                    # When iteration = 1, training accuracy is optimized. So there is no point to iterate again
                    break
    
                # print()

            assert best_loss_attribute is not None
            # print("Best accuracy:", best_loss*len(XTrain))

            

            self._xhat, self._selectedFeatureIndex, self._assignList = best_loss_attribute 
            # print("Best:", best_loss)
                    

            
            # If learned rule is empty, it can be discarded
            if(self._xhat[0].sum() == 0):
                if(verbose):
                    print("Terminating becuase current rule is empty")
                break
            else:
                # TODO reorder self_xhat
                xhat_computed.append(self._xhat[0])
                selectedFeatureIndex_computed += [val + idx * self.numFeatures for val in self._selectedFeatureIndex]

            # print(classification_report(yTrain, yhat, target_names=np.unique(yTrain).astype("str")))
            # print(accuracy_score(yTrain, yhat))
            
            
            # remove samples that are covered by current clause. 
            # Depending on CNF/DNF, definition of coverage is different
            """
                When yhat is different than 0 (i.e., does not satisfy the IF condition), it is still considered in the next iteration because 
                the final rule is nested if-else.
            """
            self._learn_parameter()
            yhat = self.predict(XTrain)
            if(self.ruleType == "CNF"):
                mask = (yhat == 1)
            else:
                mask = (yhat == 0)

            if(verbose):    
                print("Coverage:", len(yTrain[~mask]) , "samples")

            
            # If no sample is removed, next iteration will generate the same hypothesis, hence the process is terminated
            if(len(yTrain[~mask])  == 0):
                if(verbose):
                    print("Terminating becuase no new sample is removed by current rule")
                break

            XTrain = XTrain[mask]
            yTrain = yTrain[mask]


        
        
        
        
        # Get back to initial configuration
        self.numClause = len(xhat_computed)
        self._xhat = np.array(xhat_computed)
        self._selectedFeatureIndex = selectedFeatureIndex_computed
        self.verbose = verbose
        

        # parameters learned for rule
        self._learn_parameter()

        
    
    def _fit_decision_lists(self, XTrain, yTrain):

        num_outer_idx = 2

        # know which class is majority
        majority = np.argmax(np.bincount(yTrain))
        all_classes = np.unique(yTrain) # all classes in y

        # sample size is variable. Therefore we will always maintain this sample size as maximum sample size for all iterations
        sample_size = self.batchsize
        

        # Use MaxSAT-based rule learner
        ruleType_orig = self.ruleType
        self.ruleType = "DNF"
        # self.timeOut = int(self.timeOut/(num_outer_idx * self.numClause))
        self.timeOut = float(self.timeOut/self.numClause)
        k = self.numClause
        self.numClause = 1
        self.clause_target = []
        xhat_computed = []
        verbose = self.verbose
        self.verbose = False
        
        
        # iteratively learn a DNF clause for 1, ..., k iterations
        for idx in range(k):
            self._fit_start_time = time()
                

            
            yTrain_orig = yTrain.copy()
            
            if(verbose):
                print("\n\n\n")
                print(idx)
                print("total samples:", len(yTrain))
                print("Time left:", self.timeOut - time() + self._fit_start_time)




            
            # decide target class, at this point, the problem reduces to binary classification
            target_class = np.argmax(np.bincount(yTrain))
            self.clause_target.append(target_class)
            yTrain = (yTrain == target_class).astype(int)
            

            self.iterations = max(2**math.floor(math.log2(len(XTrain)/sample_size)),1)
            if(verbose):
                print("Iterations:", self.iterations)

            
            
            best_loss = self.dataFidelity * XTrain.shape[0] + self.numFeatures * self.weightFeature
            best_loss_attribute = None
            self._assignList = []
            for outer_idx in range(num_outer_idx):

                # time check
                if(time() - self._fit_start_time > self.timeOut):
                    continue
                

                """
                    Two heuristics: 
                    1. random shuffle on batch (typically better performing)
                    2. without randomness
                """
                XTrains, yTrains = pyrulelearn.utils._numpy_partition(self, XTrain, yTrain)
                batch_order = None
                random_shuffle_batch = False
                if(random_shuffle_batch):
                    batch_order = random.sample(range(self.iterations), self.iterations)
                else:
                    batch_order = range(self.iterations)

                for each_batch in tqdm(batch_order, disable = not verbose):
                    # time check
                    if(time() - self._fit_start_time > self.timeOut):
                        continue
                    
                    if(self.verbose):
                        print("\nTraining started for batch: ", each_batch+1)
                    pyrulelearn.maxsat_wrap._learnModel(self, XTrains[each_batch], yTrains[each_batch], isTest=False)

                    
                    # performance
                    self._learn_parameter()
                    yhat = self.predict(XTrain)
                    acc = accuracy_score(yTrain, yhat)
                    def _loss(acc, num_sample, rule_size):
                        return (1-acc) * self.dataFidelity * num_sample + rule_size * self.weightFeature
                    loss = _loss(acc, XTrain.shape[0], len(self._selectedFeatureIndex))
                    if(loss <= best_loss):
                        # print()
                        # print(acc, len(self._selectedFeatureIndex))
                        # print(loss)
                        best_loss = loss
                        best_loss_attribute = (self._xhat, self._selectedFeatureIndex, self._assignList)
                    else:
                        if(best_loss_attribute is not None):
                            self._assignList = best_loss_attribute[2]


                if(self.iterations == 1):
                    # When iteration = 1, training accuracy is optimized. So there is no point to iterate again
                    break
    
                # print()

            assert best_loss_attribute is not None
            # print("Best accuracy:", best_loss*len(XTrain))

            

            self._xhat, self._selectedFeatureIndex, self._assignList = best_loss_attribute 
            # print("Best:", best_loss)
                    

            
            # If learned rule is empty, it can be discarded
            if(self._xhat[0].sum() == 0):
                if(len(self.clause_target) > 0):
                    self.clause_target = self.clause_target[:-1]
                if(verbose):
                    print("Terminating becuase current rule is empty")
                break
            else:
                # TODO reorder self_xhat
                xhat_computed.append(self._xhat[0])

            # print(classification_report(yTrain, yhat, target_names=np.unique(yTrain).astype("str")))
            # print(accuracy_score(yTrain, yhat))
            
            
            # remove samples that are covered by current DNF clause            
            """
                When yhat is different than 0 (i.e., does not satisfy the IF condition), it is still considered in the next iteration because 
                the final rule is nested if-else.
            """
            self._learn_parameter()
            yhat = self.predict(XTrain)
            mask = (yhat == 0)
            XTrain = XTrain[mask]
            yTrain = yTrain_orig[mask]
            if(verbose):    
                print("Coverage:", len(yTrain_orig[~mask]) , "samples")
            
            # If no sample is removed, next iteration will generate the same hypothesis, hence the process is terminated
            if(len(yTrain_orig[~mask])  == 0):
                if(verbose):
                    print("Terminating becuase no new sample is removed by current rule")
                break

        
        
        """
        Default rule
        """
        xhat_computed.append(np.zeros(self.numFeatures))
        reach_once = False
        for each_class in all_classes:
            if(each_class not in self.clause_target):
                self.clause_target.append(each_class)
                reach_once = True
                break
        if(not reach_once):
            self.clause_target.append(majority)

        
        # Get back to initial configuration
        self.numClause = len(self.clause_target)
        self._xhat = xhat_computed
        self.ruleType = ruleType_orig
        self.verbose = verbose


        # parameters learned for rule
        self.threshold_literal_learned = [selected_columns.sum() for selected_columns in self._xhat]
        self.threshold_clause_learned = None

        # print(self.clause_target)
        # print(self._xhat)
        # print(self.threshold_clause_learned, self.threshold_literal_learned)

    
    
    def fit(self, XTrain, yTrain, recursive=True):


        self._fit_mode = True

        self._fit_start_time = time()    
        XTrain = pyrulelearn.utils._transform_binary_matrix(XTrain)
        yTrain = np.array(yTrain, dtype=bool)
        
            

            

        if(self.ruleType not in ["CNF", "DNF", "relaxed_CNF", "decision lists", "decision sets"]):
            raise ValueError(self.ruleType)

        self.trainingSize = XTrain.shape[0]
        if(self.trainingSize > 0):
            self.numFeatures = len(XTrain[0])
        if(self.trainingSize < self.batchsize):
            self.batchsize = self.trainingSize



        if(self.ruleType == "relaxed_CNF"):
            self._fit_relaxed_CNF(XTrain, yTrain)
            self._fit_mode = False
            return

        if(self.ruleType == "decision lists"):
            self._fit_decision_lists(XTrain, yTrain)
            self._fit_mode = False
            return

        if(self.ruleType == "decision sets"):
            self._fit_decision_sets(XTrain, yTrain)
            self._fit_mode = False
            return


        if(recursive):
            self._fit_CNF_DNF_recursive(XTrain, yTrain)
            self._fit_mode = False
            return


        



        self.iterations = 2**math.floor(math.log2(XTrain.shape[0]/self.batchsize))
        
        best_loss = self.dataFidelity * XTrain.shape[0] + self.numFeatures * self.weightFeature * self.numClause
        best_loss_attribute = None
        num_outer_idx = 2
        cnt = 0
        self._assignList = []
        for outer_idx in range(num_outer_idx):

            # time check
            if(time() - self._fit_start_time > self.timeOut):
                continue
            

            XTrains, yTrains = pyrulelearn.utils._numpy_partition(self, XTrain, yTrain)
            batch_order = None
            random_shuffle_batch = False
            if(random_shuffle_batch):
                batch_order = random.sample(range(self.iterations), self.iterations)
            else:
                batch_order = range(self.iterations)

            for each_batch in tqdm(batch_order, disable = not self.verbose):
                # time check
                if(time() - self._fit_start_time > self.timeOut):
                    continue
                
                if(self.verbose):
                    print("\nTraining started for batch: ", each_batch+1)
                pyrulelearn.maxsat_wrap._learnModel(self, XTrains[each_batch], yTrains[each_batch], isTest=False)

                

                # performance
                cnt += 1
                self._learn_parameter()
                yhat = self.predict(XTrain)
                acc = accuracy_score(yTrain, yhat)
                def _loss(acc, num_sample, rule_size):
                    assert rule_size <= self.numFeatures
                    return (1-acc) * self.dataFidelity * num_sample + rule_size * self.weightFeature
                loss = _loss(acc, XTrain.shape[0], len(self._selectedFeatureIndex))
                if(loss <= best_loss):
                    # print()
                    # print(acc, len(self._selectedFeatureIndex))
                    # print(loss)
                    best_loss = loss
                    best_loss_attribute = (self._xhat, self._selectedFeatureIndex, self._assignList)
                else:
                    if(best_loss_attribute is not None):
                        self._assignList = best_loss_attribute[2]
                
                

            if(self.iterations == 1):
                # When iteration = 1, training accuracy is optimized. So there is no point to iterate again
                break

       
        assert best_loss_attribute is not None
        self._xhat, self._selectedFeatureIndex, self._assignList = best_loss_attribute 
        self._learn_parameter()
        self._fit_mode = False
        return 

        

        
        

    def predict(self, XTest):

        if(not self._fit_mode):
            XTest = pyrulelearn.utils._transform_binary_matrix(XTest)
            # XTest = pyrulelearn.utils._add_dummy_columns(XTest)
        assert self.numFeatures == XTest.shape[1], str(self.numFeatures) + " " + str(XTest.shape[1])
        

        y_hat = []
        self.coverage = []
        if(self.ruleType in ["CNF", "DNF", "relaxed_CNF"]):
            
            """
                Expensive
            """
            if(False):
                yTest = [1 for _ in XTest]
                for i in range(len(yTest)):
                    dot_value = [0 for eachLevel in range(self.numClause)]
                    for eachLevel in range(self.numClause):
                        if(self.ruleType == "relaxed_CNF"):
                            dot_value[eachLevel] = np.dot(XTest[i], np.array(self._assignList[eachLevel * self.numFeatures: (eachLevel + 1) * self.numFeatures ]))
                        elif(self.ruleType in ['CNF', 'DNF']):
                            dot_value[eachLevel] = np.dot(XTest[i], self._xhat[eachLevel])
                        else:
                            raise ValueError

                    if (yTest[i] == 1):
                        correctClauseCount = 0
                        for eachLevel in range(self.numClause):
                            if (dot_value[eachLevel] >= self.threshold_literal_learned[eachLevel]):
                                correctClauseCount += 1
                        if (correctClauseCount >= self.threshold_clause_learned):
                            y_hat.append(1)
                        else:
                            y_hat.append(0)

                    else:
                        correctClauseCount = 0
                        for eachLevel in range(self.numClause):
                            if (dot_value[eachLevel] < self.threshold_literal_learned[eachLevel]):
                                correctClauseCount += 1
                        if (correctClauseCount > self.numClause - self.threshold_clause_learned):
                            y_hat.append(0)
                        else:
                            y_hat.append(1)

                
            # Matrix multiplication
            if(True):
                if(self.ruleType == "relaxed_CNF"):
                    self._xhat = np.array(self._assignList[:self.numClause * self.numFeatures]).reshape(self.numClause, self.numFeatures)


                # dot_matrix = XTest.dot(self._xhat.T)
                # y_hat = ((dot_matrix >= np.array(self.threshold_literal_learned)).sum(axis=1) >= self.threshold_clause_learned).astype(int)

                # considers non zero columns only
                start_prediction_time = time()
                nonzero_columns = np.nonzero(np.any(self._xhat, axis=0))[0]
                dot_matrix = XTest[:, nonzero_columns].dot(self._xhat[:, nonzero_columns].T)
                y_hat = ((dot_matrix >= np.array(self.threshold_literal_learned)).sum(axis=1) >= self.threshold_clause_learned).astype(int)
                self._prediction_time += time() - start_prediction_time
        
                
                # assert np.array_equal(y_hat_, y_hat)



        elif(self.ruleType in ["decision lists", "decision sets"]):
            
            cnt_voting_function = 0
            cnt_reach_default_rule = 0

            self.coverage = [0 for _ in range(self.numClause)]

            assert len(self.clause_target) == self.numClause
            for example in XTest:
                reached_verdict = False
                possible_outcome = []
                for eachLevel in range(self.numClause):
                    dot_value = np.dot(example, self._xhat[eachLevel])
                    assert dot_value <= self.threshold_literal_learned[eachLevel]
                    if(dot_value == self.threshold_literal_learned[eachLevel]):
                        reached_verdict = True
                        self.coverage[eachLevel] += 1
                        if(self.ruleType == "decision lists"):
                            y_hat.append(self.clause_target[eachLevel])
                            if(eachLevel == self.numClause - 1):
                                cnt_reach_default_rule += 1
                            
                            break
                        possible_outcome.append(self.clause_target[eachLevel])

                        
                assert reached_verdict
                if(self.ruleType == "decision sets"):
                    # refine possible outcomes
                    default_outcome = possible_outcome[-1]
                    possible_outcome = possible_outcome[:-1]
                    if(len(possible_outcome) > 0):
                        # most frequent
                        cnt_voting_function += 1
                        y_hat.append(max(set(possible_outcome), key = possible_outcome.count))
                    else:
                        cnt_reach_default_rule += 1
                        y_hat.append(default_outcome)
            
            if(self.verbose):
                print("\n")
                print("Voting function:", cnt_voting_function)
                print("Default rule:", cnt_reach_default_rule)
                print("Coverage:", self.coverage)



        else:
            raise ValueError(self.ruleType)

        # y_hat = np.array(y_hat)
        return y_hat

        
        
        # if(self.verbose):
        #     print("\nPrediction through MaxSAT formulation")
        # predictions = self.__learnModel(XTest, yTest, isTest=True)
        # yhat = []
        # for i in range(len(predictions)):
        #     if (int(predictions[i]) > 0):
        #         yhat.append(1 - yTest[i])
        #     else:
        #         yhat.append(yTest[i])
        # return yhat

    
    def _learn_parameter(self):
        # parameters learned for rule
        if(self.ruleType=="CNF"):
            self.threshold_literal_learned = [1 for i in range(self.numClause)]
        elif(self.ruleType=="DNF"):
            self.threshold_literal_learned = [len(selected_columns) for selected_columns in self._get_selected_column_index()]
        else:
            raise ValueError

        if(self.ruleType=="CNF"):
            self.threshold_clause_learned = self.numClause
        elif(self.ruleType=="DNF"):
            self.threshold_clause_learned = 1
        else:
            raise ValueError
    
    
    def get_rule(self, features, show_decision_lists=False):

        if(2 * len(features) == self.numFeatures):
            features = [str(feature) for feature in features]
            features += ["not " + str(feature) for feature in features]
            
        assert len(features) == self.numFeatures

        if(self.ruleType == "relaxed_CNF"):  # naive copy paste
            no_features = len(features)
            # self.rule_size = 0
            rule = '[ ( '
            for eachLevel in range(self.numClause):

                for literal_index in range(no_features):
                    if (self._assignList[eachLevel * no_features + literal_index] >= 1):
                        rule += " " + features[literal_index] + "  +"
                rule = rule[:-1]
                rule += ' )>= ' + str(self.threshold_literal_learned[eachLevel]) + "  ]"

                if (eachLevel < self.numClause - 1):
                    rule += ' +\n[ ( '
            rule += "  >= " + str(self.threshold_clause_learned)


            return rule
        else:

            self.names = []
            for i in range(self.numClause):
                xHatElem = self._xhat[i]
                inds_nnz = np.where(abs(xHatElem) > 1e-4)[0]
                self.names.append([features[ind] for ind in inds_nnz])

            
            if(self.ruleType == "CNF"):
                return " AND\n".join([" OR ".join(name) for name in self.names])
            elif(self.ruleType == "DNF"):
                if(not show_decision_lists):
                    return " OR\n".join([" AND ".join(name) for name in self.names])
                else:
                    return "\n".join([("If " if idx == 0 else ("Else if " if len(name) > 0 else "Else")) +  " AND ".join(name) + ": class = 1" for idx, name in enumerate(self.names)])
            elif(self.ruleType == "decision lists"):
                
                #TODO Can intermediate rule be empty?
                
                return "\n".join([("If " if idx == 0 else ("Else if " if len(name) > 0 else "Else")) +  " AND ".join(name) + ": class = " + str(self.clause_target[idx]) for idx, name in enumerate(self.names)])
            elif(self.ruleType == "decision sets"):
                return "\n".join([("If " if idx < len(self.names) - 1 else "Else ") +  " AND ".join(name) + ": class = " + str(self.clause_target[idx]) for idx, name in enumerate(self.names)])
            else:
                raise ValueError

            