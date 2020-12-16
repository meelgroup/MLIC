

# Contact: Bishwamittra Ghosh [email: bghosh@u.nus.edu]

import numpy as np
import pandas as pd
import warnings
import math
import os
from sklearn.model_selection import train_test_split
import random
import subprocess
warnings.simplefilter(action='ignore', category=FutureWarning)


class imli():
    def __init__(self, iterations=-1, num_clause=1, data_fidelity=10, weight_feature=1, threshold_literal=-1, threshold_clause=-1,
                 solver="open-wbo", rule_type="CNF", samplesize=0.5,
                 work_dir=".", time_out=1024, verbose=False):
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

        
        self.iterations = iterations
        self.numClause = num_clause
        self.dataFidelity = data_fidelity
        self.weightFeature = weight_feature
        self.solver = solver
        self.ruleType = rule_type
        self.workDir = work_dir
        self.verbose = verbose
        self.__selectedFeatureIndex = []
        self.timeOut = time_out
        self.memlimit = 1000*16
        self.__isEntropyBasedDiscretization = False
        self.learn_threshold_literal = False
        self.learn_threshold_clause = False
        self.threshold_literal = threshold_literal
        self.threshold_clause = threshold_clause
        self._imli__columnInfo = None


        if(self.ruleType!="CNF"  and self.ruleType!="DNF" and self.ruleType!="relaxed_CNF"):
            print("\nError rule type. Choices are [CNF, DNF, relaxed_CNF]")
            return


        if(self.ruleType == "relaxed_CNF"):
            self.solver = "cplex"  # this is the default solver for learning rules in relaxed_CNFs
            self.samplesize = samplesize

    def discretize_orange(self, csv_file):
        import Orange
        self.__isEntropyBasedDiscretization = True
        data = Orange.data.Table(csv_file)
        # Run impute operation for handling missing values
        imputer = Orange.preprocess.Impute()
        data = imputer(data)
        # Discretize datasets
        discretizer = Orange.preprocess.Discretize()
        discretizer.method = Orange.preprocess.discretize.EntropyMDL(
            force=False)
        discetized_data = discretizer(data)
        categorical_columns = [elem.name for elem in discetized_data.domain[:-1]]
        # Apply one hot encoding on X (Using continuizer of Orange)
        continuizer = Orange.preprocess.Continuize()
        binarized_data = continuizer(discetized_data)
        
        # make another level of binarization
        X=[]
        for sample in binarized_data.X:
            X.append([int(feature) for feature in sample]+ [int(1-feature) for feature in sample])
    

        columns = []
        for i in range(len(binarized_data.domain)-1):
            column = binarized_data.domain[i].name
            if("<" in column):
                column = column.replace("=<", "_l_")
            elif("≥" in column):
                column = column.replace("=≥", "_ge_")
            elif("=" in column):
                if("-" in column):
                    column = column.replace("=", "_eq_(")
                    column = column+")"
                else:
                    column = column.replace("=", "_eq_")
                    column = column
            columns.append(column)
        # print(self.columns)

        # make negated columns
        num_features=len(columns)
        for index in range(num_features):
            columns.append("not_"+columns[index])
        # print(self.columns)
    

        if(self.verbose):
            print("Applying entropy based discretization using Orange library")
            print("- file name: ", csv_file)
            print("- the number of discretized features:", len(columns))

        return np.array(X), np.array([int(value) for value in binarized_data.Y]),  columns

    def __repr__(self):
        print("\n\nIMLI:->")
        return '\n'.join(" - %s: %s" % (item, value) for (item, value) in vars(self).items() if "__" not in item)

    def get_selected_column_index(self):
        return_list = [[] for i in range(self.numClause)]
        ySize = self.numFeatures
        for elem in self.__selectedFeatureIndex:
            new_index = int(elem)-1
            return_list[int(new_index/ySize)].append(new_index % ySize)
        return return_list

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
        if(self.ruleType=="relaxed_CNF"):
            return self.threshold_literal_learned
        elif(self.ruleType=="CNF"):
            return [1 for i in range(self.numClause)]
        elif(self.ruleType=="DNF"):
            return [len(selected_columns) for selected_columns in self.get_selected_column_index()]
        else:
            return

    def get_threshold_clause(self):
        if(self.ruleType=="relaxed_CNF"):
            return self.threshold_clause_learned
        elif(self.ruleType=="CNF"):
            return self.numClause
        elif(self.ruleType=="DNF"):
            return 1
        else:
            return


    def discretize(self, file, categorical_column_index=[], column_seperator=",", frac_present=0.9, num_thresholds=4):

        # Quantile probabilities
        quantProb = np.linspace(1. / (num_thresholds + 1.), num_thresholds / (num_thresholds + 1.), num_thresholds)
        # List of categorical columns
        if type(categorical_column_index) is pd.Series:
            categorical_column_index = categorical_column_index.tolist()
        elif type(categorical_column_index) is not list:
            categorical_column_index = [categorical_column_index]
        data = pd.read_csv(file, sep=column_seperator, header=0, error_bad_lines=False)

        columns = data.columns
        # if (self.verbose):˚
        #     print(data)
        #     print(columns)
        #     print(categorical_column_index)
        if (self.verbose):
            print("\n\nApplying quantile based discretization")
            print("- file name: ", file)
            print("- categorical features index: ", categorical_column_index)
            print("- number of bins: ", num_thresholds)
            # print("- features: ", columns)
            print("- number of features:", len(columns))

        # if (self.verbose):
        #     print(data.columns)

        columnY = columns[-1]

        data.dropna(axis=1, thresh=frac_present * len(data), inplace=True)
        data.dropna(axis=0, how='any', inplace=True)

        y = data.pop(columnY).copy()

        # Initialize dataframe and thresholds
        X = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))
        thresh = {}
        column_counter = 1
        self.__columnInfo = []
        # Iterate over columns
        count = 0
        for c in data:
            # number of unique values
            valUniq = data[c].nunique()

            # Constant column --- discard
            if valUniq < 2:
                continue

            # Binary column
            elif valUniq == 2:
                # Rename values to 0, 1
                X[('is', c, '')] = data[c].replace(np.sort(data[c].unique()), [0, 1])
                X[('is not', c, '')] = data[c].replace(np.sort(data[c].unique()), [1, 0])

                temp = [1, column_counter, column_counter + 1]
                self.__columnInfo.append(temp)
                column_counter += 2

            # Categorical column
            elif (count in categorical_column_index) or (data[c].dtype == 'object'):
                # if (self.verbose):
                #     print(c)
                #     print(c in categorical_column_index)
                #     print(data[c].dtype)
                # Dummy-code values
                Anew = pd.get_dummies(data[c]).astype(int)
                Anew.columns = Anew.columns.astype(str)
                # Append negations
                Anew = pd.concat([Anew, 1 - Anew], axis=1, keys=[(c, '=='), (c, '!=')])
                # Concatenate
                X = pd.concat([X, Anew], axis=1)

                temp = [2, column_counter, column_counter + 1]
                self.__columnInfo.append(temp)
                column_counter += 2

            # Ordinal column
            elif np.issubdtype(data[c].dtype, int) | np.issubdtype(data[c].dtype, float):
                # Few unique values
                # if (self.verbose):
                #     print(data[c].dtype)
                if valUniq <= num_thresholds + 1:
                    # Thresholds are sorted unique values excluding maximum
                    thresh[c] = np.sort(data[c].unique())[:-1]
                # Many unique values
                else:
                    # Thresholds are quantiles excluding repetitions
                    thresh[c] = data[c].quantile(q=quantProb).unique()
                # Threshold values to produce binary arrays
                Anew = (data[c].values[:, np.newaxis] <= thresh[c]).astype(int)
                Anew = np.concatenate((Anew, 1 - Anew), axis=1)
                # Convert to dataframe with column labels
                Anew = pd.DataFrame(Anew,
                                    columns=pd.MultiIndex.from_product([[c], ['<=', '>'], thresh[c].astype(str)]))
                # Concatenate
                # print(A.shape)
                # print(Anew.shape)
                X = pd.concat([X, Anew], axis=1)

                addedColumn = len(Anew.columns)
                addedColumn = int(addedColumn / 2)
                temp = [3]
                temp = temp + [column_counter + nc for nc in range(addedColumn)]
                column_counter += addedColumn
                self.__columnInfo.append(temp)
                temp = [4]
                temp = temp + [column_counter + nc for nc in range(addedColumn)]
                column_counter += addedColumn
                self.__columnInfo.append(temp)
            else:
                # print(("Skipping column '" + c + "': data type cannot be handled"))
                continue
            count += 1

        if(self.verbose):
            print("\n\nAfter applying discretization")
            print("- number of discretized features: ", len(X.columns))
        return X.values, y.values.ravel(), X.columns

    def __fit_relaxed_CNF(self, XTrain, yTrain):

        
        if (self.threshold_clause == -1):
            self.learn_threshold_clause = True
        if (self.threshold_literal == -1):
            self.learn_threshold_literal = True

        if(self.iterations==-1): # when not specified
            self.iterations = int(math.ceil(1/self.samplesize))
            
        self.trainingSize = len(XTrain)
        self.__assignList = []
        
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
            XTrain_sampled, yTrain_sampled = self.__generatSamples(XTrain, yTrain)

            assert len(XTrain[0]) == len(XTrain_sampled[0])

            self.__call_cplex(XTrain_sampled, yTrain_sampled)

    def fit(self, XTrain, yTrain):

        if(self.ruleType!="CNF"  and self.ruleType!="DNF" and self.ruleType!="relaxed_CNF"):
            print("\n\nError rule type. Choices are [CNF, DNF, relaxed_CNF]")
            return

        self.trainingSize = len(XTrain)
        if(self.trainingSize > 0):
            self.numFeatures = len(XTrain[0])


        if(self.ruleType == "relaxed_CNF"):
            self.__fit_relaxed_CNF(XTrain, yTrain)
            return

        if(self.iterations == -1):
            self.iterations = 2**math.floor(math.log2(len(XTrain)/32))
            # print("Batchs:" + str(self.iterations))

        

        XTrains, yTrains = self.__getBatchWithEqualProbability(XTrain, yTrain)

        self.__assignList = []
        for each_batch in range(self.iterations):
            if(self.verbose):
                print("\nTraining started for batch: ", each_batch+1)
            self.__learnModel(XTrains[each_batch], yTrains[each_batch], isTest=False)

    def predict(self, XTest, yTest):

        if(self.ruleType == "relaxed_CNF"):
            y_hat = []
            for i in range(len(yTest)):
                dot_value = [0 for eachLevel in range(self.numClause)]
                for j in range(len(XTest[i])):
                    for eachLevel in range(self.numClause):
                        dot_value[eachLevel] += XTest[i][j] * \
                            self.__assignList[eachLevel * len(XTest[i]) + j]
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
            return y_hat

        if(self.verbose):
            print("\nPrediction through MaxSAT formulation")
        predictions = self.__learnModel(XTest, yTest, isTest=True)
        yhat = []
        for i in range(len(predictions)):
            if (int(predictions[i]) > 0):
                yhat.append(1 - yTest[i])
            else:
                yhat.append(yTest[i])
        return yhat

    def _cmd_exists(self, cmd):
        return subprocess.call("type " + cmd, shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0

    def __learnModel(self, X, y, isTest):
        # temp files to save maxsat query in wcnf format
        WCNFFile = self.workDir + "/" + "model.wcnf"
        outputFileMaxsat = self.workDir + "/" + "model_out.txt"

        # generate maxsat query for dataset
        if (self.ruleType == 'DNF'):
            #  negate yVector for DNF rules
            self.__generateWcnfFile(X, [1 - int(y[each_y]) for each_y in
                                        range(len(y))],
                                    len(X[0]), WCNFFile,
                                    isTest)

        elif(self.ruleType == "CNF"):
            self.__generateWcnfFile(X, y, len(X[0]),
                                    WCNFFile,
                                    isTest)
        else:
            print("\n\nError rule type")

        # call a maxsat solver
        if(self.solver == "open-wbo" or "maxhs"):  # solver has timeout and experimented with open-wbo only
            if(self._cmd_exists(self.solver)):
                if(self.iterations == -1):
                    cmd = self.solver + '   ' + WCNFFile + ' -cpu-lim=' + str(self.timeOut) + ' > ' + outputFileMaxsat
                else:
                    if(int(math.ceil(self.timeOut/self.iterations)) < 1):  # give at lest 1 second as cpu-lim
                        cmd = self.solver + '   ' + WCNFFile + ' -cpu-lim=' + str(1) + ' > ' + outputFileMaxsat
                    else:
                        cmd = self.solver + '   ' + WCNFFile + ' -cpu-lim=' + str(int(math.ceil(self.timeOut/self.iterations))) + ' > ' + outputFileMaxsat
                        # print(int(math.ceil(self.timeOut/self.iterations)))
            else:
                raise Exception("Solver not found")   
        else:
            cmd = self.solver + '   ' + WCNFFile + ' > ' + outputFileMaxsat

        os.system(cmd)

        # delete temp files
        cmd = "rm " + WCNFFile
        os.system(cmd)

        # parse result of maxsat solving
        f = open(outputFileMaxsat, 'r')
        lines = f.readlines()
        f.close()
        solution = ''
        for line in lines:
            if (line.strip().startswith('v')):
                solution = line.strip().strip('v ')
                break

        fields = solution.split()
        TrueRules = []
        TrueErrors = []
        zeroOneSolution = []

        if(not self.__isEntropyBasedDiscretization and self._imli__columnInfo is not None):
            fields = self.__pruneRules(fields, len(X[0]))

        for field in fields:
            if (int(field) > 0):
                zeroOneSolution.append(1.0)
            else:
                zeroOneSolution.append(0.0)
            if (int(field) > 0):

                if (abs(int(field)) <= self.numClause * len(X[0])):

                    TrueRules.append(field)
                elif (self.numClause * len(X[0]) < abs(int(field)) <= self.numClause * len(
                        X[0]) + len(y)):
                    TrueErrors.append(field)

        if (self.verbose and isTest == False):
            print("\n\nBatch tarining complete")
            print("- number of literals in the rule: " + str(len(TrueRules)))
            print("- number of training errors:    " + str(len(TrueErrors)) + " out of " + str(len(y)))
        self.__xhat = []

        for i in range(self.numClause):
            self.__xhat.append(np.array(
                zeroOneSolution[i * len(X[0]):(i + 1) * len(X[0])]))
        err = np.array(zeroOneSolution[len(X[0]) * self.numClause: len(
            X[0]) * self.numClause + len(y)])

        # delete temp files
        cmd = "rm " + outputFileMaxsat
        os.system(cmd)

        if (not isTest):
            self.__assignList = fields[:self.numClause * len(X[0])]
            self.__selectedFeatureIndex = TrueRules

            # print(self.__selectedFeatureIndex)

        return fields[self.numClause * len(X[0]):len(y) + self.numClause * len(X[0])]

    def __pruneRules(self, fields, xSize):
        # algorithm 1 in paper

        new_fileds = fields
        end_of_column_list = [self.__columnInfo[i][-1] for i in range(len(self.__columnInfo))]
        freq_end_of_column_list = [[[0, 0] for i in range(len(end_of_column_list))] for j in range(self.numClause)]
        variable_contained_list = [[[] for i in range(len(end_of_column_list))] for j in range(self.numClause)]

        for i in range(self.numClause * xSize):
            if ((int(fields[i])) > 0):
                variable = (int(fields[i]) - 1) % xSize + 1
                clause_position = int((int(fields[i]) - 1) / xSize)
                for j in range(len(end_of_column_list)):
                    if (variable <= end_of_column_list[j]):
                        variable_contained_list[clause_position][j].append(clause_position * xSize + variable)
                        freq_end_of_column_list[clause_position][j][0] += 1
                        freq_end_of_column_list[clause_position][j][1] = self.__columnInfo[j][0]
                        break
        for l in range(self.numClause):

            for i in range(len(freq_end_of_column_list[l])):
                if (freq_end_of_column_list[l][i][0] > 1):
                    if (freq_end_of_column_list[l][i][1] == 3):
                        variable_contained_list[l][i] = variable_contained_list[l][i][:-1]
                        for j in range(len(variable_contained_list[l][i])):
                            new_fileds[variable_contained_list[l][i][j] - 1] = "-" + str(
                                variable_contained_list[l][i][j])
                    elif (freq_end_of_column_list[l][i][1] == 4):
                        variable_contained_list[l][i] = variable_contained_list[l][i][1:]
                        for j in range(len(variable_contained_list[l][i])):
                            new_fileds[variable_contained_list[l][i][j] - 1] = "-" + str(
                                variable_contained_list[l][i][j])
        return new_fileds

    def __getBatchWithEqualProbability(self, X, y):
        '''
            Steps:
                1. seperate data based on class value
                2. Batch each seperate data into Batch_count batches using test_train_split method with 50% part in each
                3. merge one seperate batche from each class and save
            :param X:
            :param y:
            :param Batch_count:
            :param location:
            :param file_name_header:
            :param column_set_list: uses for incremental approach
            :return:
            '''
        Batch_count = self.iterations
        # y = y.values.ravel()
        max_y = int(y.max())
        min_y = int(y.min())

        X_list = [[] for i in range(max_y - min_y + 1)]
        y_list = [[] for i in range(max_y - min_y + 1)]
        level = int(math.log(Batch_count, 2.0))
        for i in range(len(y)):
            inserting_index = int(y[i])
            y_list[inserting_index - min_y].append(y[i])
            X_list[inserting_index - min_y].append(X[i])

        final_Batch_X_train = [[] for i in range(Batch_count)]
        final_Batch_y_train = [[] for i in range(Batch_count)]
        for each_class in range(len(X_list)):
            Batch_list_X_train = [X_list[each_class]]
            Batch_list_y_train = [y_list[each_class]]

            for i in range(level):
                for j in range(int(math.pow(2, i))):
                    A_train_1, A_train_2, y_train_1, y_train_2 = train_test_split(
                        Batch_list_X_train[int(math.pow(2, i)) + j - 1],
                        Batch_list_y_train[int(math.pow(2, i)) + j - 1],
                        test_size=0.5,
                        random_state = 22)  # random state for keeping consistency between lp and maxsat approach
                    Batch_list_X_train.append(A_train_1)
                    Batch_list_X_train.append(A_train_2)
                    Batch_list_y_train.append(y_train_1)
                    Batch_list_y_train.append(y_train_2)

            Batch_list_y_train = Batch_list_y_train[Batch_count - 1:]
            Batch_list_X_train = Batch_list_X_train[Batch_count - 1:]

            for i in range(Batch_count):
                final_Batch_y_train[i] = final_Batch_y_train[i] + Batch_list_y_train[i]
                final_Batch_X_train[i] = final_Batch_X_train[i] + Batch_list_X_train[i]

        return final_Batch_X_train[:Batch_count], final_Batch_y_train[:Batch_count]

    def __learnSoftClauses(self, isTestPhase, xSize, yVector):
        cnfClauses = ''
        numClauses = 0

        if (isTestPhase):
            topWeight = self.dataFidelity * len(yVector) + 1 + self.weightFeature * xSize * self.numClause
            numClauses = 0
            for i in range(1, self.numClause * xSize + 1):
                numClauses += 1
                cnfClauses += str(self.weightFeature) + ' ' + str(-i) + ' 0\n'
            for i in range(self.numClause * xSize + 1, self.numClause * xSize + len(yVector) + 1):
                numClauses += 1
                cnfClauses += str(self.dataFidelity) + ' ' + str(-i) + ' 0\n'

            # for testing, the positive assigned feature variables are converted to hard clauses
            # so that  their assignment is kept consistent and only noise variables are considered soft,
            for each_assign in self.__assignList:
                numClauses += 1
                cnfClauses += str(topWeight) + ' ' + each_assign + ' 0\n'
        else:
            # applicable for the 1st Batch
            isEmptyAssignList = True

            total_additional_weight = 0
            positiveLiteralWeight = self.weightFeature
            for each_assign in self.__assignList:
                isEmptyAssignList = False
                numClauses += 1
                if (int(each_assign) > 0):

                    cnfClauses += str(positiveLiteralWeight) + ' ' + each_assign + ' 0\n'
                    total_additional_weight += positiveLiteralWeight

                else:
                    cnfClauses += str(self.weightFeature) + ' ' + each_assign + ' 0\n'
                    total_additional_weight += self.weightFeature

            # noise variables are to be kept consisitent (not necessary though)
            for i in range(self.numClause * xSize + 1,
                           self.numClause * xSize + len(yVector) + 1):
                numClauses += 1
                cnfClauses += str(self.dataFidelity) + ' ' + str(-i) + ' 0\n'

            # for the first step
            if (isEmptyAssignList):
                for i in range(1, self.numClause * xSize + 1):
                    numClauses += 1
                    cnfClauses += str(self.weightFeature) + ' ' + str(-i) + ' 0\n'
                    total_additional_weight += self.weightFeature

            topWeight = int(self.dataFidelity * len(yVector) + 1 + total_additional_weight)

        if(self.verbose):
            print("- number of soft clauses: ", numClauses)

        return topWeight, numClauses, cnfClauses

    def __generateWcnfFile(self, AMatrix, yVector, xSize, WCNFFile,
                           isTestPhase):
        # learn soft clauses associated with feature variables and noise variables
        topWeight, numClauses, cnfClauses = self.__learnSoftClauses(isTestPhase, xSize,
                                                                    yVector)

        # learn hard clauses,
        additionalVariable = 0
        for i in range(len(yVector)):
            noise = self.numClause * xSize + i + 1

            # implementation of tseitin encoding
            if (yVector[i] == 0):
                new_clause = str(topWeight) + " " + str(noise)
                for each_level in range(self.numClause):
                    new_clause += " " + str(additionalVariable + each_level + len(yVector) + self.numClause * xSize + 1)
                new_clause += " 0\n"
                cnfClauses += new_clause
                numClauses += 1

                for each_level in range(self.numClause):
                    for j in range(len(AMatrix[i])):
                        if (int(AMatrix[i][j]) == 1):
                            numClauses += 1
                            new_clause = str(topWeight) + " -" + str(
                                additionalVariable + each_level + len(yVector) + self.numClause * xSize + 1)

                            new_clause += " -" + str(int(j + each_level * xSize + 1))
                            new_clause += " 0\n"
                            cnfClauses += new_clause
                additionalVariable += self.numClause

            else:
                for each_level in range(self.numClause):
                    numClauses += 1
                    new_clause = str(topWeight) + " " + str(noise)
                    for j in range(len(AMatrix[i])):
                        if (int(AMatrix[i][j]) == 1):
                            new_clause += " " + str(int(j + each_level * xSize + 1))
                    new_clause += " 0\n"
                    cnfClauses += new_clause

        # write in wcnf format
        header = 'p wcnf ' + str(additionalVariable + xSize * self.numClause + (len(yVector))) + ' ' + str(
            numClauses) + ' ' + str(topWeight) + '\n'
        f = open(WCNFFile, 'w')
        f.write(header)
        f.write(cnfClauses)
        f.close()

        if(self.verbose):
            print("- number of Boolean variables:", additionalVariable + xSize * self.numClause + (len(yVector)))
            print("- number of hard and soft clauses:", numClauses)

    def get_rule(self, features):

        if(self.ruleType == "relaxed_CNF"):  # naive copy paste
            no_features = len(features)
            # self.rule_size = 0
            rule = '[ ( '
            for eachLevel in range(self.numClause):

                for literal_index in range(no_features):
                    if (self.__assignList[eachLevel * no_features + literal_index] >= 1):
                        # rule += "  X_" + str(literal_index + 1) + "  +"
                        if(self.__isEntropyBasedDiscretization):
                            rule += " " + features[literal_index] + "  +"
                        else:
                            rule += " " + ' '.join(features[literal_index]) + "  +"
                        # self.rule_size += 1
                rule = rule[:-1]
                rule += ' )>= ' + str(self.threshold_literal_learned[eachLevel]) + "  ]"

                if (eachLevel < self.numClause - 1):
                    rule += ' +\n[ ( '
            rule += "  >= " + str(self.threshold_clause_learned)

            if(self.__isEntropyBasedDiscretization):

                rule = rule.replace('_l_', ' < ')
                rule = rule.replace('_ge_', ' >= ')
                rule = rule.replace('_eq_', ' = ')

            return rule

        generatedRule = '( '
        for i in range(self.numClause):
            xHatElem = self.__xhat[i]
            inds_nnz = np.where(abs(xHatElem) > 1e-4)[0]

            if(self.__isEntropyBasedDiscretization or self._imli__columnInfo is None):
                str_clauses = [''.join(features[ind]) for ind in inds_nnz]
            else:
                str_clauses = [' '.join(features[ind]) for ind in inds_nnz]
            if (self.ruleType == "CNF"):
                rule_sep = ' %s ' % "OR"
            else:
                rule_sep = ' %s ' % "AND"
            rule_str = rule_sep.join(str_clauses)
            if (self.ruleType == 'DNF'):
                rule_str = rule_str.replace('<=', '??').replace('>', '<=').replace('??', '>')
                rule_str = rule_str.replace('==', '??').replace('!=', '==').replace('??', '!=')
                rule_str = rule_str.replace('is', '??').replace('is not', 'is').replace('??', 'is not')

            generatedRule += rule_str
            if (i < self.numClause - 1):
                if (self.ruleType == "DNF"):
                    generatedRule += ' ) OR \n( '
                if (self.ruleType == 'CNF'):
                    generatedRule += ' ) AND \n( '
        generatedRule += ')'

        if(self.__isEntropyBasedDiscretization):
            generatedRule = generatedRule.replace('_l_', ' < ')
            generatedRule = generatedRule.replace('_ge_', ' >= ')
            generatedRule = generatedRule.replace('_eq_', ' = ')

        return generatedRule

    def __generatSamples(self, XTrain, yTrain):

        num_pos_samples = sum(x > 0 for x in yTrain)

        list_of_random_index = random.sample(
            [i for i in range(num_pos_samples)], int(num_pos_samples * self.samplesize)) + random.sample(
            [i for i in range(num_pos_samples, self.trainingSize)], int((self.trainingSize - num_pos_samples) * self.samplesize))

        # print(int(self.trainingSize * self.samplesize))
        XTrain_sampled = [XTrain[i] for i in list_of_random_index]
        yTrain_sampled = [yTrain[i] for i in list_of_random_index]

        assert len(list_of_random_index) == len(set(list_of_random_index)), "sampling is not uniform"

        return XTrain_sampled, yTrain_sampled

    def __call_cplex(self, A, y):
        import cplex
        no_features = -1
        no_samples = len(y)
        if(no_samples > 0):
            no_features = len(A[0])
        else:
            print("- error: the dataset is corrupted, does not have sufficient samples")

        if (self.verbose):
            print("- no of features: ", no_features)
            print("- no of samples : ", no_samples)

        # Establish the Linear Programming Model
        myProblem = cplex.Cplex()

        feature_variable = []
        variable_list = []
        objective_coefficient = []
        variable_count = 0

        for eachLevel in range(self.numClause):
            for i in range(no_features):
                feature_variable.append(
                    "b_" + str(i + 1) + str("_") + str(eachLevel + 1))

        variable_list = variable_list + feature_variable

        slack_variable = []
        for i in range(no_samples):
            slack_variable.append("s_" + str(i + 1))

        variable_list = variable_list + slack_variable

        if (self.learn_threshold_clause):
            variable_list.append("eta_clause")

        if (self.learn_threshold_literal):
            # consider different threshold when learning mode is on
            for eachLevel in range(self.numClause):
                variable_list.append("eta_clit_"+str(eachLevel))

        for i in range(len(y)):
            for eachLevel in range(self.numClause):
                variable_list.append("ax_" + str(i + 1) +
                                     str("_") + str(eachLevel + 1))

        myProblem.variables.add(names=variable_list)

        # encode the objective function:

        if(self.verbose):
            print("- weight feature: ", self.weightFeature)
            print("- weight error:   ", self.dataFidelity)

        if(self.iterations == 1 or len(self.__assignList) == 0):  # is called in the first iteration
            for eachLevel in range(self.numClause):
                for i in range(no_features):
                    objective_coefficient.append(self.weightFeature)
                    myProblem.variables.set_lower_bounds(variable_count, 0)
                    myProblem.variables.set_upper_bounds(variable_count, 1)
                    myProblem.variables.set_types(
                        variable_count, myProblem.variables.type.continuous)
                    myProblem.objective.set_linear(
                        [(variable_count, objective_coefficient[variable_count])])
                    variable_count += 1
        else:
            for eachLevel in range(self.numClause):
                for i in range(no_features):
                    if (self.__assignList[eachLevel * no_features + i] > 0):
                        objective_coefficient.append(-self.weightFeature)
                    else:
                        objective_coefficient.append(self.weightFeature)

                    myProblem.variables.set_lower_bounds(variable_count, 0)
                    myProblem.variables.set_upper_bounds(variable_count, 1)
                    myProblem.variables.set_types(variable_count, myProblem.variables.type.continuous)
                    myProblem.objective.set_linear([(variable_count, objective_coefficient[variable_count])])
                    variable_count += 1

        # slack_variable = []
        for i in range(no_samples):
            objective_coefficient.append(self.dataFidelity)
            myProblem.variables.set_types(
                variable_count, myProblem.variables.type.continuous)
            myProblem.variables.set_lower_bounds(variable_count, 0)
            myProblem.variables.set_upper_bounds(variable_count, 1)
            myProblem.objective.set_linear(
                [(variable_count, objective_coefficient[variable_count])])
            variable_count += 1

        myProblem.objective.set_sense(myProblem.objective.sense.minimize)

        var_eta_clause = -1

        if (self.learn_threshold_clause):
            myProblem.variables.set_types(
                variable_count, myProblem.variables.type.integer)
            myProblem.variables.set_lower_bounds(variable_count, 0)
            myProblem.variables.set_upper_bounds(variable_count, self.numClause)
            var_eta_clause = variable_count
            variable_count += 1

        var_eta_literal = [-1 for eachLevel in range(self.numClause)]
        constraint_count = 0

        if (self.learn_threshold_literal):

            for eachLevel in range(self.numClause):
                myProblem.variables.set_types(
                    variable_count, myProblem.variables.type.integer)
                myProblem.variables.set_lower_bounds(variable_count, 0)
                myProblem.variables.set_upper_bounds(variable_count, no_features)
                var_eta_literal[eachLevel] = variable_count
                variable_count += 1

                constraint = []

                for j in range(no_features):
                    constraint.append(1)

                constraint.append(-1)

                myProblem.linear_constraints.add(
                    lin_expr=[
                        cplex.SparsePair(ind=[eachLevel * no_features + j for j in range(no_features)] + [var_eta_literal[eachLevel]],
                                         val=constraint)],
                    rhs=[0],
                    names=["c" + str(constraint_count)],
                    senses=["G"]
                )
                constraint_count += 1

        for i in range(len(y)):
            if (y[i] == 1):

                auxiliary_index = []

                for eachLevel in range(self.numClause):
                    constraint = [int(feature) for feature in A[i]]

                    myProblem.variables.set_types(
                        variable_count, myProblem.variables.type.integer)
                    myProblem.variables.set_lower_bounds(variable_count, 0)
                    myProblem.variables.set_upper_bounds(variable_count, 1)

                    constraint.append(no_features)

                    auxiliary_index.append(variable_count)

                    if (self.learn_threshold_literal):

                        constraint.append(-1)

                        myProblem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(
                                ind=[eachLevel * no_features + j for j in range(no_features)] + [variable_count,
                                                                                                 var_eta_literal[eachLevel]],
                                val=constraint)],
                            rhs=[0],
                            names=["c" + str(constraint_count)],
                            senses=["G"]
                        )

                        constraint_count += 1

                    else:

                        myProblem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(
                                ind=[eachLevel * no_features +
                                     j for j in range(no_features)] + [variable_count],
                                val=constraint)],
                            rhs=[self.threshold_literal],
                            names=["c" + str(constraint_count)],
                            senses=["G"]
                        )

                        constraint_count += 1

                    variable_count += 1

                if (self.learn_threshold_clause):

                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[i + self.numClause * no_features,
                                 var_eta_clause] + auxiliary_index,
                            # 1st slack variable = level * no_features
                            val=[self.numClause, -1] + [-1 for j in range(self.numClause)])],
                        rhs=[- self.numClause],
                        names=["c" + str(constraint_count)],
                        senses=["G"]
                    )

                    constraint_count += 1

                else:

                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            # 1st slack variable = level * no_features
                            ind=[i + self.numClause * no_features] + auxiliary_index,
                            val=[self.numClause] + [-1 for j in range(self.numClause)])],
                        rhs=[- self.numClause + self.threshold_clause],
                        names=["c" + str(constraint_count)],
                        senses=["G"]
                    )

                    constraint_count += 1

            else:

                auxiliary_index = []

                for eachLevel in range(self.numClause):
                    constraint = [int(feature) for feature in A[i]]
                    myProblem.variables.set_types(
                        variable_count, myProblem.variables.type.integer)
                    myProblem.variables.set_lower_bounds(variable_count, 0)
                    myProblem.variables.set_upper_bounds(variable_count, 1)

                    constraint.append(- no_features)

                    auxiliary_index.append(variable_count)

                    if (self.learn_threshold_literal):

                        constraint.append(-1)

                        myProblem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(
                                ind=[eachLevel * no_features + j for j in range(no_features)] + [variable_count,
                                                                                                 var_eta_literal[eachLevel]],
                                val=constraint)],
                            rhs=[-1],
                            names=["c" + str(constraint_count)],
                            senses=["L"]
                        )

                        constraint_count += 1
                    else:

                        myProblem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(
                                ind=[eachLevel * no_features +
                                     j for j in range(no_features)] + [variable_count],
                                val=constraint)],
                            rhs=[self.threshold_literal - 1],
                            names=["c" + str(constraint_count)],
                            senses=["L"]
                        )

                        constraint_count += 1

                    variable_count += 1

                if (self.learn_threshold_clause):

                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[i + self.numClause * no_features,
                                 var_eta_clause] + auxiliary_index,
                            # 1st slack variable = level * no_features
                            val=[self.numClause, 1] + [-1 for j in range(self.numClause)])],
                        rhs=[1],
                        names=["c" + str(constraint_count)],
                        senses=["G"]
                    )

                    constraint_count += 1

                else:

                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            # 1st slack variable = level * no_features
                            ind=[i + self.numClause * no_features] + auxiliary_index,
                            val=[self.numClause] + [-1 for j in range(self.numClause)])],
                        rhs=[- self.threshold_clause + 1],
                        names=["c" + str(constraint_count)],
                        senses=["G"]
                    )

                    constraint_count += 1

        # set parameters
        if(self.verbose):
            print("- timelimit: ",  self.timeOut/self.iterations)
        myProblem.parameters.clocktype.set(1)  # cpu time (exact time)
        myProblem.parameters.timelimit.set(self.timeOut/self.iterations)
        myProblem.parameters.workmem.set(self.memlimit)
        myProblem.set_log_stream(None)
        myProblem.set_error_stream(None)
        myProblem.set_warning_stream(None)
        myProblem.set_results_stream(None)
        # myProblem.parameters.mip.tolerances.mipgap.set(0.2)
        myProblem.parameters.mip.limits.treememory.set(self.memlimit)
        myProblem.parameters.workdir.set(self.workDir)
        myProblem.parameters.mip.strategy.file.set(2)
        myProblem.parameters.threads.set(1)

        # Solve the model and print the answer
        start_time = myProblem.get_time()
        start_det_time = myProblem.get_dettime()
        myProblem.solve()
        # solution.get_status() returns an integer code
        status = myProblem.solution.get_status()

        end_det_time = myProblem.get_dettime()

        end_time = myProblem.get_time()
        if (self.verbose):
            print("- Total solve time (sec.):", end_time - start_time)
            print("- Total solve dettime (sec.):", end_det_time - start_det_time)

            print("- Solution status = ", myProblem.solution.status[status])
            print("- Objective value = ", myProblem.solution.get_objective_value())
            print("- mip relative gap (should be zero):", myProblem.solution.MIP.get_mip_relative_gap())

        #  retrieve solution: do rounding

        self.__assignList = []
        self.__selectedFeatureIndex = []
        # if(self.verbose):
        #     print(" - selected feature index")
        for i in range(len(feature_variable)):
            if(myProblem.solution.get_values(feature_variable[i]) > 0):
                self.__assignList.append(1)
                self.__selectedFeatureIndex.append(i+1)
            else:
                self.__assignList.append(0)
                # self.__selectedFeatureIndex.append(i+1)
        # print(self.__selectedFeatureIndex)
        
        # self.__assignList.append(myProblem.solution.get_values(feature_variable[i]))

        for i in range(len(slack_variable)):
            self.__assignList.append(myProblem.solution.get_values(slack_variable[i]))

        # update parameters
        if (self.learn_threshold_clause and self.learn_threshold_literal):

            self.threshold_literal_learned = [int(myProblem.solution.get_values(var_eta_literal[eachLevel])) for eachLevel in range(self.numClause)]
            self.threshold_clause_learned = int(myProblem.solution.get_values(var_eta_clause))

        elif (self.learn_threshold_clause):
            self.threshold_literal_learned = [self.threshold_literal for eachLevel in range(self.numClause)]
            self.threshold_clause_learned = int(myProblem.solution.get_values(var_eta_clause))

        elif (self.learn_threshold_literal):
            self.threshold_literal_learned = [int(myProblem.solution.get_values(var_eta_literal[eachLevel])) for eachLevel in range(self.numClause)]
            self.threshold_clause_learned = self.threshold_clause

        if(self.verbose):
            print("- cplex returned the solution")
