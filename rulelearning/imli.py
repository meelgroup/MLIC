import numpy as np
import pandas as pd
import warnings
import math
import os
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)


class imli():
    def __init__(self, numPartition=-1, numClause=1, dataFidelity=10, weightFeature=1, solver="open-wbo", ruleType="CNF",
                 workDir="."):

        '''

        :param numPartition: no of partitions of training dataset
        :param numClause: no of clause in the formula
        :param dataFidelity: weight corresponding to accuracy
        :param weightFeature: weight corresponding to selected features
        :param solver: specify the (name of the) bin of the solver; bin must be in the path
        :param ruleType: type of rule {CNF,DNF}
        :param workDir: working directory
        :param verbose: True for debug
        '''
        self.numPartition = numPartition
        self.numClause = numClause
        self.dataFidelity = dataFidelity
        self.weightFeature = weightFeature
        self.solver = solver
        self.ruleType = ruleType
        self.workDir = workDir
        self.verbose = False # not necessary
        self.trainingError = 0
        self.selectedFeatureIndex = []
        self.columns = []

    def __repr__(self):
        return "<imli numPartition:%s numClause:%s dataFidelity:%s weightFeature:%s " \
               "solver:%s ruleType:%s workDir:%s>" % (self.numPartition,
                                                      self.numClause, self.dataFidelity,
                                                      self.weightFeature, self.solver,
                                                      self.ruleType, self.workDir)

    def getColumns(self):
        return [str(a) + str(" ") + str(b) + str(" ") + str(c) for (a, b, c) in self.columns]

    def getTrainingError(self):
        return self.trainingError

    def getSelectedColumnIndex(self):
        return_list=[[] for i in range(self.numClause)]
        ySize=len(self.columns)
        for elem  in self.selectedFeatureIndex:
            new_index=int(elem)-1
            return_list[int(new_index/ySize)].append(new_index%ySize)
        return return_list

    def getNumOfPartition(self):
        return self.numPartition

    def getNumOfClause(self):
        return self.numClause

    def getWeightFeature(self):
        return self.weightFeature

    def getRuleType(self):
        return self.ruleType

    def getWorkDir(self):
        return self.workDir

    def getWeightDataFidelity(self):
        return self.dataFidelity

    def getSolver(self):
        return self.solver

    def discretize(self, file, categoricalColumnIndex=[], columnSeperator=",", fracPresent=0.9, numThreshold=9):


        # Quantile probabilities
        quantProb = np.linspace(1. / (numThreshold + 1.), numThreshold / (numThreshold + 1.), numThreshold)
        # List of categorical columns
        if type(categoricalColumnIndex) is pd.Series:
            categoricalColumnIndex = categoricalColumnIndex.tolist()
        elif type(categoricalColumnIndex) is not list:
            categoricalColumnIndex = [categoricalColumnIndex]
        data = pd.read_csv(file, sep=columnSeperator, header=0, error_bad_lines=False)

        columns = data.columns
        # if (self.verbose):
        #     print(data)
        #     print(columns)
        #     print(categoricalColumnIndex)
        if (self.verbose):
            print("before discrertization: ")
            print("features: ", columns)
            print("index of categorical features: ",categoricalColumnIndex)
        #
        # if (self.verbose):
        #     print(data.columns)

        columnY = columns[-1]

        data.dropna(axis=1, thresh=fracPresent * len(data), inplace=True)
        data.dropna(axis=0, how='any', inplace=True)

        y = data.pop(columnY).copy()

        # Initialize dataframe and thresholds
        X = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))
        thresh = {}
        column_counter = 1
        self.columnInfo = []
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
                self.columnInfo.append(temp)
                column_counter += 2

            # Categorical column
            elif (count in categoricalColumnIndex) or (data[c].dtype == 'object'):
                # if (self.verbose):
                #     print(c)
                #     print(c in categoricalColumnIndex)
                #     print(data[c].dtype)
                # Dummy-code values
                Anew = pd.get_dummies(data[c]).astype(int)
                Anew.columns = Anew.columns.astype(str)
                # Append negations
                Anew = pd.concat([Anew, 1 - Anew], axis=1, keys=[(c, '=='), (c, '!=')])
                # Concatenate
                X = pd.concat([X, Anew], axis=1)

                temp = [2, column_counter, column_counter + 1]
                self.columnInfo.append(temp)
                column_counter += 2

            # Ordinal column
            elif np.issubdtype(data[c].dtype, int) | np.issubdtype(data[c].dtype, float):
                # Few unique values
                # if (self.verbose):
                #     print(data[c].dtype)
                if valUniq <= numThreshold + 1:
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
                self.columnInfo.append(temp)
                temp = [4]
                temp = temp + [column_counter + nc for nc in range(addedColumn)]
                column_counter += addedColumn
                self.columnInfo.append(temp)
            else:
                # print(("Skipping column '" + c + "': data type cannot be handled"))
                continue
            count += 1
        self.columns = X.columns
        return X.as_matrix(), y.values.ravel()

    def fit(self, XTrain, yTrain):

        if(self.numPartition==-1):
            self.numPartition=2**math.floor(math.log2(len(XTrain)/32))


        self.trainingError = 0
        self.trainingSize = len(XTrain)

        XTrains, yTrains = self.partitionWithEqualProbability(XTrain, yTrain)

        self.assignList = []
        for each_partition in range(self.numPartition):
            self.learnModel(XTrains[each_partition], yTrains[each_partition], isTest=False)

    def predict(self, XTest, yTest):
        predictions = self.learnModel(XTest, yTest, isTest=True)
        yhat = []
        for i in range(len(predictions)):
            if (int(predictions[i]) > 0):
                yhat.append(1 - yTest[i])
            else:
                yhat.append(yTest[i])
        return yhat

    def learnModel(self, X, y, isTest):
        # temp files to save maxsat query in wcnf format
        WCNFFile = self.workDir + "/" + "model.wcnf"
        outputFileMaxsat = self.workDir + "/" + "model_out.txt"

        # generate maxsat query for dataset
        if (self.ruleType == 'and'):
            #  negate yVector for DNF rules
            self.generateWCNFFile(X, [1 - int(y[each_y]) for each_y in
                                      range(len(y))],
                                  len(X[0]), WCNFFile,
                                  isTest)

        else:
            self.generateWCNFFile(X, y, len(X[0]),
                                  WCNFFile,
                                  isTest)

        # call a maxsat solver

        cmd = self.solver + '   ' + WCNFFile + ' > ' + outputFileMaxsat
        os.system(cmd)

        # delete temp files
        cmd = "rm " + WCNFFile
        os.system(cmd)

        # parse result of maxsat solving
        f = open(outputFileMaxsat, 'r')
        lines = f.readlines()
        f.close()
        optimumFound = False
        bestSolutionFound = False
        solution = ''
        for line in lines:
            if (self.solver == "maxroster" and line.strip().startswith('v')):
                solution = line.strip().strip('v ')
                break
            elif (self.solver == "LMHS" and line.strip().startswith('v')):
                solution = line.strip().strip('v ')
                break
            elif (self.solver == "qmaxsat" and line.strip().startswith('v')):
                solution = solution + " " + line.strip().strip('v ')


            elif (line.strip().startswith('v') and optimumFound):
                solution = line.strip().strip('v ')
                break
            elif (line.strip().startswith('c ') and bestSolutionFound):
                solution = line.strip().strip('c ')
                break
            elif (line.strip().startswith('s OPTIMUM FOUND')):
                optimumFound = True
                # if (self.verbose):
                #     print("Optimum solution found")
            elif (line.strip().startswith('c Best Model Found:')):
                bestSolutionFound = True
                # if (self.verbose):
                #     print("Best solution found")

        fields = solution.split()
        TrueRules = []
        TrueErrors = []
        zeroOneSolution = []

        fields = self.pruneRules(fields, len(X[0]))
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

        # if (self.verbose):
        #     print("The number of True Rule are: " + str(len(TrueRules)))
        #     print("The number of errors are:    " + str(len(TrueErrors)) + " out of " + str(len(y)))
        self.xhat = []

        for i in range(self.numClause):
            self.xhat.append(np.array(
                zeroOneSolution[i * len(X[0]):(i + 1) * len(X[0])]))
        err = np.array(zeroOneSolution[len(X[0]) * self.numClause: len(
            X[0]) * self.numClause + len(y)])

        # delete temp files
        cmd = "rm " + outputFileMaxsat
        os.system(cmd)

        if (not isTest):
            self.assignList = fields[:self.numClause * len(X[0])]
            self.trainingError += len(TrueErrors)
            self.selectedFeatureIndex = TrueRules

        return fields[self.numClause * len(X[0]):len(y) + self.numClause * len(X[0])]

    def pruneRules(self, fields, xSize):
        # algorithm 1 in paper

        new_fileds = fields
        end_of_column_list = [self.columnInfo[i][-1] for i in range(len(self.columnInfo))]
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
                        freq_end_of_column_list[clause_position][j][1] = self.columnInfo[j][0]
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

    def partitionWithEqualProbability(self, X, y):
        '''
            Steps:
                1. seperate data based on class value
                2. partition each seperate data into partition_count batches using test_train_split method with 50% part in each
                3. merge one seperate batche from each class and save
            :param X:
            :param y:
            :param partition_count:
            :param location:
            :param file_name_header:
            :param column_set_list: uses for incremental approach
            :return:
            '''
        partition_count = self.numPartition
        # y = y.values.ravel()
        max_y = int(y.max())
        min_y = int(y.min())

        X_list = [[] for i in range(max_y - min_y + 1)]
        y_list = [[] for i in range(max_y - min_y + 1)]
        level = int(math.log(partition_count, 2.0))
        for i in range(len(y)):
            inserting_index = int(y[i])
            y_list[inserting_index - min_y].append(y[i])
            X_list[inserting_index - min_y].append(X[i])

        final_partition_X_train = [[] for i in range(partition_count)]
        final_partition_y_train = [[] for i in range(partition_count)]
        for each_class in range(len(X_list)):
            partition_list_X_train = [X_list[each_class]]
            partition_list_y_train = [y_list[each_class]]

            for i in range(level):
                for j in range(int(math.pow(2, i))):
                    A_train_1, A_train_2, y_train_1, y_train_2 = train_test_split(
                        partition_list_X_train[int(math.pow(2, i)) + j - 1],
                        partition_list_y_train[int(math.pow(2, i)) + j - 1],
                        test_size=0.5,
                        random_state=None)  # random state for keeping consistency between lp and maxsat approach
                    partition_list_X_train.append(A_train_1)
                    partition_list_X_train.append(A_train_2)
                    partition_list_y_train.append(y_train_1)
                    partition_list_y_train.append(y_train_2)

            partition_list_y_train = partition_list_y_train[partition_count - 1:]
            partition_list_X_train = partition_list_X_train[partition_count - 1:]

            for i in range(partition_count):
                final_partition_y_train[i] = final_partition_y_train[i] + partition_list_y_train[i]
                final_partition_X_train[i] = final_partition_X_train[i] + partition_list_X_train[i]

        return final_partition_X_train[:partition_count], final_partition_y_train[:partition_count]

    def learnSoftClauses(self, isTestPhase, xSize, yVector):
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
            for each_assign in self.assignList:
                numClauses += 1
                cnfClauses += str(topWeight) + ' ' + each_assign + ' 0\n'
        else:
            # applicable for the 1st partition
            isEmptyAssignList = True

            total_additional_weight = 0;
            positiveLiteralWeight = self.weightFeature
            for each_assign in self.assignList:
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

        return topWeight, numClauses, cnfClauses

    def generateWCNFFile(self, AMatrix, yVector, xSize, WCNFFile,
                         isTestPhase):
        # learn soft clauses associated with feature variables and noise variables
        topWeight, numClauses, cnfClauses = self.learnSoftClauses(isTestPhase, xSize,
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

    def getRule(self):
        generatedRule = '( '
        for i in range(self.numClause):
            xHatElem = self.xhat[i]
            inds_nnz = np.where(abs(xHatElem) > 1e-4)[0]

            str_clauses = [' '.join(self.columns[ind]) for ind in inds_nnz]
            if (self.ruleType == "CNF"):
                rule_sep = ' %s ' % "or"
            else:
                rule_sep = ' %s ' % "and"
            rule_str = rule_sep.join(str_clauses)
            if (self.ruleType == 'DNF'):
                rule_str = rule_str.replace('<=', '??').replace('>', '<=').replace('??', '>')
                rule_str = rule_str.replace('==', '??').replace('!=', '==').replace('??', '!=')
                rule_str = rule_str.replace('is', '??').replace('is not', 'is').replace('??', 'is not')

            generatedRule += rule_str
            if (i < self.numClause - 1):
                if (self.ruleType == "DNF"):
                    generatedRule += ' ) or \n( '
                if (self.ruleType == 'CNF'):
                    generatedRule += ' ) and \n( '
        generatedRule += ')'

        return generatedRule
