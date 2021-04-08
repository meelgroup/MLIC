import subprocess
import math
import os
import numpy as np
from time import time

# from pyrulelearn
import pyrulelearn.utils


def _generateWcnfFile(imli, AMatrix, yVector, xSize, WCNFFile,
                        isTestPhase):

    # learn soft clauses associated with feature variables and noise variables
    topWeight, formula_builder = _learnSoftClauses(imli, isTestPhase, xSize,
                                                                yVector)
    
    # learn hard clauses,
    additionalVariable = 0
    y_len = len(yVector)

    precomputed_vars = [each_level * xSize for each_level in range(imli.numClause)]
    variable_head =  y_len + imli.numClause * xSize + 1

    for i in range(y_len):
        noise = imli.numClause * xSize + i + 1

        # implementation of tseitin encoding
        if (yVector[i] == 0):

            new_clause = str(topWeight) + " " + str(noise)
            
            # for each_level in range(imli.numClause):
                # new_clause += " " + str(additionalVariable + each_level + len(yVector) + imli.numClause * xSize + 1)
            formula_builder.append((" ").join(map(str, [topWeight, noise] + [additionalVariable + each_level + variable_head for each_level in range(imli.numClause)] + [0])))
            # new_clause += " 0\n"
            # cnfClauses += new_clause
            # numClauses += 1

        
            mask = AMatrix[i] == 1 
            dummy = np.arange(1, xSize+1)[mask]
            for j in dummy:
                for each_level in range(imli.numClause):
                    # numClauses += 1
                    # new_clause = str(topWeight) + " -" + str(additionalVariable + variable_head + each_level) + " -" + str(j + precomputed_vars[each_level])
                    formula_builder.append((" ").join(map(str, [topWeight, -1 * (additionalVariable + variable_head + each_level), -1 * (j + precomputed_vars[each_level]), 0])))
                    # cnfClauses += new_clause + " 0\n"

            additionalVariable += imli.numClause

        else:
        
            mask = AMatrix[i] == 1 
            dummy = np.arange(1, xSize+1)[mask]
            for each_level in range(imli.numClause):
                # cnfClauses += str(topWeight) + " " + str(noise) + " " + (" ").join(map(str, dummy + each_level * xSize)) + " 0\n"
                formula_builder.append((" ").join(map(str, [topWeight, noise] + list(dummy + each_level * xSize) + [0])))
                # numClauses += 1

    # cnfClauses = ("\n").join([(" ").join(map(str, each_clause)) for each_clause in formula_builder])


    # write in wcnf format
    start_demo_time = time()
    num_clauses = len(formula_builder)
    header = 'p wcnf ' + str(additionalVariable + variable_head - 1) + ' ' + str(num_clauses) + ' ' + str(topWeight) + "\n"
    
    with open(WCNFFile, 'w') as file:
        file.write(header)
        # write in chunck of 500 clauses
        # chunck_size = 500
        # for i in range(0, num_clauses, chunck_size):
        #     file.writelines(' '.join(str(var) for var in clause) + '\n' for clause in formula_builder[i:i + chunck_size])
        file.write("\n".join(formula_builder))

    imli._demo_time += time() - start_demo_time

    
    if(imli.verbose):
        print("- number of Boolean variables:", additionalVariable + xSize * imli.numClause + (len(yVector)))
        



def _learnSoftClauses(imli, isTestPhase, xSize, yVector):
    # cnfClauses = ''
    # numClauses = 0

    
    formula_builder = []

    if (isTestPhase):
        topWeight = imli.dataFidelity * len(yVector) + 1 + imli.weightFeature * xSize * imli.numClause
        # numClauses = 0
        for i in range(1, imli.numClause * xSize + 1):
            # numClauses += 1
            # cnfClauses += str(imli.weightFeature) + ' ' + str(-i) + ' 0\n'
            formula_builder.append((" ").join(map(str, [imli.weightFeature, -i, 0])))
        for i in range(imli.numClause * xSize + 1, imli.numClause * xSize + len(yVector) + 1):
            # numClauses += 1
            # cnfClauses += str(imli.dataFidelity) + ' ' + str(-i) + ' 0\n'
            formula_builder.append((" ").join(map(str, [imli.dataFidelity, -i, 0])))

        # for testing, the positive assigned feature variables are converted to hard clauses
        # so that  their assignment is kept consistent and only noise variables are considered soft,
        for each_assign in imli._assignList:
            # numClauses += 1
            # cnfClauses += str(topWeight) + ' ' + str(each_assign) + ' 0\n'
            formula_builder.append((" ").join(map(str, [topWeight, each_assign, 0])))
    else:
        # applicable for the 1st Batch
        isEmptyAssignList = True

        total_additional_weight = 0
        positiveLiteralWeight = imli.weightFeature
        for each_assign in imli._assignList:
            isEmptyAssignList = False
            # numClauses += 1
            if (each_assign > 0):

                # cnfClauses += str(positiveLiteralWeight) + ' ' + str(each_assign) + ' 0\n'
                formula_builder.append((" ").join(map(str, [positiveLiteralWeight, each_assign, 0])))
                total_additional_weight += positiveLiteralWeight

            else:
                # cnfClauses += str(imli.weightFeature) + ' ' + str(each_assign) + ' 0\n'
                formula_builder.append((" ").join(map(str, [imli.weightFeature, each_assign, 0])))
                total_additional_weight += imli.weightFeature

        # noise variables are to be kept consisitent (not necessary though)
        for i in range(imli.numClause * xSize + 1,
                        imli.numClause * xSize + len(yVector) + 1):
            # numClauses += 1
            # cnfClauses += str(imli.dataFidelity) + ' ' + str(-i) + ' 0\n'
            formula_builder.append((" ").join(map(str, [imli.dataFidelity, -i, 0])))

        # for the first step
        if (isEmptyAssignList):
            for i in range(1, imli.numClause * xSize + 1):
                # numClauses += 1
                # cnfClauses += str(imli.weightFeature) + ' ' + str(-i) + ' 0\n'
                formula_builder.append((" ").join(map(str, [imli.weightFeature, -i, 0])))
                total_additional_weight += imli.weightFeature

        topWeight = int(imli.dataFidelity * len(yVector) + 1 + total_additional_weight)

    # print(formula_builder)
    # cnfClauses = ("\n").join([(" ").join(map(str, each_clause)) for each_clause in formula_builder])
    # assert dummy == cnfClauses[:-1]
    # quit()
    if(imli.verbose):
        print("- number of soft clauses: ", len(formula_builder))

    return topWeight, formula_builder



def _pruneRules(imli, fields, xSize):
    # algorithm 1 in paper

    new_fileds = fields
    end_of_column_list = [imli.__columnInfo[i][-1] for i in range(len(imli.__columnInfo))]
    freq_end_of_column_list = [[[0, 0] for i in range(len(end_of_column_list))] for j in range(imli.numClause)]
    variable_contained_list = [[[] for i in range(len(end_of_column_list))] for j in range(imli.numClause)]

    for i in range(imli.numClause * xSize):
        if ((int(fields[i])) > 0):
            variable = (int(fields[i]) - 1) % xSize + 1
            clause_position = int((int(fields[i]) - 1) / xSize)
            for j in range(len(end_of_column_list)):
                if (variable <= end_of_column_list[j]):
                    variable_contained_list[clause_position][j].append(clause_position * xSize + variable)
                    freq_end_of_column_list[clause_position][j][0] += 1
                    freq_end_of_column_list[clause_position][j][1] = imli.__columnInfo[j][0]
                    break
    for l in range(imli.numClause):

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



def _cmd_exists(imli, cmd):
    return subprocess.call("type " + cmd, shell=True, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0

def _learnModel(imli, X, y, isTest):
    # X = pyrulelearn.utils._add_dummy_columns(X)

    # temp files to save maxsat query in wcnf format
    WCNFFile = imli.workDir + "/" + "model.wcnf"
    outputFileMaxsat = imli.workDir + "/" + "model_out.txt"
    num_features = len(X[0])
    num_samples = len(y)

    start_wcnf_generation = time()
    # generate maxsat query for dataset
    if (imli.ruleType == 'DNF'):
        #  negate yVector for DNF rules
        _generateWcnfFile(imli, X, [1 - int(y[each_y]) for each_y in
                                    range(num_samples)],
                                num_features, WCNFFile,
                                isTest)

    elif(imli.ruleType == "CNF"):
        _generateWcnfFile(imli, X, y, num_features,
                                WCNFFile,
                                isTest)
    else:
        print("\n\nError rule type")

    imli._wcnf_generation_time += time() - start_wcnf_generation

    
    solver_start_time = time()
    # call a maxsat solver
    if(imli.solver in ["open-wbo", "maxhs", 'satlike-cw', 'uwrmaxsat', 'tt-open-wbo-inc', 'open-wbo-inc']):  # solver has timeout and experimented with open-wbo only
        # if(_cmd_exists(imli, imli.solver)):
        if(True):
            # timeout_ = None

            # if(imli.iterations == -1):
            #     timeout_ = imli.timeOut
            # else:
            #     if(int(math.ceil(imli.timeOut/imli.iterations)) < 5):  # give at lest 1 second as cpu-lim
            #         timeout_ = 5
            #     else:
            #         timeout_ = int(math.ceil(imli.timeOut/imli.iterations))

            # assert timeout_ != None

            # left time is allocated for the solver
            timeout_ = max(int(imli.timeOut - time() + imli._fit_start_time), 5)

            
            if(imli.solver in ['open-wbo', 'maxhs', 'uwrmaxsat']):
                    cmd = imli.solver + '   ' + WCNFFile + ' -cpu-lim=' + str(timeout_) + ' > ' + outputFileMaxsat
            # incomplete solvers
            elif(imli.solver in ['satlike-cw', 'tt-open-wbo-inc', 'open-wbo-inc']):
                cmd = "timeout " + str(timeout_) + " " + imli.solver + '   ' + WCNFFile + ' > ' + outputFileMaxsat
            else:
                raise ValueError
            
        else:
            raise Exception("Solver not found")   
    else:
        raise Warning(imli.solver + " not configured as a MaxSAT solver in this implementation")
        cmd = imli.solver + '   ' + WCNFFile + ' > ' + outputFileMaxsat

    # print(cmd)

    os.system(cmd)
    imli._solver_time += time() - solver_start_time
    

    # delete temp files
    # cmd = "rm " + WCNFFile
    # os.system(cmd)

    


    solution = ''

    # # parse result of maxsat solving
    # f = open(outputFileMaxsat, 'r')
    # lines = f.readlines()
    # f.close()
    # # Always consider the last solution
    # for line in lines:
    #     if (line.strip().startswith('v')):
    #         solution = line.strip().strip('v ')

    # read line by line
    with open(outputFileMaxsat) as f:
        line = f.readline()
        while line:
            if (line.strip().startswith('v')):
                solution = line.strip().strip('v ')     
            line = f.readline()

            
    if(imli.solver in ['satlike-cw', 'tt-open-wbo-inc']):
        solution = (" ").join([str(idx+1) if(val == "1") else "-" + str(idx+1) for idx,val in enumerate(solution)])

    

    fields = [int(field) for field in solution.split()]
    TrueRules = []
    TrueErrors = []
    zeroOneSolution = []
        

    for field in fields:
        if (field > 0):
            zeroOneSolution.append(1.0)
        else:
            zeroOneSolution.append(0.0)
            
        if (field > 0):

            if (abs(field) <= imli.numClause * num_features):
                TrueRules.append(field)
            elif (imli.numClause * num_features < abs(field) <= imli.numClause * num_features + num_samples):
                TrueErrors.append(field)

    if (imli.verbose and isTest == False):
        print("\n\nBatch training complete")
        print("- number of literals in the rule: " + str(len(TrueRules)))
        print("- number of training errors:    " + str(len(TrueErrors)) + " out of " + str(num_samples))
    imli._xhat = []

    for i in range(imli.numClause):
        imli._xhat.append(np.array(
            zeroOneSolution[i * num_features:(i + 1) * num_features]))
    err = np.array(zeroOneSolution[num_features * imli.numClause: len(
        X[0]) * imli.numClause + num_samples])


    if(imli.ruleType == "DNF"):
        actual_feature_len = int(imli.numFeatures/2)
        imli._xhat = np.array([np.concatenate((each_xhat[actual_feature_len:], each_xhat[:actual_feature_len])) for each_xhat in imli._xhat])
    if(imli.ruleType == "CNF"):
        imli._xhat = np.array(imli._xhat)
    
    

    # delete temp files
    # cmd = "rm " + outputFileMaxsat
    # os.system(cmd)

    if (not isTest):
        imli._assignList = fields[:imli.numClause * num_features]
        imli._selectedFeatureIndex = TrueRules

        # print(imli._selectedFeatureIndex)

    

    return fields[imli.numClause * num_features:num_samples + imli.numClause * num_features]


