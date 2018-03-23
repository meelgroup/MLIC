# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 08:27:42 2016

"""

import numpy as np
import pandas as pd

###########################################
#%% Define function
def discretization(train_file, test_file, colSep, rowHeader, colNames, col_y, fracPresent=0.9, colCateg=[], numThresh=9, valEq_y=None):
    ''' Load CSV file and process data for rule learner
    
    Inputs:    
    filePath = full path to CSV file
    colSep = column separator
    rowHeader = index of row containing column names or None
    colNames = column names (if none in file or to override)
    col_y = name of target column
    fracPresent = fraction of non-missing values needed to use a column (default 0.9)
    colCateg = list of names of categorical columns
    numThresh = number of quantile thresholds used to binarize ordinal variables (default 9)
    valEq_y = value to test for equality to binarize non-binary target column
    Outputs:
    A = binary feature DataFrame
    y = target column'''

    # Quantile probabilities    
    quantProb = np.linspace(1./(numThresh + 1.), numThresh/(numThresh + 1.), numThresh)
    # List of categorical columns
    if type(colCateg) is pd.Series:
        colCateg = colCateg.tolist()
    elif type(colCateg) is not list:
        colCateg = [colCateg]

    #%% Read CSV file
    training_data = pd.read_csv(train_file, sep=colSep, names=colNames, header=rowHeader, error_bad_lines=False)
    test_data =  pd.read_csv(test_file, sep=colSep, names=colNames, header=rowHeader, error_bad_lines=False)
    ### dmm: if no col_y specified -- use the last
    if col_y == None:
        if colNames is None:
            colNames = training_data.columns
        col_y = colNames[-1]
        colNames = colNames[:-1]
    #%% Remove missing values
    # Remove columns with too many missing values
    training_data.dropna(axis=1, thresh=fracPresent * len(training_data), inplace=True) 
    #test_data.dropna(axis=1, thresh=fracPresent * len(training_data), inplace=True)
    # Remove rows with any missing values
    training_data.dropna(axis=0, how='any', inplace=True)
    test_data.dropna(axis=0, how='any', inplace=True)
    #%% Separate target column
    y_train = training_data.pop(col_y).copy()
    y_test = test_data.pop(col_y).copy()
    # Binarize if value for equality test provided
    if valEq_y:
        y_train = (y_train == valEq_y).astype(int)
        y_test = (y_test == valEq_y).astype(int)
    # Ensure y is binary and contains no missing values
    assert y_train.nunique() == 2, "Target 'y' must be binary"
    assert y_test.nunique() == 2, "Target 'y' must be binary"
    assert y_train.count() == len(y_train), "Target 'y' must not contain missing values"
    assert y_test.count() == len(y_test), "Target 'y' must not contain missing values"
    # Rename values to 0, 1
    y_train.replace(np.sort(y_train.unique()), [0, 1], inplace=True)
    y_test.replace(np.sort(y_test.unique()), [0, 1], inplace=True)
    #%% Binarize features
    
    # Initialize dataframe and thresholds
    A_train = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))
    A_test = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))
    thresh = {}
    
    # Iterate over columns
    for c in training_data:
        # number of unique values    
        valUniq = training_data[c].nunique()
        
        # Constant column --- discard
        if valUniq < 2:
            continue
        
        # Binary column
        elif valUniq == 2:
            # Rename values to 0, 1
            A_train[(c, '', '')] = training_data[c].replace(np.sort(training_data[c].unique()), [0, 1])
            A_train[(c, 'not', '')] = training_data[c].replace(np.sort(training_data[c].unique()), [1, 0])
            A_test[(c, '', '')] = test_data[c].replace(np.sort(training_data[c].unique()), [0, 1])
            A_test[(c, 'not', '')] = test_data[c].replace(np.sort(training_data[c].unique()), [1, 0])
        # Categorical column
        elif (c in colCateg) or (training_data[c].dtype == 'object'):
            # Dummy-code values
            Anew_train = pd.get_dummies(training_data[c]).astype(int)
            Anew_train.columns = Anew_train.columns.astype(str)
            Anew_test = pd.get_dummies(training_data[c]).astype(int)
            Anew_test.columns = Anew_test.columns.astype(str)
            # Append negations
            Anew_train = pd.concat([Anew_train, 1-Anew_train], axis=1, keys=[(c,'=='), (c,'!=')])
            Anew_test = pd.concat([Anew_test, 1-Anew_test], axis=1, keys=[(c,'=='), (c,'!=')])
            # Concatenate
            A_train = pd.concat([A_train, Anew_train], axis=1)
            A_test = pd.concat([A_test, Anew_test], axis=1)
        # Ordinal column
        elif np.issubdtype(training_data[c].dtype, int) | np.issubdtype(training_data[c].dtype, float):
            # Few unique values
            if valUniq <= numThresh + 1:
                # Thresholds are sorted unique values excluding maximum
                thresh[c] = np.sort(training_data[c].unique())[:-1]
            # Many unique values
            else:
                # Thresholds are quantiles excluding repetitions
                thresh[c] = training_data[c].quantile(q=quantProb).unique()
            # Threshold values to produce binary arrays
            Anew_train = (training_data[c].values[:, np.newaxis] <= thresh[c]).astype(int)
            Anew_test = (test_data[c].values[:, np.newaxis] <= thresh[c]).astype(int)
            Anew_train = np.concatenate((Anew_train, 1 - Anew_train), axis=1)
            Anew_test = np.concatenate((Anew_test, 1 - Anew_test), axis=1)
            # Convert to dataframe with column labels
            Anew_train = pd.DataFrame(Anew_train, columns=pd.MultiIndex.from_product([[c], ['<=', '>'], thresh[c].astype(str)]))
            Anew_test = pd.DataFrame(Anew_test, columns=pd.MultiIndex.from_product([[c], ['<=', '>'], thresh[c].astype(str)]))
            # Concatenate
            A_train = pd.concat([A_train, Anew_train], axis=1)
            A_test = pd.concat([A_test, Anew_test], axis=1)
        else:
            print(("Skipping column '" + c + "': data type cannot be handled"))
            continue
    return A_train,A_test, y_train, y_test
#%% Call function if run as script
