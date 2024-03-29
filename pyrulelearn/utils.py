import Orange
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import random
from feature_engine import discretisers as dsc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



def discretize_orange(csv_file, verbose=False):
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
    
    X=[]
    # # make another level of binarization
    # for sample in binarized_data.X:
    #     X.append([int(feature) for feature in sample]+ [int(1-feature) for feature in sample])
    X = binarized_data.X


    columns = []
    for i in range(len(binarized_data.domain)-1):
        column = binarized_data.domain[i].name
        if("<" in column):
            column = column.replace("=<", ' < ')
        elif("≥" in column):
            column = column.replace("=≥", ' >= ')
        elif("=" in column):
            if("-" in column):
                column = column.replace("=", " = (")
                column = column+")"
            else:
                column = column.replace("=", " = ")
                column = column
        columns.append(column)

    
    # make negated columns
    # num_features=len(columns)
    # for index in range(num_features):
    #     columns.append("not "+columns[index])


    if(verbose):
        print("Applying entropy based discretization using Orange library")
        print("- file name: ", csv_file)
        print("- the number of discretized features:", len(columns))

    return np.array(X), np.array([int(value) for value in binarized_data.Y]),  columns



def get_scaled_df(X):
    # scale the feature values 
    sc = StandardScaler()
    X = sc.fit_transform(X)
    return X



def process(csv_file, verbose=False):
    df = pd.read_csv(csv_file)
    prev_columns = list(df.columns)

    target = None
    real_valued_columns = []
    categorical_columns = []
    
    
    for idx, column in enumerate(prev_columns):

        if(column.startswith("i#")):
            del df[column]
            continue
        


        if(column.startswith("C#")):
            column = column[2:]
            real_valued_columns.append(column)
        elif(column.startswith("D#")):
            column = column[2:]
            categorical_columns.append(column)
        elif(column.startswith("cD#")):
            column = column[3:]
            target = column
        else:
            raise ValueError(str(column) + " is not recognized")
        
        df.rename({prev_columns[idx] : column}, axis=1, inplace=True)

    assert len([target] + categorical_columns + real_valued_columns) == len(df.columns)

    # scale 
    scaler = MinMaxScaler()
    if(len(real_valued_columns) > 0):
        df[real_valued_columns] = scaler.fit_transform(df[real_valued_columns])


    # drop rows with null values
    df = df.dropna()

    X_orig = df.drop([target], axis=1)
    y_orig = df[target]

    X = get_one_hot_encoded_df(X_orig, columns_to_one_hot=categorical_columns)

    
    X_discretized, binner_dict = get_discretized_df(X_orig, columns_to_discretize=real_valued_columns, verbose=False)
    # print(binner_dict)
    X_discretized = get_one_hot_encoded_df(X_discretized, columns_to_one_hot=list(X_discretized.columns), good_name=binner_dict)


    return X.values, y_orig.values, list(X.columns), X_discretized.values, y_orig.values, list(X_discretized.columns)

    



def get_discretized_df(data, columns_to_discretize = None, verbose=False):
    """ 
    returns train_test_splitted and discretized df
    """

    binner_dict_ = {}
    
    if(columns_to_discretize is None):
        columns_to_discretize = list(data.columns)

    if(verbose):
        print("Applying discretization\nAttribute bins")
    for variable in columns_to_discretize:
        bins = min(10, len(data[variable].unique()))
        if(verbose):
            print(variable, bins)
            
        # set up the discretisation transformer
        disc  = dsc.EqualWidthDiscretiser(bins=bins, variables = [variable])
        
        # fit the transformer
        disc.fit(data)

        if(verbose):
            print(disc.binner_dict_)

        for key in disc.binner_dict_:
            assert key not in binner_dict_
            binner_dict_[key] = disc.binner_dict_[key]

        # transform the data
        data = disc.transform(data)
        if(verbose):
            print(data[variable].unique())
        
        
    return data, binner_dict_


def get_one_hot_encoded_df(df, columns_to_one_hot, good_name = {}, verbose = False):
    """  
    Apply one-hot encoding on categircal df and return the df
    """
    if(verbose):
        print("\n\nApply one-hot encoding on categircal attributes")
    for column in columns_to_one_hot:
        if(column not in df.columns):
            if(verbose):
                print(column, " is not considered in classification")
            continue 

        # Apply when there are more than two categories or the binary categories are string objects.
        unique_categories = df[column].unique()
        if(len(unique_categories) > 2):
            one_hot = pd.get_dummies(df[column])
            if(verbose):
                print(column, " has more than two unique categories", list(one_hot.columns))

            if(len(one_hot.columns)>1):
                if(column not in good_name):
                    one_hot.columns = [column + " = " + str(c) for c in one_hot.columns]
                else:
                    # print(column, one_hot.columns)
                    one_hot.columns = [str(good_name[column][idx]) + " <= " + column + " < "  +  str(good_name[column][idx + 1]) for idx in one_hot.columns]
            else:
                one_hot.columns = [column for c in one_hot.columns]
            df = df.drop(column,axis = 1)
            df = df.join(one_hot)
        else:
            # print(column, unique_categories)
            if(0 in unique_categories and 1 in unique_categories):
                if(verbose):
                    print(column, " has categories 1 and 0")

                continue
            if(len(unique_categories) == 2):
                df[column] = df[column].map({unique_categories[0]: 0, unique_categories[1]: 1})
            else:
                assert len(unique_categories) == 1
                df[column] = df[column].map({unique_categories[0]: 0})
            if(verbose):
                print("Applying following mapping on attribute", column, "=>", unique_categories[0], ":",  0, "|", unique_categories[1], ":", 1)
    if(verbose):
        print("\n")
    return df



def _discretize(imli, file, categorical_column_index=[], column_seperator=",", frac_present=0.9, num_thresholds=4, verbose=False):

    # Quantile probabilities
    quantProb = np.linspace(1. / (num_thresholds + 1.), num_thresholds / (num_thresholds + 1.), num_thresholds)
    # List of categorical columns
    if type(categorical_column_index) is pd.Series:
        categorical_column_index = categorical_column_index.tolist()
    elif type(categorical_column_index) is not list:
        categorical_column_index = [categorical_column_index]
    data = pd.read_csv(file, sep=column_seperator, header=0, error_bad_lines=False)

    columns = data.columns
    if (verbose):
        print("\n\nApplying quantile based discretization")
        print("- file name: ", file)
        print("- categorical features index: ", categorical_column_index)
        print("- number of bins: ", num_thresholds)
        # print("- features: ", columns)
        print("- number of features:", len(columns))


    columnY = columns[-1]

    data.dropna(axis=1, thresh=frac_present * len(data), inplace=True)
    data.dropna(axis=0, how='any', inplace=True)

    y = data.pop(columnY).copy()

    # Initialize dataframe and thresholds
    X = pd.DataFrame(columns=pd.MultiIndex.from_arrays([[], [], []], names=['feature', 'operation', 'value']))
    thresh = {}
    column_counter = 1
    imli.__columnInfo = []
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
            imli.__columnInfo.append(temp)
            column_counter += 2

        # Categorical column
        elif (count in categorical_column_index) or (data[c].dtype == 'object'):
            # if (imli.verbose):
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
            imli.__columnInfo.append(temp)
            column_counter += 2

        # Ordinal column
        elif np.issubdtype(data[c].dtype, int) | np.issubdtype(data[c].dtype, float):
            # Few unique values
            # if (imli.verbose):
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
            imli.__columnInfo.append(temp)
            temp = [4]
            temp = temp + [column_counter + nc for nc in range(addedColumn)]
            column_counter += addedColumn
            imli.__columnInfo.append(temp)
        else:
            # print(("Skipping column '" + c + "': data type cannot be handled"))
            continue
        count += 1

    if(verbose):
        print("\n\nAfter applying discretization")
        print("- number of discretized features: ", len(X.columns))
    return X.values, y.values.ravel(), X.columns


def _transform_binary_matrix(X):
    X = np.array(X)
    assert np.array_equal(X, X.astype(bool)), "Feature array is not binary. Try imli.discretize or imli.discretize_orange"
    X_complement = 1 - X
    return np.hstack((X,X_complement)).astype(bool)



def _generateSamples(imli, XTrain, yTrain):

    num_pos_samples = sum(y > 0 for y in yTrain)
    relative_batch_size = float(imli.batchsize/len(yTrain))

    list_of_random_index = random.sample(
        [i for i in range(num_pos_samples)], int(num_pos_samples * relative_batch_size)) + random.sample(
        [i for i in range(num_pos_samples, imli.trainingSize)], int((imli.trainingSize - num_pos_samples) * relative_batch_size))
    
    # print(int(imli.trainingSize * imli.batchsize))
    XTrain_sampled = [XTrain[i] for i in list_of_random_index]
    yTrain_sampled = [yTrain[i] for i in list_of_random_index]

    assert len(list_of_random_index) == len(set(list_of_random_index)), "sampling is not uniform"

    return XTrain_sampled, yTrain_sampled

def _numpy_partition(imli, X, y):
    y = y.copy()
    # based on numpy split
    result = np.hstack((X,y.reshape(-1,1)))
    # np.random.seed(22)
    # np.random.shuffle(result)
    result = np.array_split(result, imli.iterations)
    return [np.delete(batch,-1, axis=1) for batch in result], [batch[:,-1] for batch in result]
    

def _getBatchWithEqualProbability(imli, X, y):
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
    Batch_count = imli.iterations
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

            # # to numpy
            # final_Batch_X_train[i] = np.array(final_Batch_X_train[i])
            # final_Batch_y_train[i] = np.array(final_Batch_y_train[i])


    return final_Batch_X_train[:Batch_count], final_Batch_y_train[:Batch_count]


