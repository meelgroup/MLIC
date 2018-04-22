from pandas import DataFrame, Series
import pandas as pd
import json
import numpy as np
from datetime import datetime as dt
from datetime import date
import time as tm
from glob import glob
import pylab as plt
import patsy  ### create R-like formulas


###########################################
def load_large_census_data(fname_census, fname_census_tst):
    ''' load UCI Census Income dataset
    keep it as training + test to match the avaialble classif. accuracy results
    '''

    feat_names = ['AMJIND','AMJOCC','ARACE','AREORGN','ASEX','AUNMEM','AUNTYPE','AWKSTAT','CAPGAIN','CAPLOSS','DIVVAL',
                  'FEDTAX','FILESTAT','GRINREG','GRINST','HHDFMX','HHDREL','MARSUPWT','MIGMTR1','MIGMTR3','MIGMTR4','MIGSAME',
                  'MIGSUN','NOEMP','PARENT','PEARNVAL','PEFNTVTY','PEMNTVTY','PENATVTY','PRCITSHP','PTOTVAL','SEOTR','TAXINC',
                  'VETQVA','VETYN','WKSWORK']

    census = pd.read_table(fname_census, sep = ',', header = False, 
                           names = feat_names)
    census_tst = pd.read_table(fname_census_tst, sep = ',', header= False, names = census.columns)

    print("")


    ### removing NaNs
    print("Removing rows with missing data")
    census = census.dropna()
    census_tst = census_tst.dropna()
    #census_tst.index = census_tst.index + len(census)  ### change the index to enable concatenation

    inds_tr = np.arange(len(census))
    inds_tst = np.arange(len(inds_tr), len(inds_tr) + len(census_tst))
    
    census = pd.concat([census, census_tst], ignore_index = True)

    ### find out what kind of features we're dealing with
    col_names = census.columns
    #col_names = [x.replace(' ', '_') for x in census.columns]

    print("WARNING: patsy is dropping the _reference_ label, need to disable this.. ")
    patsy_formula = '+'.join(col_names) + '-1'  ### -1 to remove intercept

    X = patsy.dmatrix(patsy_formula, census, return_type = 'dataframe')

    #del X['Intercept']  ### is there a way to do it in dmatrix directly
    #new_X_col_names = [x.replace(']','').replace('[T. ', ':') for x in X.columns]
    new_X_col_names = [x.replace(']','').replace('[T.', ':').replace('[ ', ':').replace(' ', '') for x in X.columns]
    X.columns = new_X_col_names

    label_col = 'WKSWORK:50000+.'
    ind_label = np.where(X.columns == label_col)[0][0]
    cols_reorder = X.columns.tolist()
    cols_reorder[ind_label] = cols_reorder[0]
    cols_reorder[0] = label_col
    X = X[cols_reorder]

    X_tr = X.iloc[inds_tr, :]
    X_tst = X.iloc[inds_tst, :]

    return(census, X_tr, X_tst)

###########################################
def load_census_data(fname_census, fname_census_tst):
    ''' load UCI Adult Census dataset
    keep it as training + test to match the avaialble classif. accuracy results
    '''

    census = pd.read_table(fname_census, sep = ',', header = False, 
                           names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
                                    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
                                    'hours_per_week', 'native_country', 'label'])
    census_tst = pd.read_table(fname_census_tst, sep = ',', header= False, names = census.columns)

    ### removing NaNs
    print("Removing rows with missing data")
    census = census.dropna()
    census_tst = census_tst.dropna()
    #census_tst.index = census_tst.index + len(census)  ### change the index to enable concatenation

    inds_tr = np.arange(len(census))
    inds_tst = np.arange(len(inds_tr), len(inds_tr) + len(census_tst))
    
    census = pd.concat([census, census_tst], ignore_index = True)

    ### find out what kind of features we're dealing with
    col_names = [x.replace(' ', '_') for x in census.columns]

    print("WARNING: patsy is dropping the _reference_ label, need to disable this.. ")
    patsy_formula = '+'.join(col_names) + '-1'  ### -1 to remove intercept

    X = patsy.dmatrix(patsy_formula, census, return_type = 'dataframe')

    #del X['Intercept']  ### is there a way to do it in dmatrix directly
    #new_X_col_names = [x.replace(']','').replace('[T. ', ':') for x in X.columns]
    new_X_col_names = [x.replace(']','').replace('[T.', ':').replace('[ ', ':').replace(' ', '') for x in X.columns]
    X.columns = new_X_col_names

    label_col = 'label:>50K'
    ind_label = np.where(X.columns == label_col)[0][0]
    cols_reorder = X.columns.tolist()
    cols_reorder[ind_label] = cols_reorder[0]
    cols_reorder[0] = label_col
    X = X[cols_reorder]

    X_tr = X.iloc[inds_tr, :]
    X_tst = X.iloc[inds_tst, :]

    return(census, X_tr, X_tst)

###########################################
def load_uci_census_data():
    print("Testing data-loading for BCS rule learner")
    
    fname_census = './Data/UCI_Adult/adult.data'
    fname_census_test = './Data/UCI_Adult/adult.test'
    
    ### load census data + get a representation suitable for BCS-rules
    census, X_tr, X_tst = load_census_data(fname_census, fname_census_test)
    
    fname_tr_out = '/home/dmaliout/Work/Python/dmmMisc/BooleanRules/Data/UCI_Adult/uci_adult_tr.csv'
    fname_tst_out = '/home/dmaliout/Work/Python/dmmMisc/BooleanRules/Data/UCI_Adult/uci_adult_tst.csv'

    print(("Writing out files: tr %s and tst %s" % (fname_tr_out, fname_tst_out)))
    X_tr.to_csv(fname_tr_out, header = True, mode = 'w', index = False)
    X_tst.to_csv(fname_tst_out, header = True, mode = 'w', index = False)
    
###########################################
def load_uci_large_census_data():
    print("Testing data-loading for BCS rule learner")
    
    fname_census = '/home/dmaliout/Work/Python/dmmMisc/BooleanRules/Data/CensusIncome/census-income.data'
    fname_census_test = '/home/dmaliout/Work/Python/dmmMisc/BooleanRules/Data/CensusIncome/census-income.test'

    ### load census data + get a representation suitable for BCS-rules
    census, X_tr, X_tst = load_large_census_data(fname_census, fname_census_test)

    fname_tr_out = '/home/dmaliout/Work/Python/dmmMisc/BooleanRules/Data/CensusIncome/uci_census_tr.csv'
    fname_tst_out = '/home/dmaliout/Work/Python/dmmMisc/BooleanRules/Data/CensusIncome/uci_census_tst.csv'
    
    if 0:  ### see feature names:
        print(("Num columns: %d orig, %d expanded" % (len(census.columns), X_tr.shape[1])))
        for col in census.columns:
            print(("----", col))
            print((census[col].value_counts()))

    print(("Writing out files: tr %s and tst %s" % (fname_tr_out, fname_tst_out)))
    X_tr.to_csv(fname_tr_out, header = True, mode = 'w', index = False)
    X_tst.to_csv(fname_tst_out, header = True, mode = 'w', index = False)

###########################################
if __name__ == "__main__":
    #main()
    #load_uci_census_data()
    load_uci_large_census_data()
