import numpy as np, scipy as sp
import pandas 
import argparse, os
import sys
import scipy.io as sio, pickle
try:
    from pycpx import CPlexModel
except ImportError:
    pass
from load_process_data_BCS import load_process_data_BCS
sys.path.append('../LMHS/LMHS/bin/')
sys.path.append('../RuleLearning/')
try:
    from MultiLevelLearnRules import LearnRules
except ImportError:
    pass

###########################################
def load_UCI_data(training_data, test_data, rule_type, tool_type,runIndex):
    ''' Iris example ''' 

    colSep = ','
    rowHeader = 0
    colNames = None # ['X1', 'X2', 'sepal length', 'sepal width', 'petal length', 'petal width', 'iris species']
    col_y = None #colNames[-1]

    A_df_train, A_df_test, y_train, y_test = load_process_data_BCS(training_data, test_data, colSep, rowHeader, colNames, col_y, valEq_y=None)
    A_train = A_df_train.values
    A_test = A_df_test.values
    col_to_feat = get_col_to_features_map(A_df_train)
    fname_traindump =  "/tmp/"+training_data.split('/')[-1].replace('.csv', '_'+str(runIndex)+'_tempdata.pk')
    fname_testdump = "/tmp/"+test_data.split('/')[-1].replace('.csv', '_'+str(runIndex)+'_tempdata.pk')
    if (tool_type == 'sat'):
        if (rule_type == 'and'):
            dump_LPrule_data(fname_traindump, A_train, 1-y_train, col_to_feat)
            dump_LPrule_data(fname_testdump, A_test, 1-y_test, col_to_feat)
        else:
            dump_LPrule_data(fname_traindump, A_train, y_train, col_to_feat)
            dump_LPrule_data(fname_testdump,A_test, y_test, col_to_feat)
    return A_df_train, A_train, A_test, y_train, y_test, col_to_feat, fname_traindump, fname_testdump
###########################################
def run_UCI_example(training_data, test_data, lambda_reg, rule_type, tool_type, mValue, timeout, level, groupNoiseFlag, runIndex):
    #A_df, y, col_to_feat = load_iris_data()
    A_df_train, A_train, A_test, y_train, y_test, col_to_feat, fname_traindump, fname_testdump = load_UCI_data(training_data,test_data, rule_type, tool_type,runIndex)
    assignList = []
    alpha = lambda_reg
    beta = 1
    if (lambda_reg < 1):
        alpha = int(1)
        beta = int(1/lambda_reg)
    print("TRAINING PHASE")
    if rule_type == 'and':
        x_hat, xi_err, assignList = LearnRules(fname_traindump, mValue, alpha, beta, 1, timeout, rule_type, level,
                                    groupNoiseFlag, runIndex, assignList)
    elif rule_type == 'or':
        x_hat, xi_err, assignList = LearnRules(fname_traindump, mValue, alpha, beta, 1, timeout, rule_type, level,
                                        groupNoiseFlag, runIndex, assignList)
    else:
        assert False, "Need the rule-type to be (or/and)"
    rule_str_rec = recover_rule_df(A_df_train, x_hat, rule_type,level)
    print("PRINTING RULE")
    print(rule_str_rec)


    ###perform testing
    print("TEST PHASE\n")
    #timeout = 1200
    if rule_type == 'and':
        if (tool_type == 'sat'):
            x_hat, xi_err, assignList = LearnRules(fname_testdump, mValue, alpha, beta, 1, timeout, rule_type, level, 
                                        groupNoiseFlag, runIndex, assignList)
    elif rule_type == 'or':
        if (tool_type == 'sat'):
            x_hat, xi_err, assignList = LearnRules(fname_testdump, mValue, alpha, beta, 1, timeout, rule_type, level, 
                                        groupNoiseFlag, runIndex, assignList)
    else:
        assert False, "Need the rule-type to be (or/and)"
 
    ### calculate errors + display the rule
    nnz_rule = np.sum(np.abs(x_hat) >= 1e-4)
    rule_str_rec = recover_rule_df(A_df_train, x_hat, rule_type,level)
    print(rule_str_rec)
    #err,err_zero_to_one, err_ones_to_zero,y_hat = calculate_error(x_hat, rule_type, A, y)
    #print(("Learned rule with %d non-zeros and %d errors out of %d:" % (nnz_rule, err, len(y_hat))))
    #print(("Zeros to One errors %d and Ones to zeros %d:\n>>>") % (err_zero_to_one, err_ones_to_zero))

###########################################
def  get_col_to_features_map(A_df):
    ''' produce a column to feature map ''' 

    cols_A = A_df.columns
    feats = [col[0] for col in cols_A]    
    #feat_map = dict(list(enumerate(np.unique(feats))))
    feat_ind = {}; ind = 1  
    for f in feats:
        if not f in feat_ind:
            feat_ind[f] = ind
            ind+=1
    signs = [col[1] for col in cols_A]

    col_to_feat = []
    for ind in range(len(feats)):
        #print "%s %s" % (feats[ind], signs[ind])
        feat_sign = "-" if signs[ind] == '>' else ''
        feat_str = "%s%d" % (feat_sign, feat_ind[feats[ind]])
        #print feat_str
        col_to_feat.append(int(feat_str))

    print("")

    return col_to_feat
###########################################
def dump_LPrule_data(fname_datadump, A, y, col_to_feat):
    ''' save A, y, col_to_feat ''' 
    
    data_dict = {'A': A, 'y': list(y), 'col_to_feat' : col_to_feat} ## tolist()
    pickle.dump( data_dict, open(fname_datadump, "wb" ))


###########################################
def calculate_error(x_hat, rule_type, A, y):
    ''' evaluate the error of the BCS classifier
    here we're using a very basic rounding up (need to do better) '''

    thr_sm = 1e-3

    ### round the solution + make binary
    if rule_type == 'and':
        x_rnd = np.double(x_hat > thr_sm)
        y_hat = 1-np.double(np.dot(A,x_rnd) >= 1)

    elif rule_type == 'or':
        x_rnd = np.double(x_hat > thr_sm)
        y_hat = np.double(np.dot(A, x_rnd) >= 1)
    else:
        assert 0, "rule-types must be or/and "
    err = np.sum(y_hat != y)
    allOnes = np.ones_like(y)
    err_zero_to_one = np.sum(allOnes == (y_hat-y))
    err_ones_to_zero = np.sum(allOnes == (y - y_hat))
    return err, err_zero_to_one, err_ones_to_zero, y_hat

###########################################
def recover_rule_df(A_df, x_hat_vec, rule_type,level):
    ''' represent a learned rule in human-readable form, 
    data-frame version'''
    compound_str = '('
    for i in range(level):
        x_hat = x_hat_vec[i]
        inds_nnz = np.where(abs(x_hat) > 1e-4)[0]
        rule_str_rec = '';

        str_clauses = [' '.join(A_df.columns[ind]) for ind in inds_nnz]
        rule_sep = ' %s ' % rule_type
    
        rule_str = rule_sep.join(str_clauses)
        if (rule_type == 'and'):
            rule_str = rule_str.replace('<=','??').replace('>','<=').replace('??','>') 
        compound_str += rule_str
        if (i < level-1):
            if (rule_type == 'and'):
                compound_str += ' ) or ( '
            if (rule_type == 'or'):
                compound_str += ' ) and ( '
    compound_str += ')'
    return compound_str
###########################################
def recover_rule_compound(x_hat, ind_i_all, thr_all, rule_type, feat_names=None):
    ''' represent the learned rule in a human-readable form'''

    if feat_names: 
        do_feat_names = 1;
        lbl_name = feat_names[0]; var_names = feat_names[1:]
    else: 
        do_feat_names = 0

    if rule_type == 'and':
        rule_and_or_str = ' and ';
        ind_i_all = -ind_i_all;  ### we use de-morgan's laws to transform
        ### and(X^c) to not( or (x))        
    elif rule_type== 'or':
        rule_and_or_str = ' or ';
    else:
        assert False, "can only handle and/or rules"

    inds_nnz = np.where(abs(x_hat) > 1e-4)[0]
    rule_str_rec = ''

    #### start building up textual representation of the rule entry by entry
    for i, ind_c in enumerate(inds_nnz):
        ### ind_i_all contains two pieces of info: direction (via sign) and feature number (abs)
        sign_rule = np.sign(ind_i_all[ind_c])
        ind_var = abs(ind_i_all[ind_c])-1
        thr = thr_all[ind_c]
    
        if np.isnan(thr):  ### binary features            
            sign_str = ' not ' if sign_rule == -1 else '';
            if i > 1: 
                rule_str_rec = rule_str_rec + rule_and_or_str
        
            if do_feat_names:
                rule_str_rec = rule_str_rec +  " %s %s \n" % (sign_str, var_names[ind_var])
            else:
                rule_str_rec = rule_str_rec + " %s x(%d) \n" % (sign_str, ind_var)
    
        
        else:   ### continuous features
            sign_str = ' < ' if sign_rule == -1 else ' > '
            if i > 1: 
                rule_str_rec = rule_str_rec + rule_and_or_str
        
            if do_feat_names:
                rule_str_rec = rule_str_rec + " %s %s %.5f \n" % (var_names[ind_var], sign_str, thr)
            else:
                rule_str_rec = rule_str_rec + " x(%d) %s %.5f \n" % (ind_var, sign_str, thr)

    return rule_str_rec

###########################################
if __name__ == "__main__":
    
    #run_iris_example()

    
    ### Parse the command line arguments to understand what we need to do:
    parser = argparse.ArgumentParser()
    parser.add_argument("trainingfilename", help="Data filename")
    parser.add_argument("testfilename", help= "Test filename")
    parser.add_argument("-lambda_reg", nargs = '?', type = float, default = 1.0, 
                        help="Interpretability parameter")
    parser.add_argument("-rule_type", nargs = "?", choices = ["and", "or"], default = "or", 
                       help = "Type of a rule (and / or)")
    parser.add_argument("-tool_type", nargs="?", choices=["sat"], default="sat",help = "Type of tool (sat)")
    parser.add_argument("-timeout", nargs="?", type=float, default=1200, help="timeout for tool")
    parser.add_argument("-mValue", nargs="?", type = int, default=1, help="m for m of N")
    parser.add_argument("-groupNoise",nargs="?", choices=["0","1"], default="1", help="Group Noise Flag")
    parser.add_argument("-level",nargs="?",type=int,default=3, help="level")
    parser.add_argument("-runIndex", nargs="?",type=int,default=1,help="runIndex")
    args = parser.parse_args()
    training_data = args.trainingfilename
    test_data = args.testfilename
    lambda_reg = args.lambda_reg
    rule_type = args.rule_type
    tool_type = args.tool_type
    timeout = args.timeout
    mValue = args.mValue
    level = args.level
    groupNoiseFlag = False
    runIndex = args.runIndex
    if (args.groupNoise == "1"):
        groupNoiseFlag = True
    assert os.path.exists(training_data), "Filename does not exist %s" % training_data
    assert os.path.exists(test_data), "Filename does not exist %s" %test_data
    print("Info:"+str(tool_type)+":"+str(rule_type)+":"+str(level)+":"+str(lambda_reg)+"\n")
    print("Timeout:"+str(timeout))
    run_UCI_example(training_data, test_data, lambda_reg, rule_type, tool_type, mValue, timeout, level, groupNoiseFlag, runIndex)
