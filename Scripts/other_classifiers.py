import numpy as np, scipy as sp, pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model, metrics, svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score as AUC, accuracy_score


#############################################
def load_check_data(fname_tr, fname_tst):
    ''' load train and test data, check format, + convert to bin / numeric ''' 
    
    print "Loading file %s" % fname_tr
    X_tr_pd = pd.read_table(fname_tr, sep = ',', header= 0)
    print "Loading file %s" % fname_tst
    X_tst_pd = pd.read_table(fname_tst, sep = ',', header= 0)
    
    ### some data-sets (e.g. ilpd) have missing values (nans) -- fix these by replacing with 0's
    X_tr_pd.fillna(0, inplace=True)
    X_tst_pd.fillna(0, inplace=True)
    

    assert np.all(X_tr_pd.columns == X_tst_pd.columns), "Need X_tr and X_tst to have same features"

    N_tr = len(X_tr_pd)

    ### convert the format together --> then split
    X_tst_pd.index = X_tst_pd.index + N_tr
    X_pd_all = pd.concat([X_tr_pd, X_tst_pd])

    target_name = X_pd_all.columns[-1]

    #X_pd_all = X_pd_all.iloc[:100,:]
    X_pd_dict = X_pd_all.T.to_dict().values()
    
    vec = DictVectorizer()
    X_all = vec.fit_transform(X_pd_dict).toarray()
    feat_names = vec.get_feature_names()
    num_feat = len(feat_names)
    
    target_ind = feat_names.index(target_name)    
    feat_names = feat_names.remove(target_name)

    y_tr = X_all[:N_tr,target_ind].flatten()
    y_tst = X_all[N_tr:,target_ind].flatten()

    ## Split back into train and test
    inds_feat = range(num_feat); 
    inds_feat.remove(target_ind); 
    inds_feat = np.array(inds_feat)

    X_tr = X_all[:N_tr, inds_feat]   
    X_tst = X_all[N_tr:, inds_feat]


    return X_tr, X_tst, y_tr, y_tst, feat_names, target_name

#############################################
def  classify_exp(X_tr, X_tst, y_tr, y_tst):
    ''' train and evaluate a few classifiers ''' 
    

    ### We need to tune the params...!! (setting at some detault values for now)
    min_samp = 10
    rf = RandomForestClassifier(n_estimators = 500, min_samples_split = min_samp, n_jobs=-1)

    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': min_samp,
              'learning_rate': 0.01, 'loss': 'deviance'}
    gb = GradientBoostingClassifier(**params)

    C = 0.01
    logreg = linear_model.LogisticRegression(penalty = 'l1', C = C)
    C = 1.0
    svc = svm.SVC(kernel='linear', C=C, probability=True)

    classifiers = {"rf": rf, "logreg": logreg} ## "GradBoost": gb,"SVC" : svc
    results = {}  ### save results for all classifiers

    for cl_name in classifiers.keys():
        cl = classifiers[cl_name]
        
        cl.fit( X_tr, y_tr )
        y_hat = cl.predict( X_tst)
        p = cl.predict_proba( X_tst )
        
        acc = accuracy_score(y_tst, y_hat)
        auc = AUC( y_tst, p[:,1] )

        results[cl_name] = {'acc' : acc, 'auc': auc}
        
        print "%s: accuracy: %.2f, AUC: %.2f" % (cl_name, acc, auc)

    return results

#############################################
def sample_data_classify_exp():
    ''' run a few classifiers on sample train and test data ''' 

    data_dir = '/home/ksm/Research/kuldeep_ibm/LPRules/'
    
    #exp_name = 'adult_data_bintarget'
    exp_name = 'ilpd'; 
    
    num_exp = 10
    all_results = []

    for enum in range(num_exp):
        fname_tr = data_dir + 'Train/' + '%s_%d_train.csv' % (exp_name, enum)
        fname_tst = data_dir + 'Test/' + '%s_%d_test.csv' % (exp_name, enum)

        X_tr, X_tst, y_tr, y_tst, feat_names, target_name = load_check_data(fname_tr, fname_tst)
    

        results = classify_exp(X_tr, X_tst, y_tr, y_tst)
        all_results.append(results)

    ### now summarize the results
    all_cl = all_results[0].keys()
    num_cl = len(all_cl)
    all_acc = np.zeros([num_exp, num_cl])
    all_auc = np.zeros([num_exp, num_cl])
    for cl_ind, cl in enumerate(all_cl):
        for exp in range(num_exp):
            all_acc[exp, cl_ind] = all_results[exp][cl]['acc']
            all_auc[exp, cl_ind] = all_results[exp][cl]['auc']

            
    print "Acc:"
    print all_acc
    print "AUC:"
    print all_auc
    print ""
#############################################
def gen_logreg(C_grid = []):
    ''' generator of logreg classifiers param. by C'''

    if len(C_grid) == 0:
        C_grid = np.logspace(-3, 1, 10)

    for C in C_grid:
        logreg = linear_model.LogisticRegression(penalty = 'l1', C = C)
        
        yield(logreg, C)

#############################################
def gen_rf(min_samp_grid = []):
    ''' generator of rf classifiers param. by min_leaf'''

    if len(min_samp_grid) == 0:
        min_samp_grid = [3, 5, 10, 25, 50, 100, 250, 500]

    for min_samp in min_samp_grid:
        rf = RandomForestClassifier(n_estimators = 500, min_samples_split = min_samp, n_jobs=-1)
        
        yield(rf, min_samp)

#############################################
def run_all_classifiers_cv(cl_name):
    ''' get results for all the classifiers '''     
    
    data_dir = 'data/'

    datasets = ['adult_data_bintarget', 'credit_card_clients', 'ilpd', \
                'ionosphere', 'iris_bintarget', 'parkinsons', 'pima-indians-diabetes', \
                'TomsHardware_bintarget', 'transfusion', 'Twitter_bintarget_small', 'wdbc']
    #    datasets = ['ilpd', 'ionosphere']

    num_exp = 10
    
    for exp_name in datasets:

        results = []

        ### loop over the cv-fold (to be consistent with SAT results these are saved in files)
        for e_num in range(num_exp):     

            fname_tr = data_dir + 'Train/' + '%s_%d_train.csv' % (exp_name, e_num)
            fname_tst = data_dir + 'Test/' + '%s_%d_test.csv' % (exp_name, e_num)
        
            X_tr, X_tst, y_tr, y_tst, feat_names, target_name = load_check_data(fname_tr, fname_tst)

            if cl_name == 'rf':
                gen_cl = gen_rf; cv_param = 'min_samp'
            elif cl_name == 'logreg':
                gen_cl = gen_logreg; cv_param = 'C'

            ### generator over the classifier over a predefined range of paramters
            for cl, min_samp in gen_cl():
                
                cl.fit( X_tr, y_tr )
                y_hat = cl.predict( X_tst)
                p = cl.predict_proba( X_tst )
                
                acc = accuracy_score(y_tst, y_hat)
                auc = AUC( y_tst, p[:,1] )
            
                results.append([e_num, min_samp, acc, auc])

        print np.array(results)
    
        results_pd = pd.DataFrame(results, index=None, columns = ['exp_num', cv_param, 'acc', 'auc'])
    
        #print results_pd

        av_cv = results_pd.groupby(cv_param).mean()
        del av_cv['exp_num']
    
        print av_cv
    
        fname_out = 'Results/%s_%s_results.csv' % (exp_name, cl_name)
        av_cv.to_csv(fname_out)    


#############################################
def main():
    ''' pick your poison''' 

    #sample_data_classify_exp()
    for cl_name in ['logreg', 'rf']:
        run_all_classifiers_cv(cl_name = cl_name)

#############################################
if __name__ == "__main__":
    main()

