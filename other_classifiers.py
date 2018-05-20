import numpy as np, scipy as sp, pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model, metrics, svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score as AUC, accuracy_score

import subprocess, os, re, time, argparse

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

    data_dir = os.getcwd()+"/" 
    
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
def gen_nn(num_nbs_grid = [], max_nn = None, dist_p = 1):
    ''' generator of nearest-neighbors classifiers param. by num-nbs'''

    if len(num_nbs_grid) == 0:
        num_nbs_grid = [1, 3, 5, 11, 25, 51, 101, 501]

    if max_nn:
        num_nbs_grid = [x for x in num_nbs_grid if x <= max_nn]
    
    for nn in num_nbs_grid:
        knn = KNeighborsClassifier(n_neighbors = nn, p = dist_p)
        
        yield(knn, nn)
#############################################
def gen_svc(C_grid = []):
    ''' generator of SVC classifiers param. by C'''

    if len(C_grid) == 0:
        C_grid = np.logspace(-3, 1, 10)

    for C in C_grid:
        svc = svm.SVC(kernel='linear', C=C, probability=True)
        
        yield(svc, C)
#############################################
def run_all_classifiers_cv(cl_name,example_index, param_index):
    ''' get results for all the classifiers '''     
    
    data_dir = os.getcwd()+'/'

    all_datasets = ['iris_bintarget', 'ilpd', 'adult_data_bintarget', 'credit_card_clients', \
                'ionosphere', 'iris_bintarget', 'parkinsons', 'pima-indians-diabetes', \
                'TomsHardware_bintarget', 'transfusion', 'Twitter_bintarget_small', 'wdbc']
    datasets = []
    if (example_index == -1):
        datasets = all_datasets
    else:
        datasets.append(all_datasets[example_index])
    
    num_exp = 10
    for exp_name in datasets:
        results = []
        timeTakenData = []
        ### loop over the cv-fold (to be consistent with SAT results these are saved in files)
        for e_num in range(num_exp):     

            fname_tr = data_dir + 'Train/' + '%s_%d_train.csv' % (exp_name, e_num)
            fname_tst = data_dir + 'Test/' + '%s_%d_test.csv' % (exp_name, e_num)
        
            X_tr, X_tst, y_tr, y_tst, feat_names, target_name = load_check_data(fname_tr, fname_tst)

            if cl_name == 'rf':
                gen_cl = gen_rf; cv_param = 'min_samp'
            elif cl_name == 'logreg':
                gen_cl = gen_logreg; cv_param = 'C'
            elif cl_name == 'nn':
                gen_cl = lambda: gen_nn(max_nn = len(y_tr)); cv_param = 'num_nbs'
            elif cl_name == 'svc':
                gen_cl = gen_svc; cv_param = 'C'
            else:
                assert False, "No such classifier %s" % cl_name
            print type(gen_cl())
            #cl, min_samp= gen_cl()[0]
            
        
            ### generator over the classifier over a predefined range of paramters
            #for i in range(0,1):
            param_wait_index = 0
            for cl, min_samp in gen_cl():
                if (not(param_wait_index == param_index)):
                    param_wait_index += 1
                    continue
                param_wait_index += 1
                startTime = time.time()
                cl.fit( X_tr, y_tr )
                endTime = time.time()
                timeTaken = endTime- startTime
                y_hat = cl.predict( X_tst)
                p = cl.predict_proba( X_tst )
                
                acc = accuracy_score(y_tst, y_hat)
                auc = AUC( y_tst, p[:,1] )
            
                results.append([e_num, min_samp, acc, auc])
                timeTakenData.append([e_num, min_samp, timeTaken])
        print np.array(results)
    
        results_pd = pd.DataFrame(results, index=None, columns = ['exp_num', cv_param, 'acc', 'auc'])
        timeTaken_pd = pd.DataFrame(timeTakenData, index = None, columns = ['exp_num', cv_param, 'Time'])
        #print results_pd

        av_cv = results_pd.groupby(cv_param).mean()
        av_time = timeTaken_pd.groupby(cv_param).mean()
        del av_cv['exp_num']
        del av_time['exp_num']
        fname_out = 'Results/%s_%s_%d_results.csv' % (exp_name, cl_name,param_index)
        av_cv.to_csv(fname_out)
        fname_out = 'Results/time_%s_%s_%d_results.csv' % (exp_name, cl_name, param_index)
        av_time.to_csv(fname_out)

############################################
def parse_weka_training_out(weka_out):
    andCount = 0
    shouldConsiderRuleCount = False
    shouldConsiderAccuracy = False
    learnedRules = ''
    for line in weka_out.split('\n'):
        line = line.strip()
        if re.match('Number of Rules',line):
            shouldConsiderRuleCount = False
        if (re.match('Correctly Classified Instances',line)):
            field_list = line.split()
            training_accuracy = float(field_list[-2])/100
            return andCount, training_accuracy, learnedRules
        if (shouldConsiderRuleCount):
            if (not(bool(re.search('=>',line)))):
                continue
            learnedRules +=line+'\n'
            field_list = line.split(' and ')
            andCount += len(field_list)
        if (re.match('JRIP rules',line)):
            shouldConsiderRuleCount = True

    return 0
#############################################
def parse_weka_out(weka_out, do_debug = False):
    '''  parse the weird output weka file '''

    y_true, y_hat = [], []

    for line in weka_out.split('\n'):
        line = line.strip()
        if re.match('=== Predictions', line):
            continue
        if re.match('inst#', line):
            continue
        if len(line) == 0:
            continue

        field_list = line.split()
        if len(field_list) == 5:
            del field_list[3]
        
        if do_debug:
            print ' | '.join(field_list)

        y_true_ln = field_list[1].split(':')[1]
        y_hat_ln = field_list[2].split(':')[1]
        y_true.append(y_true_ln)
        y_hat.append(y_hat_ln)

    y_true_arr = np.array(y_true)
    y_hat_arr = np.array(y_hat)
    acc = np.sum(y_true_arr == y_hat_arr)*1.0/ len(y_hat_arr)

        
    return acc
#############################################
def run_weka_RIPPER(cl_name,example_index):
    ''' get results for weka RIPPER ''' 

    data_dir = os.getcwd()+'/'

    all_datasets = ['ilpd','adult_data_bintarget', 'credit_card_clients', \
                'ionosphere', 'iris_bintarget', 'parkinsons', 'pima-indians-diabetes', \
                'TomsHardware_bintarget', 'transfusion', 'Twitter_bintarget_small', 'wdbc']
    datasets = []
    if (example_index == -1):
        datasets = all_datasets
    else:
        datasets.append(all_datasets[example_index])
    print type(datasets)
    for exp_name in datasets:
        ### weka params
        weka_cp = os.environ.get('WEKA_PATH')+'weka.jar'
        weka_cl = 'weka.classifiers.rules.JRip'

        results = []
        timeTakenData = []
        ### pre-computed cv-fold
        ruleText = ''
        for e_num in range(0,10):

            fname_tr = data_dir + 'Train/' + '%s_%d_train.arff' % (exp_name, e_num)
            fname_tst = data_dir + 'Test/' + '%s_%d_test.arff' % (exp_name, e_num)
        
            weka_model_fname = 'TempData/%s_%d_jRip.model' % (exp_name, e_num)
            
            for min_leaf in [1, 2, 3, 5, 10, 15, 25, 50, 100]:
                print ">>>>>>>>>> trying min_leaf %d" % min_leaf    

                weka_tr_cmd = 'java -Xmx8G -cp %s %s -N %d -t %s -x 10 -c -1 -d %s' % (weka_cp, weka_cl, min_leaf, fname_tr, weka_model_fname)
                print "Executing %s" % weka_tr_cmd
                startTime = time.time()
                p = subprocess.Popen(weka_tr_cmd, stdout=subprocess.PIPE, shell=True)
                (out, err) = p.communicate()
                p_status = p.wait()
                endTime = time.time()
                timeTaken = endTime - startTime
                andCount,training_accuracy, learnedRules = parse_weka_training_out(out)
                
                print "Evaluating the learned model on test-set %s" % fname_tst
                weka_tst_cmd = 'java -Xmx8G -cp %s %s -l %s -T %s -p 0' % (weka_cp, weka_cl, weka_model_fname, fname_tst)
                print weka_tst_cmd
                p = subprocess.Popen(weka_tst_cmd, stdout=subprocess.PIPE, shell=True)
                (weka_out, err) = p.communicate()
                p_status = p.wait()
                
                acc = parse_weka_out(weka_out)
                print "%s accuracy is %.4f" % (weka_cl, acc)
                ruleText += 'Exp_'+str(e_num)+'_'+str(min_leaf)+' '+str(acc)+' '+str(andCount)+'\n'
                ruleText += learnedRules+'\n'
                results.append([e_num, min_leaf, acc, andCount,training_accuracy])
                timeTakenData.append([e_num, min_leaf, timeTaken])
                
        print np.array(results)
        results_pd = pd.DataFrame(results, index=None, columns = ['exp_num', 'min_leaf', 'acc', 'RuleSize', 'accuracy'])    
        #print results_pd
        timeTaken_pd = pd.DataFrame(timeTakenData, index = None, columns = ['exp_num', 'min_leaf', 'Time'])
        
        av_time = timeTaken_pd.groupby('min_leaf').mean()
        av_cv = results_pd.groupby('min_leaf').mean()
        del av_cv['exp_num']
        del av_time['exp_num']

        print av_cv
        fname_out = 'Results/%s_%s_results.csv' % (exp_name, cl_name)
        av_cv.to_csv(fname_out)
        fname_out = 'Results/time_%s_%s_results.csv' % (exp_name, cl_name)
        av_time.to_csv(fname_out)
        f = open('Results/rulesRipper_'+str(exp_name)+'.txt','w')
        f.write(ruleText)
        f.close()
    print "Done."

#############################################
def main():
    ''' pick your poison''' 

    parser = argparse.ArgumentParser()
    parser.add_argument("cl_name", help="options: nn, svc, logreg, rf, rip")
    parser.add_argument("index", help="add index of examples", default=-1, type=int)
    parser.add_argument("param_index", help="index of params", default=0, type = int)
    args = parser.parse_args()
    cl_name = args.cl_name
    example_index = args.index
    param_index = args.param_index
    if (cl_name == 'rip'):
        run_weka_RIPPER(cl_name = cl_name,example_index = example_index)
    else:
        run_all_classifiers_cv(cl_name = cl_name, example_index = example_index, param_index = param_index)
    #sample_data_classify_exp()

#############################################
if __name__ == "__main__":
    main()

