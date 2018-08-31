# MLIC

The directory "Benchmarks" consists of all the benchmark files
The directory "Scripts" has most of the scripts that were employed in testing and required for reproducibility. In particular, we have added 
"discretization.py" script that takes in the non-binarized matrix of features and categories and discretizes the matrices. 
The directory "TrainingDataBehavior" has plots for other benchmarks
similar to Figure 1 

To Run MLIC.py; you will need maxhs and pbencoder (from PBLib) to be in
the PATH variable. You can pass python MLIC.py -h to see all the
possible options

other_classifiers.py contains code for comparing MaxSAT based approach with other machine learning tools e.g., neural network, support vector classifier, logistic regression, random forest and ripper. Ripper algorithm is supported by WEKA (jRIP). Carefully check the path of weka.jar, and make sure that the path is consistent with script (run_weka_RIPPER method in script). Note that, weka takes train and test file with .arff extension and place the .arff file in Scripts/Train and Scripts/Test folder respectively.  
