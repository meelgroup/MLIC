# MLIC
This system provides MaxSAT based framework for learning interpretable classification rule. The original paper(CP 2018) can be found in the following link. 
[https://www.comp.nus.edu.sg/~meel/Papers/CP2018mm.pdf]

# Directory Description

The directory "Benchmarks" consists of all the benchmark files used for experiment. 

The directory "Scripts" has all the scripts that were employed in testing and required for reproducibility. 
"Script" has two subdirectory. In the "Scripts/MLIC" subdirectory we have added "MLIC.py" script which is entry script of this system. To Run MLIC.py, you will need maxhs and pbencoder (from PBLib) to be in
the PATH variable. 


"Scripts" also contains some additional scripts. In "Scripts/MLIC" we have added "discretization.py" script that takes in the non-binarized matrix of features and categories and discretizes the matrices. "Load_bcsrule_data.py" and "Load_process_data_BCS.py" scripts contain some supplementary methods invoked in "MLIC.py". Moreover "Scripts/RuleLearning" subdirectory contains "MultiLevelLearnRules.py" and we have implemented maxSAT based systems in  "MultiLevelLearnRules.py" script. "Scripts/MLIC" also contain two subdirectories "Train" and "Test" that contain cross folded training and test data respectively.   


The directory "TrainingDataBehavior" has plots for other benchmarks
similar to Figure 1  of MLIC paper. 



other_classifiers.py contains code for comparing MaxSAT based approach with other machine learning tools e.g., neural network, support vector classifier, logistic regression, random forest and ripper. Ripper algorithm is supported by WEKA (jRIP). Carefully check the path of weka.jar, and make sure that the path is consistent with script (run_weka_RIPPER method in script). Note that, weka takes train and test file with .arff extension and place the .arff file in "Scripts/MLIC/Train" and "Scripts/MLIC/Test" folder respectively. 

# Usage

Run MLIC.py in "Scripts\MLIC" diretory
```
python MLIC.py <trainingfile> <testfile>
```
You can pass `python MLIC.py -h` to see all the
possible options. 

Run other_classifiers.py in root directory
```
python other_classifiers.py <classifier name> <Benchmark no> 
```
other_classifiers.py is configured for a fixed number of provided benchmarks. You can add more benchmarks in "Benchmarks/Data" directory and put separate training and test data in "Scripts/MLIC/Train" and "Scripts/MLIC/Test" directory respectively.
# Contact
1. Dmitry Maliotov (dmal@alum.mit.edu)
2. Kuldeep Meel (meel@comp.nus.edu.sg)
