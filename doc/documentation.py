#!/usr/bin/env python
# coding: utf-8

# # 1. Learn Binary classification rules
# 
# This tutorial shows how to learn classification rules using MaxSAT-based incremental learning framework, IMLI. We show how to learn five popular classification rules under the same framework. 
# 
# - CNF rules (Conjunctive Normal Form)
# - DNF rules (Disjunctive Normal Form)
# - Decision sets
# - Decision lists
# - relaxed-CNF rules

# In[1]:


import sys
# sys.path.append("../")

from pyrulelearn.imli import imli
from pyrulelearn import utils
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[7]:


# Check if MaxSAT solver such as Open-WBO, MaxHS and MILP solver such as cplex is installed
import os
if(os.system("which open-wbo") != 0):
    print("Open-WBO is not installed")
if(os.system("which maxhs") != 0):
    print("MaxHS is not installed")
try:
    import cplex
except Exception as e:
    print(e)


# In[ ]:





# ### Model Configuration
# 
# Our first objective is to learn a classification rule in <em>CNF</em>, where the decision rule is ANDs of ORs of input features. For that, we specify `rule_type = CNF` inside the classification model `imli`. In this example, we learn a 2-clause rule with following hyper-parameters.
# 
# - `rule_type` sets the type of classification rule. Other possible options are DNF, decision sets, decision lists, relaxed_CNF,
# - `num_clause` decides the number of clauses in the classfication rule,
# - `data_fidelity` decides the weight on classification error during training,
# - `weight_feature` decides the weight of rule-complexity, that is, the cost of introducing a Boolean feature in the classifier rule,
# 
# 
# We require a MaxSAT solver to learn the Boolean rule. In this example, we use `open-wbo` as the MaxSAT solver. To install a MaxSAT solver, we refer to instructions in [README](../README.md).

# In[2]:


model = imli(rule_type="CNF", num_clause=2,  data_fidelity=10, weight_feature=1, timeout=100, solver="open-wbo", work_dir=".", verbose=False)


# ### Load dataset
# In this example, we learn a decision rule on `Iris` dataset. While the original dataset is used for multiclass classification, we modify it for binary classification. Our objective is to learn a decision rule that separates `Iris Versicolour` from other two classes of Iris: `Iris Setosa` and `Iris Virginica`. 
# 
# Our framework requires the training set to be discretized. In the following, we apply entropy-based discretization on the dataset. Alternatively, one can use already discretized dataset as a numpy object (or 2D list). To get the classification rule, `features` list has to be provided.

# In[3]:


X, y, features = utils.discretize_orange("../benchmarks/iris_orange.csv")
features, X


# ### Split dataset into train and test set

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ### Train the model

# In[5]:


model.fit(X_train,y_train)


# ### Report performance of the learned rule

# In[6]:


print("training report: ")
print(classification_report(y_train, model.predict(X_train), target_names=['0','1']))
print()
print("test report: ")
print(classification_report(y_test, model.predict(X_test), target_names=['0','1']))


# ### Show the learned rule

# In[7]:


rule = model.get_rule(features)
print("Learned rule is: \n")
print("An Iris flower is predicted as Iris Versicolor if")
print(rule)


# # 2. Learn decision rules as DNF
# 
# To learn a decision rule as a DNF (ORs of ANDs of input features), we specify `rule_type=DNF` in the hyper-parameters of the model. In the following, we learn a 2-clause DNF decision rule. 

# In[8]:


model = imli(rule_type="DNF", num_clause=2,  data_fidelity=10, solver="open-wbo", work_dir=".", verbose=False)
model.fit(X_train,y_train)
print("training report: ")
print(classification_report(y_train, model.predict(X_train), target_names=['0','1']))
print()
print("test report: ")
print(classification_report(y_test, model.predict(X_test), target_names=['0','1']))

print("\nRule:->")
print(model.get_rule(features))
print("\nOriginal features:")
print(features)
print("\nIn the learned rule, show original index in the feature list with phase (1: original, -1: complemented)")
print(model.get_selected_column_index())


# # 3. Learn more expressible decision rules: Relaxed-CNF rules
# 
# Our framework allows one to learn more expressible decision rules, which we call relaxed_CNF rules. This rule allows thresholds on satisfaction of clauses and literals and can learn more complex decision boundaries. See the [ECAI-2020](https://bishwamittra.github.io/publication/ecai_2020/paper.pdf) paper for more details. 
# 
# 
# In our framework, set the `rule_type=relaxed_CNF` to learn relaxed-CNF rules.

# In[9]:


model = imli(rule_type="relaxed_CNF", num_clause=2,  data_fidelity=10, solver="cplex", work_dir=".", verbose=False)
model.fit(X_train,y_train)
print("training report: ")
print(classification_report(y_train, model.predict(X_train), target_names=['0','1']))
print()
print("test report: ")
print(classification_report(y_test, model.predict(X_test), target_names=['0','1']))


# ### Understanding the decision rule
# 
# In this example, we ask the framework to learn a 2 clause rule. During training, we learn the thresholds on clauses and literals while fitting the dataset. The learned rule operates in two levels. In the first level, a clause is satisfied if the literals in the clause satisfy the learned threshold on literals. In the second level, the formula is satisfied when the threshold on clauses is satisfied.

# In[10]:


rule = model.get_rule(features)
print("Learned rule is: \n")
print("An Iris flower is predicted as Iris Versicolor if")
print(rule)
print("\nThrehosld on clause:", model.get_threshold_clause())
print("Threshold on literals: (this is a list where entries denote threholds on literals on all clauses)")
print(model.get_threshold_literal())


# # 4. Learn decision rules as decision sets and lists
# 
# ### Decision sets

# In[11]:


model = imli(rule_type="decision sets", num_clause=5,  data_fidelity=10, solver="open-wbo", work_dir=".", verbose=False)
model.fit(X_train,y_train)

print("\nRule:->")
print(model.get_rule(features))


print("\ntraining report: ")
print(classification_report(y_train, model.predict(X_train), target_names=['0','1']))
print()
print("test report: ")
print(classification_report(y_test, model.predict(X_test), target_names=['0','1']))


# ### Decision lists

# In[12]:


model = imli(rule_type="decision lists", num_clause=5,  data_fidelity=10, solver="open-wbo", work_dir=".", verbose=False)
model.fit(X_train,y_train)

print("\nRule:->")
print(model.get_rule(features))


print("\ntraining report: ")
print(classification_report(y_train, model.predict(X_train), target_names=['0','1']))
print()
print("test report: ")
print(classification_report(y_test, model.predict(X_test), target_names=['0','1']))

