# MLICv2

MLICv2 is an incremental learning framework based on  MaxSAT  for generating interpretable classification rules via partition-based training methodology. This tool  is based on our [paper](https://bishwamittra.github.io/publication/imli-ghosh.pdf) published in AAAI/ACM Conference on AI, Ethics, and Society(AIES), 2019. We have introduced this framework as imli in the paper.





# Directory Description

The directory `benchmarks/` consists of all the benchmark files used for the experiment. 

The directory `rulelearning` contains all the scripts that are employed in testing and required for reproducibility. 
In the `rulelearning` directory, we have added `imli.py` which is the incremental learning framework for generating interpretable rules. To run `imli.py`, you will need an off the self MaxSAT solver (e.g., MaxHS, Open-Wbo) to be in the PATH variable.

# PIP Install
Run the following command to install this framework.

```
pip install rulelearning
```

# Install MaxSAT solvers

To install Open-wbo, follow the instructions in the official [link](http://sat.inesc-id.pt/open-wbo/).
To install MaxHS, follow the instructions in the official [link](http://www.maxhs.org/docs/overview.html).
After the installation is complete, add the path of the binary in the PATH variable. 
```
export PATH=$PATH:'/path/to/open-wbo/'
```
Or
```
export PATH=$PATH:'/path/to/maxhs/'
```
Other off the shelf MaxSAT solvers can also be used for this framework.
# Usage

Import rulelearning in Python.
```
import rulelearning
```

Call an instance of `imli` object. Specify the parameters in `imli()` if needed. For example, in order to  set `open-wbo` as the MaxSAT solver, pass the parameter `solver="open-wbo"`.
```
model=rulelearning.imli()
```
Discretize any dataset in csv format by calling the following function. `benchmarks/` contains  a set of sample datasets.
```
X,y=model.discretize("benchmarks/iris_bintarget.csv")
```
 If the dataset contains categorical features, specify the index of such categorical features as a list in the parameter. Look for other parameter choices too. For example:
```
X,y=model.discretize("benchmarks/credit_card_clients.csv",categoricalColumnIndex=[2,3,4])
```
Train the model as follows. 
```
model.fit(X,y)
```
To retrive the learned rule, call the `getRule()` function.
```
model.getRule()
```
To compute the predictions on a test set e.g., `{XTest,yTest}`, call `predict(Xtest,yTest)` function.
```
yhat=model.predict(XTest,yTest)
```
Play with other getter functions to learn various attributes of the trained model.


For more details, refer to the source code.

# Issues, questions, bugs, etc.
Please click on "issues" at the top and [create a new issue](https://github.com/meelgroup/MLIC/issues). All issues are responded to promptly.

# Contact
[Bishwamittra Ghosh](https://bishwamittra.github.io/) (bishwa@comp.nus.edu.sg)

# How to cite
@inproceedings{GM19,<br />
author={Ghosh, Bishwamittra and  Meel, Kuldeep S.},<br />
title={imli: An Incremental Framework for MaxSAT-Based Learning of Interpretable Classification Rules},<br />
booktitle={Proceedings of AAAI/ACM Conference on AI, Ethics, and Society(AIES)},<br />
month={1},<br />
year={2019},}

# Old Versions
The old version, MLIC is available under the branch "MLIC". Please read the README of the old release to know how to compile the code. 
