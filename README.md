[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# IMLI

IMLI is an interpretable classification rule learning framework based on incremental mini-batch learning setting.  This tool can be used to learn classification rules expressible in propositional logic, in particular, in [CNF, DNF](https://bishwamittra.github.io/publication/imli-ghosh.pdf), and [relaxed CNF](https://bishwamittra.github.io/publication/ecai_2020/paper.pdf).   

This tool  is based on our [AIES-2019](https://bishwamittra.github.io/publication/imli-ghosh.pdf) and  [ECAI-2020](https://bishwamittra.github.io/publication/ecai_2020/paper.pdf) papers.




## Directory Description

The directory `benchmarks/` consists of all the benchmark files used for the experiment. 

The directory `rulelearning` contains all the scripts that are employed in testing and required for reproducibility. 
In the `rulelearning` directory, we have added `imli.py` which is the incremental learning framework for generating interpretable classification rules. 
<!-- To run `imli.py`, you will need an off the self MaxSAT solver (e.g., Open-Wbo) to be in the PATH variable. -->
To learn CNF/DNF rules, one would require an off-the-shelf MaxSAT solver (default is open-wbo) to be installed and added to the PATH variable. To learn relaxed CNF rules, one would require a linear programming solver (default is CPLEX)  to be installed. 

<!-- ## PIP Install
Run the following command to install the python library.

```
pip install rulelearning
``` -->

## Install MaxSAT solvers

To install Open-wbo, follow the instructions from [here](http://sat.inesc-id.pt/open-wbo/).
After the installation is complete, add the path of the binary to the PATH variable. 
```
export PATH=$PATH:'/path/to/open-wbo/'
```
Other off-the-shelf MaxSAT solvers can also be used for this framework.

## Install CPLEX

To install the linear programming solver, i.e., CPLEX, download and install it from [IBM](https://www.ibm.com/support/pages/downloading-ibm-ilog-cplex-optimization-studio-v1290).  To setup the Python API of CPLEX, follow the instructions from [here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

## Install Orange3

IMLI incorporates entropy-based discretization based on Orange3 library. Follow the instructions in the [Github](https://github.com/biolab/orange3) repository to install Orange3. 

## Usage

Import the Python scripts.
```
from rulelearning import imli
```

Call an instance of `imli` object and specify the parameters in `imli()`. For example, in order to  learn CNF rules, set `rule_type="CNF"`. Similarly to learn relaxed CNF rules, set `rule_type="relaxed_CNF"`.
```
model=imli.imli()
```
Discretize the dataset in the csv format using the following command. `benchmarks/` contains  a set of sample datasets.
```
X, y, features = model.discretize("iris.csv")
```
The return values are feature matrix `X`, target vector `y`, and list of discretized features `features`.

If the dataset contains categorical features, specify the index of such categorical features as a list in the parameter. Look for other parameter choices in `imli.discretize()`, i.e., number of bins. 
```
X, y, features = model.discretize("credit.csv",categorical_column_index=[2,3,4])
```
`imli.discretize()` method applies quantile-based technique in the disretization process. To apply frequency-based discretization, alternately use the following command. For that, one would require to specify the feature-type in the input CSV file. For more details, follow the instructions in `benchmarks/` directory.
```
X, y, features = model.discretize_orange("iris_orange.csv")
```

Train the model. 
```
model.fit(X,y)
```
To retrive the learned rule, call the `get_rule()` function with parameter `features`.
```
rule = model.get_rule(features)
```
To compute the predictions on a test set e.g., `{XTest,yTest}`, call `predict(Xtest,yTest)` method.
```
yhat = model.predict(XTest,yTest)
```
Play with other getter functions to learn different attributes of the trained model. For more details, refer to the source code.

## Issues, questions, bugs, etc.
Please click on "issues" at the top and [create a new issue](https://github.com/meelgroup/MLIC/issues). All issues are responded to promptly.

## Contact
[Bishwamittra Ghosh](https://bishwamittra.github.io/) (bghosh@u.nus.edu)

## How to cite

Please cite the following two papers.

@inproceedings{GM19,<br />
author={Ghosh, Bishwamittra and  Meel, Kuldeep S.},<br />
title={{IMLI}: An Incremental Framework for MaxSAT-Based Learning of Interpretable Classification Rules},<br />
booktitle={Proc. of AIES},<br />
year={2019},}

@inproceedings{GMM20,<br />
author={Ghosh, Bishwamittra and Malioutov, Dmitry and  Meel, Kuldeep S.},<br />
title={Classification Rules in Relaxed Logical Form},<br />
booktitle={Proc. of ECAI},<br />
year={2020},}

## Old Versions
The old version, MLIC (non-incremental framework) is available under the branch "MLIC". Please read the README of the old release to know how to compile the code. 
