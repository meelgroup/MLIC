[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# IMLI

IMLI is an interpretable classification rule learning framework based on incremental mini-batch learning.  This tool can be used to learn classification rules expressible in propositional logic, in particular in [CNF, DNF](https://bishwamittra.github.io/publication/imli-ghosh.pdf), and [relaxed CNF](https://bishwamittra.github.io/publication/ecai_2020/paper.pdf).   

This tool  is based on our [CP-2018](https://arxiv.org/abs/1812.01843), [AIES-2019](https://bishwamittra.github.io/publication/imli-ghosh.pdf), and  [ECAI-2020](https://bishwamittra.github.io/publication/ecai_2020/paper.pdf) papers.






# Install
- Install the PIP library.
```
pip install pyrulelearn
```

- Run `pip install -r requirements.txt` to install all necessary python packages available from pip.

This framework requires installing an off-the-shelf MaxSAT solver to learn CNF/DNF rules. Additionally, to learn relaxed-CNF rules, an LP (Linear Programming) solver is required.

### Install MaxSAT solvers

To install Open-wbo, follow the instructions from [here](http://sat.inesc-id.pt/open-wbo/).
After the installation is complete, add the path of the binary to the PATH variable. 
```
export PATH=$PATH:'/path/to/open-wbo/'
```
Other off-the-shelf MaxSAT solvers can also be used for this framework.

### Install CPLEX

To install the linear programming solver, i.e., CPLEX, download and install it from [IBM](https://www.ibm.com/support/pages/downloading-ibm-ilog-cplex-optimization-studio-v1290).  To setup the Python API of CPLEX, follow the instructions from [here](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).

# Documentation

See the documentation in the [notebook](documentation.ipynb).

## Issues, questions, bugs, etc.
Please click on "issues" at the top and [create a new issue](https://github.com/meelgroup/MLIC/issues). All issues are responded to promptly.

## Contact
[Bishwamittra Ghosh](https://bishwamittra.github.io/) (bghosh@u.nus.edu)

## Citations


@inproceedings{GMM20,<br />
author={Ghosh, Bishwamittra and Malioutov, Dmitry and  Meel, Kuldeep S.},<br />
title={Classification Rules in Relaxed Logical Form},<br />
booktitle={Proc. of ECAI},<br />
year={2020},}

@inproceedings{GM19,<br />
author={Ghosh, Bishwamittra and  Meel, Kuldeep S.},<br />
title={{IMLI}: An Incremental Framework for MaxSAT-Based Learning of Interpretable Classification Rules},<br />
booktitle={Proc. of AIES},<br />
year={2019},}

@inproceedings{MM18,<br />
author={Malioutov, Dmitry and  Meel, Kuldeep S.},<br />
title={{MLIC}: A MaxSAT-Based framework for learning interpretable classification rules},<br />
booktitle={Proceedings of International Conference on Constraint Programming (CP)},<br />
month={08},<br />
year={2018},}

## Old Versions
The old version, MLIC (non-incremental framework) is available under the branch "MLIC". Please read the README of the old release to know how to compile the code. 
