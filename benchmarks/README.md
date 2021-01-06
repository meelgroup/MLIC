# Datasets description

Find the full set of datasets from this [link](https://drive.google.com/drive/folders/1HFAxx1jM9mvnXscXXso5OR9OdoJL0ZWs?usp=sharing).
This link contains two folders: `converted_for_orange_library` and `quantile_based_discretization`. 

## Prepare datasets for  entropy-based discretization

`converted_for_orange_library` contains datasets that can be passed to the subroutine `imli.discretization_orange()`. This subroutine is based on the [entropy-based feature discretization](https://www.ijcai.org/Proceedings/93-2/Papers/022.pdf) library of [Orange](https://docs.biolab.si//3/data-mining-library/reference/preprocess.html#discretization). To prepare a dataset for `imli.discretization_orange()`,  modify the feature names as follows

1. For a categorial\discrete feature, add `D#` to the feature name. For example, if `Gender={female, male, others}` is a categorial feature in the dataset, the modified feature name is `D#Gender`.  
2. For a continuous-valued feature, add `C#`. For example, `income` is modified as `C#income`.
3. For the target (discrete) column, add `cD#`. For example, `defaulted`, that is the target column, is modified as `cD#defaulted`. 
4. To ignore any feature, add `i#` to the feature name. 

For more details, review the instructions from the Orange [documentation](https://docs.biolab.si//3/data-mining-library/reference/data.io.html).
