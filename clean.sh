rm -r build dist *egg-info
find . -type d -name  "__pycache__" -exec rm -r {} +
find . -type d -name  ".ipynb_checkpoints" -exec rm -r {} +
rm */*model*.*
rm model*.*
