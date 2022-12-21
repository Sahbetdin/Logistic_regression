# Logistic_regression
home-written logistic regression fitting algorithm. The data consists of columns that are points in Hogwarts School. The targets are the coalitions to which each student belongs.


For training use: ```python logreg_train.py dataset_train.csv```.
The script uses datasets/dataset_train.csv as training data.
That will end up with model_params.pkl, scaler_params.pkl files.
The first one contains logistic regresiion parameters, the other one are scaling parameters used for scaling columns before entering model. 


For predicting use: ```python logreg_train.py dataset_test.csv```
This will form houses.csv file where for each participant predicted coalitions are stored.
