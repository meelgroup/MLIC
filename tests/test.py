from pyrulelearn.imli import imli as imli
model = imli(rule_type="relaxed_CNF", num_clause=3, solver="open-wbo",  data_fidelity=10,  work_dir=".", verbose=True)
X, y, features = model.discretize_orange("../benchmarks/iris_orange.csv")
# split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
# Train the model
model.fit(X_train,y_train)
# Access the performance of the model. 
from sklearn.metrics import confusion_matrix
def measurement(cnf_matrix):
    # print(cnf_matrix)
    TN, FP, FN, TP = cnf_matrix.ravel()

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC*100

yhat_train = model.predict(X_train, y_train)
_, _, _, _, _, _, _, train_acc = measurement(confusion_matrix(y_train, yhat_train))
yhat_test = model.predict(X_test, y_test)
_, _, _, _, _, _, _, test_acc = measurement(confusion_matrix(y_test, yhat_test))
print("training    accuracy: ", train_acc)
print("test        accuracy: ", test_acc)

# print rule:
rule = model.get_rule(features)
print("Learned rule is: ")
print(rule)
print("index of selected columns in the rule", model.get_selected_column_index())
print(model.get_threshold_clause())
print(model.get_threshold_literal())