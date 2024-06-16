import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SVMSMOTE

df=pd.read_csv("drive/MyDrive/4th_Year_Research/Implementation/Dataset/Pre-Processed-Dataset.csv")
df.shape

"""# Splitting data into X and Y"""

X = df.drop('Target', axis=1)
y = df['Target']

X.shape

X.info()

y.head()

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(X.shape, X_train.shape, X_test.shape)



"""# ADASYN"""

# svm_smote = SVMSMOTE(sampling_strategy='auto', random_state=42, n_jobs=-1)

# # Apply SVM-SMOTE to the training data only
# X_train_resampled, y_train_resampled = svm_smote.fit_resample(X_train, y_train)

from imblearn.over_sampling import ADASYN

adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)

X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

count_of_ones = (X_train_resampled == 1).sum()
count_of_zeros = (y_train_resampled == 0).sum()
print(f"Number of rows with Target = 0: {count_of_zeros}")
print(f"Number of rows with Target = 1: {count_of_ones }")

print(X_train_resampled.shape, y_train_resampled.shape)

"""# **Model Training**

# Logistic Regression
"""

# Train Logistic Regression model
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Predict on training set
y_train_pred = clf.predict(X_train_resampled)
y_train_proba = clf.predict_proba(X_train_resampled)[:, 1]

# Predict on validation set
y_val_pred = clf.predict(X_val)
y_val_proba = clf.predict_proba(X_val)[:, 1]

# Calculate accuracy and F1 score for training set
accuracy_train = accuracy_score(y_train_resampled, y_train_pred)
f1_train = f1_score(y_train_resampled, y_train_pred)

# Calculate accuracy and F1 score for validation set
accuracy_val = accuracy_score(y_val, y_val_pred)
f1_val = f1_score(y_val, y_val_pred)

# Calculate ROC AUC score for training set
roc_auc_train = roc_auc_score(y_train_resampled, y_train_proba)

# Calculate ROC AUC score for validation set
roc_auc_val = roc_auc_score(y_val, y_val_proba)

# Print the metrics for both the training and validation sets
print(f"Logistic Regression\n")
print(f"Training Accuracy: {accuracy_train:.4f}")
print(f"Training F1 Score: {f1_train:.4f}")
print(f"Training ROC AUC Score: {roc_auc_train:.4f}")
print(f"Validation Accuracy: {accuracy_val:.4f}")
print(f"Validation F1 Score: {f1_val:.4f}")
print(f"Validation ROC AUC Score: {roc_auc_val:.4f}")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Plot ROC curve for the validation set
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Validation ROC curve (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear', 'saga'],
#     'max_iter': [100, 1000, 10000] # These solvers work well with l1 and l2 penalties.
# }
# grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', verbose=1)

# grid_search.fit(X_train, y_train)

# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)
# print("Best Model's Parameters: ", grid_search.best_params_)
# # print("Best Model's Score: ", grid_search.best_score_)
# # print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # -----------------------
# # Predict on training and testing data
# y_train_pred = clf.predict(X_train)
# y_test_pred = clf.predict(X_test)

# # Calculate accuracies
# train_accuracy = accuracy_score(y_train, y_train_pred)
# test_accuracy = accuracy_score(y_test, y_test_pred)

# print(f'Training Accuracy: {train_accuracy}')
# print(f'Testing Accuracy: {test_accuracy}')
# print('\n')
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

lr = LogisticRegression(random_state=42)
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'], 
    'max_iter' :[10,50,100,150,200]
}
    # 'max_iter' :[50,100,150,300,500]  -> 100
    # 'max_iter' :[90,100,110] -> 100
    # 'max_iter' :[98,99,100,101,102,103] -> 100

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

grid_search.fit(X_train_resampled, y_train_resampled)

print(f"Logistic Regression\n")
print("Best Parameters:", grid_search.best_params_)

y_train_pred_gs = grid_search.best_estimator_.predict(X_train_resampled)
y_val_pred_gs = grid_search.best_estimator_.predict(X_val)

accuracy_train_gs = accuracy_score(y_train_resampled, y_train_pred_gs)
f1_train_gs = f1_score(y_train_resampled, y_train_pred_gs)

accuracy_val_gs = accuracy_score(y_val, y_val_pred_gs)
f1_val_gs = f1_score(y_val, y_val_pred_gs)

# ********************
y_train_pred = clf.predict(X_train_resampled)
y_train_proba = clf.predict_proba(X_train_resampled)[:, 1]

y_val_pred = clf.predict(X_val)
y_val_proba = clf.predict_proba(X_val)[:, 1]

accuracy_train = accuracy_score(y_train_resampled, y_train_pred)
f1_train = f1_score(y_train_resampled, y_train_pred)

accuracy_val = accuracy_score(y_val, y_val_pred)
f1_val = f1_score(y_val, y_val_pred)

roc_auc_train = roc_auc_score(y_train_resampled, y_train_proba)

roc_auc_val = roc_auc_score(y_val, y_val_proba)

print(f"Training Accuracy: {accuracy_train:.4f}")
print(f"Training F1 Score: {f1_train:.4f}")
print(f"Training ROC AUC Score: {roc_auc_train:.4f}")
print(f"Validation Accuracy: {accuracy_val:.4f}")
print(f"Validation F1 Score: {f1_val:.4f}")
print(f"Validation ROC AUC Score: {roc_auc_val:.4f}")

"""# Decision Trees"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Train Decision Tree model
dt = DecisionTreeClassifier(random_state=0,max_depth=4,min_samples_split=2)
dt.fit(X_train_resampled, y_train_resampled)

# Predict on training set
y_train_pred = dt.predict(X_train_resampled)
y_train_proba = dt.predict_proba(X_train_resampled)[:, 1]

# Predict on validation set
y_val_pred = dt.predict(X_val)
y_val_proba = dt.predict_proba(X_val)[:, 1]

# Calculate accuracy and F1 score for training set
accuracy_train = accuracy_score(y_train_resampled, y_train_pred)
f1_train = f1_score(y_train_resampled, y_train_pred)

# Calculate accuracy and F1 score for validation set
accuracy_val = accuracy_score(y_val, y_val_pred)
f1_val = f1_score(y_val, y_val_pred)

# Calculate ROC AUC score for training set
roc_auc_train = roc_auc_score(y_train_resampled, y_train_proba)

# Calculate ROC AUC score for validation set
roc_auc_val = roc_auc_score(y_val, y_val_proba)

# Print the metrics for both the training and validation sets
print(f"Decision Trees\n")

print(f"Training Accuracy: {accuracy_train:.4f}")
print(f"Training F1 Score: {f1_train:.4f}")
print(f"Training ROC AUC Score: {roc_auc_train:.4f}")
print(f"Validation Accuracy: {accuracy_val:.4f}")
print(f"Validation F1 Score: {f1_val:.4f}")
print(f"Validation ROC AUC Score: {roc_auc_val:.4f}")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Plot ROC curve for the validation set
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Validation ROC curve (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

"""HyperPT DT"""

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

dt = DecisionTreeClassifier(random_state=42)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5,10,20,40,60,80],
    'min_samples_split': [5,10,20,30,40,50],
    'min_samples_leaf': [1, 2,3]
}

grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

grid_search_dt.fit(X_train_resampled, y_train_resampled)
print(f"Decision Trees HPT\n")
print("Best Parameters:", grid_search_dt.best_params_)
y_train_pred_gs_dt = grid_search_dt.best_estimator_.predict(X_train_resampled)
y_val_pred_gs_dt = grid_search_dt.best_estimator_.predict(X_val)

accuracy_train_gs_dt = accuracy_score(y_train_resampled, y_train_pred_gs_dt)
f1_train_gs_dt = f1_score(y_train_resampled, y_train_pred_gs_dt)

accuracy_val_gs_dt = accuracy_score(y_val, y_val_pred_gs_dt)
f1_val_gs_dt = f1_score(y_val, y_val_pred_gs_dt)

roc_auc_train_gs_dt = roc_auc_score(y_train_resampled, grid_search_dt.best_estimator_.predict_proba(X_train_resampled)[:, 1])

roc_auc_val_gs_dt = roc_auc_score(y_val, grid_search_dt.best_estimator_.predict_proba(X_val)[:, 1])

print(f"Decision Tree Training Accuracy: {accuracy_train_gs_dt:.4f}, Training F1 Score: {f1_train_gs_dt:.4f}, Training ROC AUC Score: {roc_auc_train_gs_dt:.4f}")
print(f"Decision Tree Validation Accuracy: {accuracy_val_gs_dt:.4f}, Validation F1 Score: {f1_val_gs_dt:.4f}, Validation ROC AUC Score: {roc_auc_val_gs_dt:.4f}")

"""# Random Forest Trees"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf = RandomForestClassifier(max_depth=5, random_state=0)
clf_rf.fit(X_train_resampled, y_train_resampled)

# Predict on training set
y_train_pred_rf = clf_rf.predict(X_train_resampled)
y_train_proba_rf = clf_rf.predict_proba(X_train_resampled)[:, 1]

# Predict on validation set
y_val_pred_rf = clf_rf.predict(X_val)
y_val_proba_rf = clf_rf.predict_proba(X_val)[:, 1]

# Calculate accuracy and F1 score for training set
accuracy_train_rf = accuracy_score(y_train_resampled, y_train_pred_rf)
f1_train_rf = f1_score(y_train_resampled, y_train_pred_rf)

# Calculate accuracy and F1 score for validation set
accuracy_val_rf = accuracy_score(y_val, y_val_pred_rf)
f1_val_rf = f1_score(y_val, y_val_pred_rf)

# Calculate ROC AUC score for training set
roc_auc_train_rf = roc_auc_score(y_train_resampled, y_train_proba_rf)

# Calculate ROC AUC score for validation set
roc_auc_val_rf = roc_auc_score(y_val, y_val_proba_rf)

# Print the metrics for both the training and validation sets
print(f"Random Forest Classifier\n")

print(f"Random Forest Training Accuracy: {accuracy_train_rf:.4f}")
print(f"Random Forest Training F1 Score: {f1_train_rf:.4f}")
print(f"Random Forest Training ROC AUC Score: {roc_auc_train_rf:.4f}")
print(f"Random Forest Validation Accuracy: {accuracy_val_rf:.4f}")
print(f"Random Forest Validation F1 Score: {f1_val_rf:.4f}")
print(f"Random Forest Validation ROC AUC Score: {roc_auc_val_rf:.4f}")

# Plot ROC curve for the validation set
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_val, y_val_proba_rf)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Validation ROC curve (AUC = {roc_auc_val_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Random Forest')
plt.legend(loc="lower right")
plt.show()

"""hyper RF"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

rf = RandomForestClassifier(random_state=42)

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10,15,20],
    'min_samples_split': [10,20],
    'min_samples_leaf': [5,10]
}
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

grid_search_rf.fit(X_train_resampled, y_train_resampled)

print(f"Random Forest Classifier HPT\n")

print("Best Parameters:", grid_search_rf.best_params_)

y_train_pred_gs_rf = grid_search_rf.best_estimator_.predict(X_train_resampled)
y_val_pred_gs_rf = grid_search_rf.best_estimator_.predict(X_val)

accuracy_train_gs_rf = accuracy_score(y_train_resampled, y_train_pred_gs_rf)
f1_train_gs_rf = f1_score(y_train_resampled, y_train_pred_gs_rf)

accuracy_val_gs_rf = accuracy_score(y_val, y_val_pred_gs_rf)
f1_val_gs_rf = f1_score(y_val, y_val_pred_gs_rf)

roc_auc_train_gs_rf = roc_auc_score(y_train_resampled, grid_search_rf.best_estimator_.predict_proba(X_train_resampled)[:, 1])

roc_auc_val_gs_rf = roc_auc_score(y_val, grid_search_rf.best_estimator_.predict_proba(X_val)[:, 1])

print(f"Training Accuracy: {accuracy_train_gs_rf:.4f}")
print(f"Training F1 Score: {f1_train_gs_rf:.4f}")
print(f"Training ROC AUC Score: {roc_auc_train_gs_rf:.4f}")
print(f"Validation Accuracy: {accuracy_val_gs_rf:.4f}")
print(f"Validation F1 Score: {f1_val_gs_rf:.4f}")
print(f"Validation ROC AUC Score: {roc_auc_val_gs_rf:.4f}")

"""# Support Vector Machine (SVM)"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Train SVM model
clf_svm = SVC(probability=True, random_state=42)
clf_svm.fit(X_train_resampled, y_train_resampled)

# Predict on training set
y_train_pred_svm = clf_svm.predict(X_train_resampled)
y_train_proba_svm = clf_svm.predict_proba(X_train_resampled)[:, 1]

# Predict on validation set
y_val_pred_svm = clf_svm.predict(X_val)
y_val_proba_svm = clf_svm.predict_proba(X_val)[:, 1]

# Calculate accuracy and F1 score for training set
accuracy_train_svm = accuracy_score(y_train_resampled, y_train_pred_svm)
f1_train_svm = f1_score(y_train_resampled, y_train_pred_svm)

# Calculate accuracy and F1 score for validation set
accuracy_val_svm = accuracy_score(y_val, y_val_pred_svm)
f1_val_svm = f1_score(y_val, y_val_pred_svm)

# Calculate ROC AUC score for training set
roc_auc_train_svm = roc_auc_score(y_train_resampled, y_train_proba_svm)

# Calculate ROC AUC score for validation set
roc_auc_val_svm = roc_auc_score(y_val, y_val_proba_svm)
print(f"Support Vector Machine\n")

# Print the metrics for both the training and validation sets
print(f"SVM Training Accuracy: {accuracy_train_svm:.4f}")
print(f"SVM Training F1 Score: {f1_train_svm:.4f}")
print(f"SVM Training ROC AUC Score: {roc_auc_train_svm:.4f}")
print(f"SVM Validation Accuracy: {accuracy_val_svm:.4f}")
print(f"SVM Validation F1 Score: {f1_val_svm:.4f}")
print(f"SVM Validation ROC AUC Score: {roc_auc_val_svm:.4f}")

# Plot ROC curve for the validation set
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_val, y_val_proba_svm)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, color='darkorange', lw=2, label=f'Validation ROC curve (AUC = {roc_auc_val_svm:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - SVM')
plt.legend(loc="lower right")
plt.show()

"""HPT SVM"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

svm = SVC(probability=True, random_state=42)

param_grid_svm = {
    'C': [1,10,40,50,60,80,100],
    'kernel': ['linear','rbf']
}

grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

grid_search_svm.fit(X_train_resampled, y_train_resampled)

print(f"Support Vector Machine Hyper Parameter Tuning\n")

print("Best Parameters:", grid_search_svm.best_params_)

y_train_pred_gs_svm = grid_search_svm.best_estimator_.predict(X_train_resampled)
y_val_pred_gs_svm = grid_search_svm.best_estimator_.predict(X_val)

accuracy_train_gs_svm = accuracy_score(y_train_resampled, y_train_pred_gs_svm)
f1_train_gs_svm = f1_score(y_train_resampled, y_train_pred_gs_svm)

accuracy_val_gs_svm = accuracy_score(y_val, y_val_pred_gs_svm)
f1_val_gs_svm = f1_score(y_val, y_val_pred_gs_svm)

roc_auc_train_gs_svm = roc_auc_score(y_train_resampled, grid_search_svm.best_estimator_.predict_proba(X_train_resampled)[:, 1])

roc_auc_val_gs_svm = roc_auc_score(y_val, grid_search_svm.best_estimator_.predict_proba(X_val)[:, 1])

print(f"Training Accuracy: {accuracy_train_gs_svm:.4f}")
print(f"Training F1 Score: {f1_train_gs_svm:.4f}")
print(f"Training ROC AUC Score: {roc_auc_train_gs_svm:.4f}")
print(f"Validation Accuracy: {accuracy_val_gs_svm:.4f}")
print(f"Validation F1 Score: {f1_val_gs_svm:.4f}")
print(f"Validation ROC AUC Score: {roc_auc_val_gs_svm:.4f}")

"""# Naive Bayers"""

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

clf_nb = GaussianNB()
clf_nb.fit(X_train_resampled, y_train_resampled)

y_train_pred_nb = clf_nb.predict(X_train_resampled)
y_train_proba_nb = clf_nb.predict_proba(X_train_resampled)[:, 1]

y_val_pred_nb = clf_nb.predict(X_val)
y_val_proba_nb = clf_nb.predict_proba(X_val)[:, 1]

accuracy_train_nb = accuracy_score(y_train_resampled, y_train_pred_nb)
f1_train_nb = f1_score(y_train_resampled, y_train_pred_nb)

accuracy_val_nb = accuracy_score(y_val, y_val_pred_nb)
f1_val_nb = f1_score(y_val, y_val_pred_nb)

roc_auc_train_nb = roc_auc_score(y_train_resampled, y_train_proba_nb)

roc_auc_val_nb = roc_auc_score(y_val, y_val_proba_nb)
print(f"Naive Bayes\n")

print(f"Naive Bayes Training Accuracy: {accuracy_train_nb:.4f}")
print(f"Naive Bayes Training F1 Score: {f1_train_nb:.4f}")
print(f"Naive Bayes Training ROC AUC Score: {roc_auc_train_nb:.4f}")
print(f"Naive Bayes Validation Accuracy: {accuracy_val_nb:.4f}")
print(f"Naive Bayes Validation F1 Score: {f1_val_nb:.4f}")
print(f"Naive Bayes Validation ROC AUC Score: {roc_auc_val_nb:.4f}")

fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_val, y_val_proba_nb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, color='darkorange', lw=2, label=f'Validation ROC curve (AUC = {roc_auc_val_nb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Naive Bayes')
plt.legend(loc="lower right")
plt.show()

"""HPT NB"""

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

gnb = GaussianNB()

param_grid_gnb = {
    'var_smoothing': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01]
}

grid_search_gnb = GridSearchCV(estimator=gnb, param_grid=param_grid_gnb, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

grid_search_gnb.fit(X_train_resampled, y_train_resampled)

print(f"Naive Bayers Hyper Parameter Tuning\n")

print("Best Parameters:", grid_search_gnb.best_params_)

y_train_pred_gs_gnb = grid_search_gnb.best_estimator_.predict(X_train_resampled)
y_val_pred_gs_gnb = grid_search_gnb.best_estimator_.predict(X_val)

accuracy_train_gs_gnb = accuracy_score(y_train_resampled, y_train_pred_gs_gnb)
f1_train_gs_gnb = f1_score(y_train_resampled, y_train_pred_gs_gnb)

accuracy_val_gs_gnb = accuracy_score(y_val, y_val_pred_gs_gnb)
f1_val_gs_gnb = f1_score(y_val, y_val_pred_gs_gnb)

roc_auc_train_gs_gnb = roc_auc_score(y_train_resampled, grid_search_gnb.best_estimator_.predict_proba(X_train_resampled)[:, 1])
roc_auc_val_gs_gnb = roc_auc_score(y_val, grid_search_gnb.best_estimator_.predict_proba(X_val)[:, 1])

print(f"Training Accuracy: {accuracy_train_gs_gnb:.4f}")
print(f"Training F1 Score: {f1_train_gs_gnb:.4f}")
print(f"Training ROC AUC Score: {roc_auc_train_gs_gnb:.4f}")
print(f"Validation Accuracy: {accuracy_val_gs_gnb:.4f}")
print(f"Validation F1 Score: {f1_val_gs_gnb:.4f}")
print(f"Validation ROC AUC Score: {roc_auc_val_gs_gnb:.4f}")

"""# XGBoost"""

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
clf_xgb = XGBClassifier(
    n_estimators=5,  
    max_depth=2,  
    learning_rate=1,  
    objective='binary:logistic',  
    random_state=42  
)
clf_xgb.fit(X_train_resampled, y_train_resampled)

y_train_pred_xgb = clf_xgb.predict(X_train_resampled)
y_train_proba_xgb = clf_xgb.predict_proba(X_train_resampled)[:, 1]

y_val_pred_xgb = clf_xgb.predict(X_val)
y_val_proba_xgb = clf_xgb.predict_proba(X_val)[:, 1]

accuracy_train_xgb = accuracy_score(y_train_resampled, y_train_pred_xgb)
f1_train_xgb = f1_score(y_train_resampled, y_train_pred_xgb)

accuracy_val_xgb = accuracy_score(y_val, y_val_pred_xgb)
f1_val_xgb = f1_score(y_val, y_val_pred_xgb)

roc_auc_train_xgb = roc_auc_score(y_train_resampled, y_train_proba_xgb)

roc_auc_val_xgb = roc_auc_score(y_val, y_val_proba_xgb)
print(f"XGBoost Classifier\n")

print(f"XGBoost Training Accuracy: {accuracy_train_xgb:.4f}")
print(f"XGBoost Training F1 Score: {f1_train_xgb:.4f}")
print(f"XGBoost Training ROC AUC Score: {roc_auc_train_xgb:.4f}")
print(f"XGBoost Validation Accuracy: {accuracy_val_xgb:.4f}")
print(f"XGBoost Validation F1 Score: {f1_val_xgb:.4f}")
print(f"XGBoost Validation ROC AUC Score: {roc_auc_val_xgb:.4f}")

# Plot ROC curve for the validation set
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_val, y_val_proba_xgb)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label=f'Validation ROC curve (AUC = {roc_auc_val_xgb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - XGBoost')
plt.legend(loc="lower right")
plt.show()

"""HPT XGB"""

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

param_grid_xgb = {
    'n_estimators': [1,3,5,10,30,50,90,100], 
    'learning_rate': [0.01, 0.1, 0.2,0.5,0.8], 
    'max_depth':[1,2,3,4,5,6,7,8] 
}

grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid_xgb, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

grid_search_xgb.fit(X_train_resampled, y_train_resampled)

print(f"XGBoost Classifier Hyper Parameter Tuning\n")

print("Best Parameters:", grid_search_xgb.best_params_)

y_train_pred_gs_xgb = grid_search_xgb.best_estimator_.predict(X_train_resampled)
y_val_pred_gs_xgb = grid_search_xgb.best_estimator_.predict(X_val)

accuracy_train_gs_xgb = accuracy_score(y_train_resampled, y_train_pred_gs_xgb)
f1_train_gs_xgb = f1_score(y_train_resampled, y_train_pred_gs_xgb)

accuracy_val_gs_xgb = accuracy_score(y_val, y_val_pred_gs_xgb)
f1_val_gs_xgb = f1_score(y_val, y_val_pred_gs_xgb)

roc_auc_train_gs_xgb = roc_auc_score(y_train_resampled, grid_search_xgb.best_estimator_.predict_proba(X_train_resampled)[:, 1])

roc_auc_val_gs_xgb = roc_auc_score(y_val, grid_search_xgb.best_estimator_.predict_proba(X_val)[:, 1])

print(f"Training Accuracy: {accuracy_train_gs_xgb:.4f}")
print(f"Training F1 Score: {f1_train_gs_xgb:.4f}")
print(f"Training ROC AUC Score: {roc_auc_train_gs_xgb:.4f}")
print(f"Validation Accuracy: {accuracy_val_gs_xgb:.4f}")
print(f"Validation F1 Score: {f1_val_gs_xgb:.4f}")
print(f"Validation ROC AUC Score: {roc_auc_val_gs_xgb:.4f}")

"""# Gradient Boost"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

clf_gb = GradientBoostingClassifier(random_state=0,n_estimators=5)
clf_gb.fit(X_train_resampled, y_train_resampled)

y_train_pred_gb = clf_gb.predict(X_train_resampled)
y_train_proba_gb = clf_gb.predict_proba(X_train_resampled)[:, 1]

y_val_pred_gb = clf_gb.predict(X_val)
y_val_proba_gb = clf_gb.predict_proba(X_val)[:, 1]

accuracy_train_gb = accuracy_score(y_train_resampled, y_train_pred_gb)
f1_train_gb = f1_score(y_train_resampled, y_train_pred_gb)

accuracy_val_gb = accuracy_score(y_val, y_val_pred_gb)
f1_val_gb = f1_score(y_val, y_val_pred_gb)

roc_auc_train_gb = roc_auc_score(y_train_resampled, y_train_proba_gb)

roc_auc_val_gb = roc_auc_score(y_val, y_val_proba_gb)
print(f"Gradient Boost\n")

print(f"Gradient Boosting Training Accuracy: {accuracy_train_gb:.4f}")
print(f"Gradient Boosting Training F1 Score: {f1_train_gb:.4f}")
print(f"Gradient Boosting Training ROC AUC Score: {roc_auc_train_gb:.4f}")
print(f"Gradient Boosting Validation Accuracy: {accuracy_val_gb:.4f}")
print(f"Gradient Boosting Validation F1 Score: {f1_val_gb:.4f}")
print(f"Gradient Boosting Validation ROC AUC Score: {roc_auc_val_gb:.4f}")

fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_val, y_val_proba_gb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_gb, tpr_gb, color='darkorange', lw=2, label=f'Validation ROC curve (AUC = {roc_auc_val_gb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Gradient Boosting')
plt.legend(loc="lower right")
plt.show()

"""HPT GB"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

gb = GradientBoostingClassifier(random_state=42)

param_grid_gb = {
    'n_estimators': [3,4,5,10], 
    'learning_rate': [0.01,0.1, 0.2,0.5], 
    'max_depth': [1,2,3,4,5,10],  
    'min_samples_split': [2,3,4], 
    'min_samples_leaf': [1,2,3],  
    'subsample': [0.8, 1.0],  
}

grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb,
                               cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

grid_search_gb.fit(X_train_resampled, y_train_resampled)

print(f"Gradient Boost Hyper Parameter Tuning\n")

print("Best Parameters:", grid_search_gb.best_params_)

y_train_pred_gs_gb = grid_search_gb.best_estimator_.predict(X_train_resampled)
y_val_pred_gs_gb = grid_search_gb.best_estimator_.predict(X_val)

accuracy_train_gs_gb = accuracy_score(y_train_resampled, y_train_pred_gs_gb)
f1_train_gs_gb = f1_score(y_train_resampled, y_train_pred_gs_gb)

accuracy_val_gs_gb = accuracy_score(y_val, y_val_pred_gs_gb)
f1_val_gs_gb = f1_score(y_val, y_val_pred_gs_gb)

roc_auc_train_gs_gb = roc_auc_score(y_train_resampled, 
        grid_search_gb.best_estimator_.predict_proba(X_train_resampled)[:, 1])

roc_auc_val_gs_gb = roc_auc_score(y_val, 
                    grid_search_gb.best_estimator_.predict_proba(X_val)[:, 1])

print(f"Training Accuracy: {accuracy_train_gs_gb:.4f}")
print(f"Training F1 Score: {f1_train_gs_gb:.4f}")
print(f"Training ROC AUC Score: {roc_auc_train_gs_gb:.4f}")
print(f"Validation Accuracy: {accuracy_val_gs_gb:.4f}")
print(f"Validation F1 Score: {f1_val_gs_gb:.4f}")
print(f"Validation ROC AUC Score: {roc_auc_val_gs_gb:.4f}")

"""# CatBoost"""

pip install catboost

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# clf_cb = CatBoostClassifier(random_state=42, auto_class_weights='Balanced', verbose=False)
clf_cb = CatBoostClassifier(iterations=20, 
                           depth=4,  
                           learning_rate=0.1, 
                           loss_function='Logloss', 
                           random_seed=42)  
clf_cb.fit(X_train_resampled, y_train_resampled)

# Predict on training set
y_train_pred_cb = clf_cb.predict(X_train_resampled)
y_train_proba_cb = clf_cb.predict_proba(X_train_resampled)[:, 1]

# Predict on validation set
y_val_pred_cb = clf_cb.predict(X_val)
y_val_proba_cb = clf_cb.predict_proba(X_val)[:, 1]

# Calculate accuracy and F1 score for training set
accuracy_train_cb = accuracy_score(y_train_resampled, y_train_pred_cb)
f1_train_cb = f1_score(y_train_resampled, y_train_pred_cb)

# Calculate accuracy and F1 score for validation set
accuracy_val_cb = accuracy_score(y_val, y_val_pred_cb)
f1_val_cb = f1_score(y_val, y_val_pred_cb)

# Calculate ROC AUC score for training set
roc_auc_train_cb = roc_auc_score(y_train_resampled, y_train_proba_cb)

# Calculate ROC AUC score for validation set
roc_auc_val_cb = roc_auc_score(y_val, y_val_proba_cb)

print(f"CatBoost Training Accuracy\n")
print(f"CatBoost Training Accuracy: {accuracy_train_cb:.4f}")
print(f"CatBoost Training F1 Score: {f1_train_cb:.4f}")
print(f"CatBoost Training ROC AUC Score: {roc_auc_train_cb:.4f}")
print(f"CatBoost Validation Accuracy: {accuracy_val_cb:.4f}")
print(f"CatBoost Validation F1 Score: {f1_val_cb:.4f}")
print(f"CatBoost Validation ROC AUC Score: {roc_auc_val_cb:.4f}")

fpr_cb, tpr_cb, thresholds_cb = roc_curve(y_val, y_val_proba_cb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_cb, tpr_cb, color='darkorange', lw=2, label=f'Validation ROC curve (AUC = {roc_auc_val_cb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - CatBoost')
plt.legend(loc="lower right")
plt.show()

"""HPT CB"""

from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

cb = CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced') 

param_grid_cb = {
    'iterations': [10,20,30,40,50,100,200,300],  
    'learning_rate': [0.01,0.1,0.2,0.3],  
    'depth': [2,3,4,5,10],  
    'l2_leaf_reg': [1,2,3],  
    'loss_function':['Logloss']
}

grid_search_cb = GridSearchCV(estimator=cb, param_grid=param_grid_cb, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

grid_search_cb.fit(X_train_resampled, y_train_resampled)

print(f"CatBoost Training Accuracy Hyper Parameter Tuning\n")

print("Best Parameters:", grid_search_cb.best_params_)

y_train_pred_gs_cb = grid_search_cb.best_estimator_.predict(X_train_resampled)
y_val_pred_gs_cb = grid_search_cb.best_estimator_.predict(X_val)

accuracy_train_gs_cb = accuracy_score(y_train_resampled, y_train_pred_gs_cb)
f1_train_gs_cb = f1_score(y_train_resampled, y_train_pred_gs_cb)

accuracy_val_gs_cb = accuracy_score(y_val, y_val_pred_gs_cb)
f1_val_gs_cb = f1_score(y_val, y_val_pred_gs_cb)

roc_auc_train_gs_cb = roc_auc_score(y_train_resampled, grid_search_cb.best_estimator_.predict_proba(X_train_resampled)[:, 1])

roc_auc_val_gs_cb = roc_auc_score(y_val, grid_search_cb.best_estimator_.predict_proba(X_val)[:, 1])

print(f"Training Accuracy: {accuracy_train_gs_cb:.4f}")
print(f"Training F1 Score: {f1_train_gs_cb:.4f}")
print(f"Training ROC AUC Score: {roc_auc_train_gs_cb:.4f}")
print(f"Validation Accuracy: {accuracy_val_gs_cb:.4f}")
print(f"Validation F1 Score: {f1_val_gs_cb:.4f}")
print(f"Validation ROC AUC Score: {roc_auc_val_gs_cb:.4f}")

"""# AdaBoost Classifier"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


clf_ab = AdaBoostClassifier(n_estimators=50, 
                  learning_rate=0.1)
clf_ab.fit(X_train_resampled, y_train_resampled)

# Predict on training set
y_train_pred_ab = clf_ab.predict(X_train_resampled)
# For ROC AUC Score, we need probability estimates of the positive class
y_train_proba_ab = clf_ab.predict_proba(X_train_resampled)[:, 1]

# Predict on validation set
y_val_pred_ab = clf_ab.predict(X_val)
# For ROC AUC Score, we need probability estimates of the positive class
y_val_proba_ab = clf_ab.predict_proba(X_val)[:, 1]

# Calculate accuracy and F1 score for training set
accuracy_train_ab = accuracy_score(y_train_resampled, y_train_pred_ab)
f1_train_ab = f1_score(y_train_resampled, y_train_pred_ab)

# Calculate accuracy and F1 score for validation set
accuracy_val_ab = accuracy_score(y_val, y_val_pred_ab)
f1_val_ab = f1_score(y_val, y_val_pred_ab)

# Calculate ROC AUC score for training set
roc_auc_train_ab = roc_auc_score(y_train_resampled, y_train_proba_ab)

# Calculate ROC AUC score for validation set
roc_auc_val_ab = roc_auc_score(y_val, y_val_proba_ab)

# Print the metrics for both the training and validation sets
print(f"AdaBoost Classifier\n")

print(f"AdaBoost Training Accuracy: {accuracy_train_ab:.4f}")
print(f"AdaBoost Training F1 Score: {f1_train_ab:.4f}")
print(f"AdaBoost Training ROC AUC Score: {roc_auc_train_ab:.4f}")
print(f"AdaBoost Validation Accuracy: {accuracy_val_ab:.4f}")
print(f"AdaBoost Validation F1 Score: {f1_val_ab:.4f}")
print(f"AdaBoost Validation ROC AUC Score: {roc_auc_val_ab:.4f}")

# Plot ROC curve for the validation set
fpr_ab, tpr_ab, thresholds_ab = roc_curve(y_val, y_val_proba_ab)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_ab, tpr_ab, color='darkorange', lw=2, label=f'Validation ROC curve (AUC = {roc_auc_val_ab:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - AdaBoost')
plt.legend(loc="lower right")
plt.show()

"""HPT ADA"""

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

base_estimator = DecisionTreeClassifier(max_depth=1)
adb = AdaBoostClassifier(base_estimator=base_estimator, random_state=42)

param_grid_adb = {
    'n_estimators': [50, 60,80, 100, 200], 
    'learning_rate': [0.01, 0.1, 1],  
    'base_estimator__max_depth': [1, 2, 3], 
}

grid_search_adb = GridSearchCV(estimator=adb, param_grid=param_grid_adb, cv=5,
                                scoring='accuracy', verbose=1, n_jobs=-1)

grid_search_adb.fit(X_train_resampled, y_train_resampled)

print(f"AdaBoost Classifier Hyper Parameter Tuning\n")

print("Best Parameters:", grid_search_adb.best_params_)

y_train_pred_gs_adb = grid_search_adb.best_estimator_.predict(X_train_resampled)
y_val_pred_gs_adb = grid_search_adb.best_estimator_.predict(X_val)

accuracy_train_gs_adb = accuracy_score(y_train_resampled, y_train_pred_gs_adb)
f1_train_gs_adb = f1_score(y_train_resampled, y_train_pred_gs_adb)

accuracy_val_gs_adb = accuracy_score(y_val, y_val_pred_gs_adb)
f1_val_gs_adb = f1_score(y_val, y_val_pred_gs_adb)

roc_auc_train_gs_adb = roc_auc_score(y_train_resampled, 
        grid_search_adb.best_estimator_.predict_proba(X_train_resampled)[:, 1])

roc_auc_val_gs_adb = roc_auc_score(y_val, 
                grid_search_adb.best_estimator_.predict_proba(X_val)[:, 1])

print(f"Training Accuracy: {accuracy_train_gs_adb:.4f}")
print(f"Training F1 Score: {f1_train_gs_adb:.4f}")
print(f"Training ROC AUC Score: {roc_auc_train_gs_adb:.4f}")
print(f"Validation Accuracy: {accuracy_val_gs_adb:.4f}")
print(f"Validation F1 Score: {f1_val_gs_adb:.4f}")
print(f"Validation ROC AUC Score: {roc_auc_val_gs_adb:.4f}")