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



"""# SVM_SMOTE"""

# Initialize SVM-SMOTE
svm_smote = SVMSMOTE(sampling_strategy='auto', random_state=42, n_jobs=-1)

X_train_resampled, y_train_resampled = svm_smote.fit_resample(X_train, y_train)

# X_train = X_train_resampled
# Y_train = Y_train_resampled

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
print(f"SVM-SMOTE LR Results")
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
print(f"SVM-SMOTE DT Results")
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

"""# Random Forest Trees"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

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

"""# Support Vector Machine (SVM)"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

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

"""# Naive Bayers"""

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Assuming X_train_resampled, y_train_resampled, X_val, and y_val are already defined

# Train Naive Bayes model
clf_nb = GaussianNB()
clf_nb.fit(X_train_resampled, y_train_resampled)

# Predict on training set
y_train_pred_nb = clf_nb.predict(X_train_resampled)
# For ROC AUC Score, we need probability estimates of the positive class
y_train_proba_nb = clf_nb.predict_proba(X_train_resampled)[:, 1]

# Predict on validation set
y_val_pred_nb = clf_nb.predict(X_val)
# For ROC AUC Score, we need probability estimates of the positive class
y_val_proba_nb = clf_nb.predict_proba(X_val)[:, 1]

# Calculate accuracy and F1 score for training set
accuracy_train_nb = accuracy_score(y_train_resampled, y_train_pred_nb)
f1_train_nb = f1_score(y_train_resampled, y_train_pred_nb)

# Calculate accuracy and F1 score for validation set
accuracy_val_nb = accuracy_score(y_val, y_val_pred_nb)
f1_val_nb = f1_score(y_val, y_val_pred_nb)

# Calculate ROC AUC score for training set
roc_auc_train_nb = roc_auc_score(y_train_resampled, y_train_proba_nb)

# Calculate ROC AUC score for validation set
roc_auc_val_nb = roc_auc_score(y_val, y_val_proba_nb)

# Print the metrics for both the training and validation sets
print(f"Naive Bayes Training Accuracy: {accuracy_train_nb:.4f}")
print(f"Naive Bayes Training F1 Score: {f1_train_nb:.4f}")
print(f"Naive Bayes Training ROC AUC Score: {roc_auc_train_nb:.4f}")
print(f"Naive Bayes Validation Accuracy: {accuracy_val_nb:.4f}")
print(f"Naive Bayes Validation F1 Score: {f1_val_nb:.4f}")
print(f"Naive Bayes Validation ROC AUC Score: {roc_auc_val_nb:.4f}")

# Plot ROC curve for the validation set
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_val, y_val_proba_nb)

# Plotting the ROC curve
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

"""# XGBoost"""

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

clf_xgb = xgb.XGBClassifier(
    n_estimators=5,
    max_depth=2,  
    learning_rate=1, 
    objective='binary:logistic', 
    random_state=42  
clf_xgb.fit(X_train_resampled, y_train_resampled)

# Predict on training set
y_train_pred_xgb = clf_xgb.predict(X_train_resampled)
y_train_proba_xgb = clf_xgb.predict_proba(X_train_resampled)[:, 1]

# Predict on validation set
y_val_pred_xgb = clf_xgb.predict(X_val)
# For ROC AUC Score, we need probability estimates of the positive class
y_val_proba_xgb = clf_xgb.predict_proba(X_val)[:, 1]

# Calculate accuracy and F1 score for training set
accuracy_train_xgb = accuracy_score(y_train_resampled, y_train_pred_xgb)
f1_train_xgb = f1_score(y_train_resampled, y_train_pred_xgb)

# Calculate accuracy and F1 score for validation set
accuracy_val_xgb = accuracy_score(y_val, y_val_pred_xgb)
f1_val_xgb = f1_score(y_val, y_val_pred_xgb)

# Calculate ROC AUC score for training set
roc_auc_train_xgb = roc_auc_score(y_train_resampled, y_train_proba_xgb)

# Calculate ROC AUC score for validation set
roc_auc_val_xgb = roc_auc_score(y_val, y_val_proba_xgb)

# Print the metrics for both the training and validation sets
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

"""# Gradient Boost"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Train Gradient Boosting model
clf_gb = GradientBoostingClassifier(random_state=0,n_estimators=5)
clf_gb.fit(X_train_resampled, y_train_resampled)

# Predict on training set
y_train_pred_gb = clf_gb.predict(X_train_resampled)
# For ROC AUC Score, we need decision function or probability estimates of the positive class
y_train_proba_gb = clf_gb.predict_proba(X_train_resampled)[:, 1]

# Predict on validation set
y_val_pred_gb = clf_gb.predict(X_val)
y_val_proba_gb = clf_gb.predict_proba(X_val)[:, 1]

# Calculate accuracy and F1 score for training set
accuracy_train_gb = accuracy_score(y_train_resampled, y_train_pred_gb)
f1_train_gb = f1_score(y_train_resampled, y_train_pred_gb)

# Calculate accuracy and F1 score for validation set
accuracy_val_gb = accuracy_score(y_val, y_val_pred_gb)
f1_val_gb = f1_score(y_val, y_val_pred_gb)

# Calculate ROC AUC score for training set
roc_auc_train_gb = roc_auc_score(y_train_resampled, y_train_proba_gb)

# Calculate ROC AUC score for validation set
roc_auc_val_gb = roc_auc_score(y_val, y_val_proba_gb)

# Print the metrics for both the training and validation sets
print(f"Gradient Boosting Training Accuracy: {accuracy_train_gb:.4f}")
print(f"Gradient Boosting Training F1 Score: {f1_train_gb:.4f}")
print(f"Gradient Boosting Training ROC AUC Score: {roc_auc_train_gb:.4f}")
print(f"Gradient Boosting Validation Accuracy: {accuracy_val_gb:.4f}")
print(f"Gradient Boosting Validation F1 Score: {f1_val_gb:.4f}")
print(f"Gradient Boosting Validation ROC AUC Score: {roc_auc_val_gb:.4f}")

# Plot ROC curve for the validation set
fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_val, y_val_proba_gb)

# Plotting the ROC curve
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

"""# CatBoost"""

pip install catboost

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

clf_cb = CatBoostClassifier(iterations=20, 
                           depth=4, 
                           learning_rate=0.1, 
                           loss_function='Logloss', 
                           random_seed=42)  
clf_cb.fit(X_train_resampled, y_train_resampled)

# Predict on training set
y_train_pred_cb = clf_cb.predict(X_train_resampled)
# For ROC AUC Score, we need probability estimates of the positive class
y_train_proba_cb = clf_cb.predict_proba(X_train_resampled)[:, 1]

# Predict on validation set
y_val_pred_cb = clf_cb.predict(X_val)
# For ROC AUC Score, we need probability estimates of the positive class
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

# Print the metrics for both the training and validation sets
print(f"CatBoost Training Accuracy: {accuracy_train_cb:.4f}")
print(f"CatBoost Training F1 Score: {f1_train_cb:.4f}")
print(f"CatBoost Training ROC AUC Score: {roc_auc_train_cb:.4f}")
print(f"CatBoost Validation Accuracy: {accuracy_val_cb:.4f}")
print(f"CatBoost Validation F1 Score: {f1_val_cb:.4f}")
print(f"CatBoost Validation ROC AUC Score: {roc_auc_val_cb:.4f}")

# Plot ROC curve for the validation set
fpr_cb, tpr_cb, thresholds_cb = roc_curve(y_val, y_val_proba_cb)

# Plotting the ROC curve
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

"""# AdaBoost Classifier"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

clf_ab = AdaBoostClassifier(n_estimators=100, 
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
print(f"AdaBoost Training Accuracy: {accuracy_train_ab:.4f}")
print(f"AdaBoost Training F1 Score: {f1_train_ab:.4f}")
print(f"AdaBoost Training ROC AUC Score: {roc_auc_train_ab:.4f}")
print(f"AdaBoost Validation Accuracy: {accuracy_val_ab:.4f}")
print(f"AdaBoost Validation F1 Score: {f1_val_cb:.4f}")
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