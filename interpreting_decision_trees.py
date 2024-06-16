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
from sklearn.pipeline import make_pipeline

pip install graphviz

df=pd.read_csv("drive/MyDrive/4th_Year_Research/Implementation/Dataset/Pre-Processed-Dataset.csv")
df.shape

"""# Splitting data into X and Y"""

X = df.drop(columns=['Target'], axis=1)
Y = df['Target']

X.shape

X.head()

Y.head()

"""

---

# Splitting Data Into Training and Testing

---

"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

print(X.shape, X_train.shape, X_test.shape)

"""# Data Sampling Using ADASYN"""

from imblearn.over_sampling import ADASYN

# Initialize ADASYN
adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=5)

# Apply ADASYN to the training data
X_train_resampled, Y_train_resampled = adasyn.fit_resample(X_train, Y_train)

X_train, Y_train = X_train_resampled, Y_train_resampled

count_of_ones = (Y_train == 1).sum()
count_of_zeros = (Y_train == 0).sum()
print(f"Number of rows with Target = 0: {count_of_zeros}")
print(f"Number of rows with Target = 1: {count_of_ones }")

"""# **Logistic Regression**

## Cross Validation - LR (5- folds)
"""

from sklearn.model_selection import cross_val_score
clf2 = LogisticRegression(max_iter=1000)
clf2.fit(X_train,Y_train)
print("After Cross Validation\n")
# Accuracy
accuracy_scores = cross_val_score(clf2, X, Y, cv=5, scoring='accuracy')
mean_accuracy = np.mean(accuracy_scores)
print("Mean Accuracy:", mean_accuracy)

# Precision
precision_scores = cross_val_score(clf2, X, Y, cv=5, scoring='precision')
mean_precision = np.mean(precision_scores)
print("Mean Precision:", mean_precision)

# Recall
recall_scores = cross_val_score(clf2, X, Y, cv=5, scoring='recall')
mean_recall = np.mean(recall_scores)
print("Mean Recall:", mean_recall)

# F1-Score
f1_scores = cross_val_score(clf2, X, Y, cv=5, scoring='f1')
mean_f1 = np.mean(f1_scores)
print("Mean F1-Score:", mean_f1)

# ROC-AUC
roc_auc_scores = cross_val_score(clf2, X, Y, cv=5, scoring='roc_auc')
mean_roc_auc = np.mean(roc_auc_scores)
print("Mean ROC-AUC:", mean_roc_auc)

"""## Lime for Logistic Regression"""

# Create the LIME explainer
#test comment
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                   feature_names=X.columns.tolist(),
                                                   class_names=['Dropout', 'Graduate'],
                                                   discretize_continuous=True)

predict_fn = clf2.predict_proba

sample_index = 9  
exp = explainer.explain_instance(X_test.values[sample_index],
                                 predict_fn,
                                 num_features=6)

# Show the explanation
exp.show_in_notebook() 
exp.as_list()

"""# SHAP For Logistic"""

import shap
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

model = CatBoostClassifier(iterations=70, depth=6, learning_rate=0.1, verbose=0)
model.fit(X_train, Y_train)

target_prediction = model.predict(X_test)
accuracy = accuracy_score(Y_test, target_prediction)
print(f'Accuracy: {accuracy:.6f}')

explainer = shap.Explainer(model)

shap_values = explainer(X_test)

"""# **Decision Trees**"""

from sklearn.tree import DecisionTreeClassifier, export_graphviz

dt_clf_unpruned = DecisionTreeClassifier(random_state=42)
dt_clf_unpruned.fit(X_train, Y_train)

target_prediction = dt_clf_unpruned.predict(X_test)

unpruned_accuracy = accuracy_score(Y_test,target_prediction)

print(f"Accuracy of unpruned decision tree: {unpruned_accuracy:.4f}")

path = dt_clf_unpruned.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

ccp_alphas = ccp_alphas[:-1]

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, Y_train)
    clfs.append(clf)

train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
ax.legend()
plt.show()

# Select the model with the highest test accuracy
optimal_alpha = ccp_alphas[np.argmax(test_scores)]
optimal_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=optimal_alpha)
optimal_clf.fit(X_train, Y_train)
target_prediction_pruned = optimal_clf.predict(X_test)

# Evaluate the pruned classifier on the test set
optimal_accuracy = accuracy_score(Y_test, target_prediction_pruned)

print(f"Optimal ccp_alpha: {optimal_alpha}")
print(f"Accuracy of pruned decision tree: {optimal_accuracy:.4f}")

import graphviz

dt_clf = DecisionTreeClassifier(random_state=42,ccp_alpha=optimal_alpha)

dt_clf.fit(X_train, Y_train)

target_prediction = clf.predict(X_test)
# print(target_prediction)

dot_data = export_graphviz(dt_clf, out_file=None,
                           feature_names=X_train.columns,
                           class_names=['Not Dropout', 'Dropout'],
                           filled=True, rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("decision_tree")  

graph.render(filename='decision_tree', format='jpeg', cleanup=True)

data_accuracy = accuracy_score(Y_test, target_prediction_pruned)
print("Accuracy:", data_accuracy)

"""## Cross Validation - (5- folds)"""

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(random_state=0)

print("After Cross Validation\n")
accuracy_scores = cross_val_score(DTC, X, Y, cv=5, scoring='accuracy')
mean_accuracy = np.mean(accuracy_scores)
print("Mean Accuracy:", mean_accuracy)

precision_scores = cross_val_score(DTC, X, Y, cv=5, scoring='precision')
mean_precision = np.mean(precision_scores)
print("Mean Precision:", mean_precision)

# Recall
recall_scores = cross_val_score(DTC, X, Y, cv=5, scoring='recall')
mean_recall = np.mean(recall_scores)
print("Mean Recall:", mean_recall)

# F1-Score
f1_scores = cross_val_score(DTC, X, Y, cv=5, scoring='f1')
mean_f1 = np.mean(f1_scores)
print("Mean F1-Score:", mean_f1)

# ROC-AUC
roc_auc_scores = cross_val_score(DTC, X, Y, cv=5, scoring='roc_auc')
mean_roc_auc = np.mean(roc_auc_scores)
print("Mean ROC-AUC:", mean_roc_auc)

"""# Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=10, random_state=0)

print("After Cross Validation\n")
# Accuracy
accuracy_scores = cross_val_score(rf, X, Y, cv=5, scoring='accuracy')
mean_accuracy = np.mean(accuracy_scores)
print("Mean Accuracy:", mean_accuracy)

# Precision
precision_scores = cross_val_score(rf, X, Y, cv=5, scoring='precision')
mean_precision = np.mean(precision_scores)
print("Mean Precision:", mean_precision)

# Recall
recall_scores = cross_val_score(rf, X, Y, cv=5, scoring='recall')
mean_recall = np.mean(recall_scores)
print("Mean Recall:", mean_recall)

# F1-Score
f1_scores = cross_val_score(rf, X, Y, cv=5, scoring='f1')
mean_f1 = np.mean(f1_scores)
print("Mean F1-Score:", mean_f1)

# ROC-AUC
roc_auc_scores = cross_val_score(rf, X, Y, cv=5, scoring='roc_auc')
mean_roc_auc = np.mean(roc_auc_scores)
print("Mean ROC-AUC:", mean_roc_auc)

rf.fit(X_train, Y_train)
y_pred = rf.predict(X_test)

# from interpret.blackbox import LimeTabular
# from interpret import show

# # Local interpretability with LIME
# lime = LimeTabular(predict_fn=rf.predict_proba,
#                    data=X_train,
#                    random_state=1)

# # Get local explanations
# lime_local = lime.explain_local(X_test[-20:], Y_test[-20:], name='LIME')
# show(lime_local)

# # Explain local prediction with your trained model
# rf_local = rf.explain_local(X_test[:100], y_test[:100], name='XGBoost')
# show(rf_local)

# # Explain global predictions with your trained model
# rf_global = rf.explain_global(name='XGBoost')
# show(rf_global)

"""# Support Vector Machines"""

# from sklearn.svm import SVC
# clfSVM = SVC(gamma='auto')

# svc = SVC()
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# clfSVM = GridSearchCV(svc, parameters)

# clfSVM.fit(X_train,Y_train)
# y_pred = clfSVM.predict(X_test)
# print("Without Scaling and without CV: ",accuracy_score(Y_test,y_pred))

# print("After Cross Validation\n")
# # Accuracy
# accuracy_scores = cross_val_score(clfSVM, X, Y, cv=5, scoring='accuracy')
# mean_accuracy = np.mean(accuracy_scores)
# print("Mean Accuracy:", mean_accuracy)

# # Precision
# precision_scores = cross_val_score(clfSVM, X, Y, cv=5, scoring='precision')
# mean_precision = np.mean(precision_scores)
# print("Mean Precision:", mean_precision)

# # Recall
# recall_scores = cross_val_score(clfSVM, X, Y, cv=5, scoring='recall')
# mean_recall = np.mean(recall_scores)
# print("Mean Recall:", mean_recall)

# # F1-Score
# f1_scores = cross_val_score(clfSVM, X, Y, cv=5, scoring='f1')
# mean_f1 = np.mean(f1_scores)
# print("Mean F1-Score:", mean_f1)

# # ROC-AUC
# roc_auc_scores = cross_val_score(clfSVM, X, Y, cv=5, scoring='roc_auc')
# mean_roc_auc = np.mean(roc_auc_scores)
# print("Mean ROC-AUC:", mean_roc_auc)

"""# Naive Bayers"""

from sklearn.naive_bayes import GaussianNB
clf2 = GaussianNB()

print("After Cross Validation\n")
# Accuracy
accuracy_scores = cross_val_score(clf2, X, Y, cv=5, scoring='accuracy')
mean_accuracy = np.mean(accuracy_scores)
print("Mean Accuracy:", mean_accuracy)

# Precision
precision_scores = cross_val_score(clf2, X, Y, cv=5, scoring='precision')
mean_precision = np.mean(precision_scores)
print("Mean Precision:", mean_precision)

# Recall
recall_scores = cross_val_score(clf2, X, Y, cv=5, scoring='recall')
mean_recall = np.mean(recall_scores)
print("Mean Recall:", mean_recall)

# F1-Score
f1_scores = cross_val_score(clf2, X, Y, cv=5, scoring='f1')
mean_f1 = np.mean(f1_scores)
print("Mean F1-Score:", mean_f1)

# ROC-AUC
roc_auc_scores = cross_val_score(clf2, X, Y, cv=5, scoring='roc_auc')
mean_roc_auc = np.mean(roc_auc_scores)
print("Mean ROC-AUC:", mean_roc_auc)

"""# XGBoost"""

clf2 = xgb.XGBClassifier(
    n_estimators=100,  
    max_depth=5,
    learning_rate=0.1, 
    objective='binary:logistic', 
    random_state=42  
)
clf2.fit(X_train, Y_train)

target_prediction = clf2.predict(X_test)

data_accuracy = accuracy_score(Y_test, target_prediction)
print("Accuracy:", data_accuracy)

print("After Cross Validation\n")
# Accuracy
accuracy_scores = cross_val_score(clf2, X, Y, cv=5, scoring='accuracy')
mean_accuracy = np.mean(accuracy_scores)
print("Mean Accuracy:", mean_accuracy)

# Precision
precision_scores = cross_val_score(clf2, X, Y, cv=5, scoring='precision')
mean_precision = np.mean(precision_scores)
print("Mean Precision:", mean_precision)

# Recall
recall_scores = cross_val_score(clf2, X, Y, cv=5, scoring='recall')
mean_recall = np.mean(recall_scores)
print("Mean Recall:", mean_recall)

# F1-Score
f1_scores = cross_val_score(clf2, X, Y, cv=5, scoring='f1')
mean_f1 = np.mean(f1_scores)
print("Mean F1-Score:", mean_f1)

# ROC-AUC
roc_auc_scores = cross_val_score(clf2, X, Y, cv=5, scoring='roc_auc')
mean_roc_auc = np.mean(roc_auc_scores)
print("Mean ROC-AUC:", mean_roc_auc)

!pip install shap

import shap

shap.initjs()

explainer = shap.Explainer(clf2)

shap_values = explainer(X)

shap.plots.waterfall(shap_values[0])

import shap
explainer = shap.TreeExplainer(rf)

start_index = 0
end_index = 10
shap_values = explainer.shap_values(X_test[start_index:end_index])
X_test[start_index:end_index]


shap.summary_plot(shap_values,X_test)

import shap

explainer = shap.TreeExplainer(rf)

start_index = 0
end_index = 10

shap_values_subset = explainer.shap_values(X_test.iloc[start_index:end_index])
X_test[start_index:end_index]

shap.summary_plot(shap_values_subset, X_test.iloc[start_index:end_index])

# import shap
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split

# # Assuming X and y are your data and labels
# # X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the XGBoost Classifier
# model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
# model.fit(X_train, Y_train)

# # Initialize SHAP TreeExplainer
# explainer = shap.TreeExplainer(model)

# # Choose an instance to explain
# instance_index = 1  # Adjust the index to the instance you want to explain
# shap_values_instance = explainer.shap_values(X_test.iloc[instance_index])

# # Plot the Tree SHAP explanation for the selected instance
# shap.force_plot(
#     explainer.expected_value, shap_values_instance, X_test.iloc[instance_index]
# )
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, Y_train)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

if len(shap_values) == 2: 
    shap_values = shap_values[1]

shap.summary_plot(shap_values, X_test, feature_names=X_test.columns.tolist())

instance_index = 0 
shap.force_plot(
    explainer.expected_value, shap_values[instance_index], X_test.iloc[instance_index]
)

"""# LIME for XGBoost"""

X_test.values[8]

Y_test.values[:50]

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                   feature_names=X.columns.tolist(),
                                                   class_names=['Dropout', 'Graduate'],
                                                   discretize_continuous=True)

predict_fn = clf2.predict_proba

sample_index = 8  
exp = explainer.explain_instance(X_test.values[sample_index],
                                 predict_fn,
                                 num_features=6)

exp.show_in_notebook()  
exp.as_list()

# from interpret.blackbox import LimeTabular
# from interpret import show

# # Local interpretability with LIME
# lime = LimeTabular(predict_fn=clf2.predict_proba,
#                    data=X_train,
#                    random_state=1)

# # Get local explanations
# lime_local = lime.explain_local(X_test[-20:], Y_test[-20:], name='LIME')
# show(lime_local)

# # Explain local prediction with your trained model
# rf_local = clf2.explain_local(X_test[:100], y_test[:100], name='XGBoost')
# show(rf_local)

# # Explain global predictions with your trained model
# rf_global = clf2.explain_global(name='XGBoost')
# show(rf_global)

# from lime import lime_tabular
# from sklearn.pipeline import make_pipeline
# lime = lime_tabular(predict_fn=clf2.predict_proba,
#                    data=X_train,
#                    random_state=1)

# ## %% Get local explanations
# # lime_local = lime.explain_local(X_test[-20:],
# #                                 y_test[-20:],
# #                                 name='LIME')

# show(lime_local)

# # %%
# # %% Explain local prediction
# rf_local = clf2.explain_local(X_test[:100], y_test[:100], name='Logistic Regression')
# show(lrr_local)

# # %% Explain global predictions
# rf_global = clf2.explain_global(name='Lasso-Logistic Regression')
# show(lrr_global)
# # %%

"""# Gradient Boost"""

from sklearn.ensemble import GradientBoostingClassifier
clf2 = GradientBoostingClassifier()

clf2.fit(X_train, Y_train)

target_prediction = clf2.predict(X_test)

accuracy = accuracy_score(Y_test, target_prediction)
print(f'Accuracy: {accuracy}')

print("After Cross Validation\n")
# Accuracy
accuracy_scores = cross_val_score(clf2, X, Y, cv=5, scoring='accuracy')
mean_accuracy = np.mean(accuracy_scores)
print("Mean Accuracy:", mean_accuracy)

# Precision
precision_scores = cross_val_score(clf2, X, Y, cv=5, scoring='precision')
mean_precision = np.mean(precision_scores)
print("Mean Precision:", mean_precision)

# Recall
recall_scores = cross_val_score(clf2, X, Y, cv=5, scoring='recall')
mean_recall = np.mean(recall_scores)
print("Mean Recall:", mean_recall)

# F1-Score
f1_scores = cross_val_score(clf2, X, Y, cv=5, scoring='f1')
mean_f1 = np.mean(f1_scores)
print("Mean F1-Score:", mean_f1)

# ROC-AUC
roc_auc_scores = cross_val_score(clf2, X, Y, cv=5, scoring='roc_auc')
mean_roc_auc = np.mean(roc_auc_scores)
print("Mean ROC-AUC:", mean_roc_auc)

"""# CatBoost"""

pip install catboost



from catboost import CatBoostClassifier

model = CatBoostClassifier(iterations=70, depth=6, learning_rate=0.1, verbose=0)
model.fit(X_train, Y_train)

target_prediction = model.predict(X_test)

accuracy = accuracy_score(Y_test, target_prediction)
print(f'Accuracy: {accuracy:.6f}')

sixth_instance = X_test.iloc[5:6]

sixth_instance_prediction = model.predict(sixth_instance)

print(f"The prediction for the 6th instance is: {sixth_instance_prediction[0]}")
print(f"Actaul value for the 6th instance is: {Y_test.iloc[5]}")

"""dropout - 174
questionnaire - 0, 14, 36, 37(random seed=42), 300
"""

import lime
import lime.lime_tabular

# random seed 51
np.random.seed(51)
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=['Dropout', 'Graduate'],
    mode='classification'
)
# 174

# for i in range(300,310):
#   instance_index = i # Adjust the index if you want to explain a different instance
#   instance = np.array(X_test.iloc[i])  # Convert the instance to a numpy array

#   # Generate a LIME explanation for this instance
#   exp = explainer.explain_instance(
#       data_row=instance,
#       predict_fn=model.predict_proba,  # CatBoost predict_proba for the explainer
#       num_features=10
#   )
#   print("Instance No : ",i)
#   exp.show_in_notebook(show_table=True, show_all=False)
instance_index = 300 
instance = np.array(X_test.iloc[instance_index])  

exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=model.predict_proba,
    num_features=10
)
exp.show_in_notebook(show_table=True, show_all=False)

for i, idx in enumerate(X_test.index):
    if(idx==3482):
      print(f"{i} : {idx}")


conditions = X_test['Age at enrollment'] == 37

filtered_instances = X_test[conditions]

filtered_indices = filtered_instances.index

for index in filtered_indices:
    print(f"Index: {index}, Feature Values: {filtered_instances.loc[index].to_dict()}")

for index, value in enumerate(Y_test):
    print(f"Index: {index}, Value: {value}")

"""# SHAP for CatBoost"""

# import shap

# # Initialize the SHAP explainer with your CatBoost model
# explainer = shap.TreeExplainer(model)

# # Compute SHAP values for the test set
# # Note: Depending on your dataset size, this can be computationally expensive
# shap_values = explainer.shap_values(X_test)

# # SHAP values are computed for each class separately in classification. To visualize global effects,
# # you might focus on the class of interest, e.g., shap_values[1] for binary classification's positive class

import shap
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

model = CatBoostClassifier(iterations=70, depth=6, learning_rate=0.1, verbose=0)
model.fit(X_train, Y_train)

target_prediction = model.predict(X_test)
accuracy = accuracy_score(Y_test, target_prediction)
print(f'Accuracy: {accuracy:.6f}')

explainer = shap.Explainer(model)

shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)

shap.plots.bar(shap_values)

shap.plots.beeswarm(shap_values)

instance_index = 300
shap.plots.waterfall(shap_values[instance_index])

shap_interaction_values = explainer.shap_interaction_values(X_test)
shap.dependence_plot(
    ('GDP', 'Course'),
    shap_interaction_values,
    X_test
)

shap.plots.heatmap(shap_values)

# explainer = shap.TreeExplainer(model)
# positive_sample = X_train.iloc[1]
# positive_class = Y_train.iloc[1]
# model.predict_proba(positive_sample.values.reshape(1,-1))
# shap_values_positive = explainer.shap_values(positive_sample)
# shap.force_plot(explainer.expected_value[1], shap_values_positive[1], positive_sample)

from catboost import CatBoostClassifier

tree_index = 0  # Index of the tree you want to visualize
model.plot_tree(tree_index)

print("After Cross Validation\n")
# Accuracy
accuracy_scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
mean_accuracy = np.mean(accuracy_scores)
print("Mean Accuracy:", mean_accuracy)

#Precision
precision_scores = cross_val_score(model, X, Y, cv=5, scoring='precision')
mean_precision = np.mean(precision_scores)
print("Mean Precision:", mean_precision)

# Recall
recall_scores = cross_val_score(model, X, Y, cv=5, scoring='recall')
mean_recall = np.mean(recall_scores)
print("Mean Recall:", mean_recall)

# F1-Score
f1_scores = cross_val_score(model, X, Y, cv=5, scoring='f1')
mean_f1 = np.mean(f1_scores)
print("Mean F1-Score:", mean_f1)

# ROC-AUC
roc_auc_scores = cross_val_score(model, X, Y, cv=5, scoring='roc_auc')
mean_roc_auc = np.mean(roc_auc_scores)
print("Mean ROC-AUC:", mean_roc_auc)

"""# AdaBoost"""

from sklearn.ensemble import AdaBoostClassifier

clf2 = AdaBoostClassifier(n_estimators=300,  
                  learning_rate=0.1)  

clf2.fit(X_train, Y_train)

target_prediction = clf2.predict(X_test)

# Evaluate the model's accuracy on the test data
accuracy = accuracy_score(Y_test, target_prediction)
print(f'Accuracy: {accuracy:.6f}')

print("After Cross Validation\n")
# Accuracy
accuracy_scores = cross_val_score(clf2, X, Y, cv=5, scoring='accuracy')
mean_accuracy = np.mean(accuracy_scores)
print("Mean Accuracy:", mean_accuracy)

# Precision
precision_scores = cross_val_score(clf2, X, Y, cv=5, scoring='precision')
mean_precision = np.mean(precision_scores)
print("Mean Precision:", mean_precision)

# Recall
recall_scores = cross_val_score(clf2, X, Y, cv=5, scoring='recall')
mean_recall = np.mean(recall_scores)
print("Mean Recall:", mean_recall)

# F1-Score
f1_scores = cross_val_score(clf2, X, Y, cv=5, scoring='f1')
mean_f1 = np.mean(f1_scores)
print("Mean F1-Score:", mean_f1)

# ROC-AUC
roc_auc_scores = cross_val_score(clf2, X, Y, cv=5, scoring='roc_auc')
mean_roc_auc = np.mean(roc_auc_scores)
print("Mean ROC-AUC:", mean_roc_auc)