!pip install interpret

import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

data=pd.read_csv("drive/MyDrive/4th_Year_Research/Implementation/Dataset/Pre-Processed-Dataset.csv")
data.head()

X = data.drop('Target', axis=1)  
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_train, y_train)

# Making predictions
predictions = ebm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))

from interpret import show

ebm_global = ebm.explain_global()
show(ebm_global)