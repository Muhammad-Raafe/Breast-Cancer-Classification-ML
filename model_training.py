from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("data.csv")
print(df.describe())
print(df.isnull().sum())
print(df.info())

print(df["diagnosis"].unique())

df=df.drop(columns=["id","Unnamed: 32"])
print(df.isnull().sum())

le=LabelEncoder()
df["diagnosis"]=le.fit_transform(df["diagnosis"])

x=df.drop("diagnosis",axis=1)
y=df['diagnosis']

x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

dc_model=DecisionTreeClassifier()

dc_param_grid = {
    'criterion': ['gini', 'entropy'],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

dc_grid = GridSearchCV(estimator=dc_model, param_grid=dc_param_grid, cv=5, scoring='accuracy')

dc_grid.fit(x_train,y_train)
dc_prediction=dc_grid.predict(x_test)

print("Best Parameters:", dc_grid.best_params_)
print("Cross Validation Score:", dc_grid.best_score_)
print("Accuracy:", accuracy_score(y_test, dc_prediction))
print("confusion_matrix:\n", confusion_matrix(y_test, dc_prediction))
print("ckassification_report:\n", classification_report(y_test, dc_prediction))





rf_model=RandomForestClassifier()


rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid=GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, scoring='accuracy')

rf_grid.fit(x_train,y_train)
rf_prediction=rf_grid.predict(x_test)

print("Best Parameters:", rf_grid.best_params_)
print("Cross Validation Score:", rf_grid.best_score_)  
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_prediction))
print("Classification Report:\n", classification_report(y_test, rf_prediction))
print("Accuracy:", accuracy_score(y_test, rf_prediction))




scaler = StandardScaler()
x_trained_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


sv_model=SVC()

sv_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
}

sv_grid=GridSearchCV(estimator=sv_model, param_grid=sv_param_grid, cv=5, scoring='accuracy')

sv_grid.fit(x_trained_scaled,y_train)
sv_prediction=sv_grid.predict(x_test_scaled)

print("Best Parameters:", sv_grid.best_params_)
print("Cross Validation Score:", sv_grid.best_score_)
print("Confusion Matrix:\n", confusion_matrix(y_test, sv_prediction))
print("Classification Report:\n", classification_report(y_test, sv_prediction))
print("Accuracy:", accuracy_score(y_test, sv_prediction))


# Comparison of all models
comparison_df = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'Support Vector Machine'],
    'Accuracy': [accuracy_score(y_test, dc_prediction), accuracy_score(y_test, rf_prediction), accuracy_score(y_test, sv_prediction)],
})

print(comparison_df)

# Visualization of the comparison
import matplotlib.pyplot as plt

models = comparison_df['Model']
accuracies = comparison_df['Accuracy']

sns.barplot(x=models, y=accuracies)
plt.title('Model Comparison')   
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()
