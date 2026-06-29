import pandas as pd
import numpy as np
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("breast-cancer.csv")
print(df.info())
print(df.isnull().sum())

le=LabelEncoder()
df["diagnosis"]=le.fit_transform(df["diagnosis"])
print(df)

x=df.drop(["id","diagnosis"],axis=1)
y=df["diagnosis"]

x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    random_state=42,
    test_size=0.2
)

model=RandomForestClassifier(random_state=42,criterion="gini",max_depth=5,min_samples_split=2,min_samples_leaf=2,n_estimators=100)
model.fit(x_train,y_train)
prediction=model.predict(x_test)

print("Accuracy Score",accuracy_score(y_test,prediction))
print("Confusion Matrix",confusion_matrix(y_test,prediction))
print("Classification Report: ",classification_report(y_test,prediction))

importance=model.feature_importances_

feature_importance=pd.DataFrame({
    "Feature":x.columns,
    "Importance":importance
})

feature_importance=feature_importance.sort_values(by="Importance",ascending=False)

print(feature_importance)

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature",
)
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.show()
