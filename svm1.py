import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
dataset=pd.read_csv("IRIS.csv")
print(dataset)
x=dataset.iloc[:,0:4]
y=dataset.iloc[:,4]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20)
model=SVC()
model.fit(x_train, y_train)
# prdeict_IRIS=model.predict(y_test)
# print(accuracy_score)
