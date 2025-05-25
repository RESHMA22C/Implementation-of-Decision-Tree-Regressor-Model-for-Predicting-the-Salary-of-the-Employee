# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

  1. Import the libraries and read the data frame using pandas
  2. Calculate the null values present in the dataset and apply label encoder.
  3.Determine test and training data set and apply decison tree regression in dataset.
  4.calculate Mean square error, data prediction and r2.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: RESHMA C
RegisterNumber:  212223040168
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

```

## Output:
## DATA HEAD:

 ![image](https://github.com/user-attachments/assets/2e2350c0-3cf1-46af-8534-f60d94fa6bd5)


## DATA INFO:

![image](https://github.com/user-attachments/assets/74a9e7a5-9f15-4074-9cd1-07cd64179fa2)


# ISNULL() AND SUM():

![image](https://github.com/user-attachments/assets/6acb83f7-c556-4971-a723-91edb5290a02)


# DATA HEAD FOR SALARY:

![image](https://github.com/user-attachments/assets/d960396b-5553-45d9-bbe7-f5bc9496615a)


# MEAN SQUARED ERROR:

![image](https://github.com/user-attachments/assets/c8795565-7d7c-4bbb-bb7c-bc845e6a01ae)

# R2 VALUE:

![image](https://github.com/user-attachments/assets/d5e19a9d-cb22-43b0-a886-7e0c265b5f16)

# DATA PREDICTION:

![image](https://github.com/user-attachments/assets/c4061e08-bcd1-4564-ac1b-374792f0adaf)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
