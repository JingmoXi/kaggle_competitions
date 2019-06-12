import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
pd.set_option('display.max_columns', None)
train_data = pd.read_csv('./data/train.csv')
train_y=train_data['Survived']
train_x = train_data.drop(['PassengerId','Survived', 'Name', 'Cabin', 'Ticket'], axis=1)
test_x = pd.read_csv('./data/test.csv')
test_x = test_x.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
total_data = [train_x, test_x]

# sex离散化 one-hot编码
for data in total_data:
    data['Sex'] = data.Sex.map({'male': 0, 'female': 1})
    data['Embarked']=data['Embarked'].fillna("S")
# age 字段的null采用均值填充
for data in total_data:
    data['Age'] = data.Age.fillna(data['Age'].mean())


enc=OneHotEncoder(sparse = False)
sex_onehot= enc.fit_transform(pd.DataFrame(train_x['Sex']))
train_x["sex_0"]=sex_onehot[:,0]
train_x["sex_1"]=sex_onehot[:,1]
train_x=train_x.drop(["Sex"], axis=1)

sex_onehot_test=enc.transform(pd.DataFrame(test_x['Sex']))
test_x["sex_0"]=sex_onehot_test[:,0]
test_x["sex_1"]=sex_onehot_test[:,1]
test_x=test_x.drop(["Sex"], axis=1)


Embarked_onehot=OneHotEncoder(sparse = False)
Embarked_onehot_data= Embarked_onehot.fit_transform(pd.DataFrame(train_x['Embarked']))
train_x["Embarked_0"]=Embarked_onehot_data[:,0]
train_x["Embarked_1"]=Embarked_onehot_data[:,1]
train_x=train_x.drop(["Embarked"], axis=1)

Embarked_onehot_data_test=Embarked_onehot.transform(pd.DataFrame(test_x['Embarked']))
test_x["Embarked_0"]=Embarked_onehot_data_test[:,0]
test_x["Embarked_1"]=Embarked_onehot_data_test[:,1]
test_x=test_x.drop(["Embarked"], axis=1)

age_one = pd.cut(train_x.Age, [0, 18, 40, 80])
# age做离散化处理
age_one = pd.get_dummies(age_one)
train_x = pd.concat([train_x, age_one], axis=1)
train_x = train_x.drop(["Age"], axis=1)


age_one = pd.cut(test_x.Age, [0, 18, 40, 80])
# age做离散化处理
age_one = pd.get_dummies(age_one)
test_x = pd.concat([test_x, age_one], axis=1)
test_x = test_x.drop(["Age"], axis=1)

Y_test = pd.read_csv('./data/gender_submission.csv').drop(['PassengerId'], axis=1)
Y_test = Y_test.values
Y_test = Y_test.flatten()

lasso= Lasso(alpha=0.3,max_iter=5000)
lasso.fit(X=train_x,y=train_y)
#test_x.isnull().any()
test_x['Fare'] = test_x.Fare.fillna(data['Fare'].mean())
predict_y=lasso.predict(test_x)


acc=[]
for i in range(len(predict_y)):
    if (Y_test[i] ==1 and  predict_y[i]>=0.5) or (Y_test[i] ==0 and  predict_y[i]<0.5):
        acc.append(1)
    else:
        acc.append(0)
#print(acc)
print(sum(acc)/len(acc))