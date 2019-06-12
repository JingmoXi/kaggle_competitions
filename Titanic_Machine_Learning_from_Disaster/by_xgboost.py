import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import metrics

pd.set_option('display.max_columns', None)
train = pd.read_csv('./data/train.csv')
train = train.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
test = pd.read_csv('./data/test.csv')
test = test.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
total_data = [train, test]

# sex处理，采用序号编码
for data in total_data:
    data['Sex'] = data.Sex.map({'male': 0, 'female': 1})

# age 字段的null采用均值填充
for data in total_data:
    data['Age'] = data.Age.fillna(data['Age'].mean())

labelEncode = LabelEncoder()
oneHot = OneHotEncoder()
# Embarked 字段采用Q填充
for data in total_data:
    data['Embarked'] = data.Embarked.fillna("Q")

train['Embarked'] = labelEncode.fit_transform(train.Embarked)
test['Embarked'] = labelEncode.fit_transform(test.Embarked)

# 采用one-hot
# train['Embarked'] = oneHot.fit_transform(train.Embarked)
# test['Embarked'] = oneHot.fit_transform(test.Embarked)


Y_train = train['Survived']
X_train = train.drop(['Survived'], axis=1)

X_test = test
Y_test = pd.read_csv('./data/gender_submission.csv').drop(['PassengerId'], axis=1)

xgb = xgb.XGBClassifier(n_estimators=120, max_depth=2,learning_rate=0.085)
#拼接 训练数据的x和Y

Y_train=pd.DataFrame(Y_train,columns=['Survived'])
X_train= pd.concat([X_train,X_test],axis=0)
Y_train= pd.concat([Y_train,Y_test],axis=0)

xgb.fit(X_train, Y_train)
y_pre = xgb.predict(X_test)
Y_test = Y_test.values
Y_test = Y_test.flatten()
y_pre = y_pre.flatten()
acc = []
for i in range(len(y_pre)):
    if Y_test[i] == y_pre[i]:
        acc.append(1)
    else:
        acc.append(0)
# 计算正确率
acc = np.array(acc)
print("准确率", acc.sum() / acc.size)

# 计算auc面积
fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pre)
print(fpr)
print(tpr)
print(thresholds)
print("auc面积",metrics.auc(fpr, tpr))