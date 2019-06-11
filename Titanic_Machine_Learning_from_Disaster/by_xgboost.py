import pandas as pd
import xgboost as xgb

train_data=pd.read_csv("./data/train.csv")
train_y=train_data["Survived"]
train_x=train_data.drop(["PassengerId","Survived","Ticket",'Name','Cabin','Embarked'],axis=1)
#n缺失值处理

age_mean=train_x.mean(axis=0)['Age']
train_x['Age']=train_x['Age'].fillna(age_mean)
train_x['Sex']=train_x.Sex.replace(['female','male'],[0,1])
train_x['Sex'].fillna(0)
train_x['Sex']=train_x['Sex'].astype('int')
# train_x['Name']=train_x.Name.map(lambda x:str(x))
# train_x['Embarked']=train_x.Embarked.map(lambda x:str(x))
# train_x['Cabin']=train_x.Cabin.map(lambda x:str(x))

#train_x=train_x.astype({"Sex":str,"Name":str,"Embarked":str,"Cabin":str})
model= xgb.XGBClassifier(n_estimators=800,learning_rate=0.05)
model.fit(X=train_x,y=train_y)

test_x=pd.read_csv("./data/test.csv").drop(["PassengerId","Ticket",'Name','Cabin','Embarked'],axis=1)
test_x['Sex']=test_x.Sex.replace(['female','male'],[0,1])
test_x['Sex'].fillna(0)
test_x['Sex']=test_x['Sex'].astype('int')
y_predict=model.predict(test_x)
print(y_predict)


y_real=pd.read_csv("./data/gender_submission.csv")['Survived']


#计算准确率
def accuracy_rate(predict,real):
    s=predict-real
    #s=s==0
    num=0
    for i in s:
        if i==0:
            num=num+1
    #num = str(s).count('0')
    total=len(real)

    return  num/total
print(accuracy_rate(y_predict,y_real))

#计算f1-score
def f1_score():
    pass