# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 22:09:06 2022
"""

import warnings
warnings.filterwarnings('ignore')
###loading dataset
import pandas as pd
df=pd.read_csv("Fraud_check.csv")
df
df.shape

####converting target variable in categorical form using Label encoder
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df["Undergrad"]=label_encoder.fit_transform(df["Undergrad"])
df["Marital.Status"]=label_encoder.fit_transform(df["Marital.Status"])
df["Urban"]=label_encoder.fit_transform(df["Urban"])
df.head()
df

###plotting pair plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(df,hue = "Taxable.Income")
### correlation matrix
sns.heatmap(df.corr())


df1=df.assign(Income=pd.cut(df['Taxable.Income'],bins=[ 0, 30000, 100000],labels=['Risky','Good']))
df1
df = df1.drop('Taxable.Income', axis =1)
df

####target and features
X= df.iloc[:,0:5]
list(X)
Y = df.iloc[:,5]
Y = label_encoder.fit_transform(Y)
Y
###train and test splitting
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.75,random_state=0)

####model creation

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,Y_train)

#accuracy

model.estimators_
model.classes_
model.n_features_
model.n_classes_
model.n_outputs_
#####model validation
from sklearn.metrics import accuracy_score,confusion_matrix
Y_pred=model.predict(X_test)
cm=confusion_matrix(Y_test,Y_pred)
cm
test_acc=accuracy_score(Y_test, Y_pred)
test_acc
Y_pred1=model.predict(X_train)
train_acc=accuracy_score(Y_train, Y_pred1)
train_acc

####model search(hyperparameter tuning)
###grid search
from sklearn.model_selection import GridSearchCV,KFold
params = {"max_depth": [3, None],"max_features": [1, 3, 10],"min_samples_split": [1, 3, 10],"min_samples_leaf": [1, 3, 10],"criterion": ["gini", "entropy"]}
model2=RandomForestClassifier()
grid=GridSearchCV(estimator=model2,param_grid=params,cv=KFold(n_splits=10))
grid_results=grid.fit(X,Y)
#summarize the results
print('best:{},using{}'.format(grid_results.best_score_,grid_results.best_params_))
means=grid_results.cv_results_['mean_test_score']
stds=grid_results.cv_results_['std_test_score']
params=grid_results.cv_results_['params']
for mean,stdev,param in zip(means,stds,params):
    print('{},{} with {}' .format(mean,stdev,param))

###kfold

from sklearn.model_selection import KFold,cross_val_score
fold=KFold(n_splits=10,shuffle=True, random_state=0)
model1=RandomForestClassifier()
result=cross_val_score(model1,X,Y,cv=fold)
result.mean(),result.std()


#####final model using kfold
from sklearn.model_selection import cross_val_score
fold=KFold(n_splits=10,shuffle=False)
model1=RandomForestClassifier(criterion= 'gini', max_depth= 3)
result=cross_val_score(model1,X,Y,cv=fold)
result
result.std()

#####final model using train and test
from sklearn.ensemble import RandomForestClassifier
final_model=RandomForestClassifier(criterion="gini",max_depth=3)
final_model.fit(X_train,Y_train)
#accuracy
from sklearn.metrics import accuracy_score
Y_pred=final_model.predict(X_test)
accuracy_score(Y_test, Y_pred)
Y_pred1=final_model.predict(X_train)
accuracy_score(Y_train, Y_pred1)



