'''
First machine learning project of classifying and predicting data
'''
#%%
'''Getting data'''

import pandas as pd

dataset=pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv",
        names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])

print(dataset.head())
#%%
'''Describing Data'''

print(dataset.shape)
dataset.describe()

print(dataset.groupby("class").size())
#%%
'''Spliting data into X and Y'''

array=dataset.values
X=array[:,0:4]
Y=array[:,4]
#%%
'''Splitting data into test and train'''

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

#%%
'''Building models'''

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

models=[]
models.append(('LR',LogisticRegression(solver='liblinear', multi_class='ovr'))) 
models.append(('DT',DecisionTreeClassifier()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('SVC',SVC(gamma='auto')))
models.append(('NB',GaussianNB()))
models.append(('LDA',LinearDiscriminantAnalysis()))

for name,model in models:
    kfold=StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
    cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    print("{} -> {} : {}".format(name,cv_results.mean(),cv_results.std()))

#%%
'''Predicting data using the best model'''

model=SVC(gamma='auto')
model.fit(X_train,Y_train)
predictions=model.predict(X_test)

#%%
'''Evaluting predictions or the accuracy of our model'''

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(accuracy_score(Y_test,predictions))
print(classification_report(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
