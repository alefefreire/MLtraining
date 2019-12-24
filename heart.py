#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 23:18:32 2019

@author: alefe
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')

heart=pd.read_csv("heart.csv")
print(heart.head(100))
print(heart.info())


#The dataset contains the following features:
#1. age(in years)
#2. sex: (1 = male; 0 = female)
#3. cp: chest pain type (tipo de dor no peito)
#4. trestbps: resting blood pressure (in mm Hg on admission to the hospital)
#5. chol: serum cholestoral in mg/dl
#6. fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)(açúcar no sangue em jejum)
#7. restecg: resting electrocardiographic results (resultados do eletro em repouso)
#-- Value 0: normal
#-- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
#-- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria 
#8. thalach: maximum heart rate achieved
#9. exang: exercise induced angina (1 = yes; 0 = no)(angina induzida por exercício), angina = dor no peito
#10. oldpeak: ST depression induced by exercise relative to rest
#11. slope: the slope of the peak exercise ST segment
#-- Value 1: upsloping
#-- Value 2: flat
#-- Value 3: downsloping 
#12. ca: number of major vessels (0-3) colored by flourosopy
#13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
#14. target: 1 or 0 
################################ EDA ##############################################################################
def bar_chart(feature):
    sick=heart[heart["target"]==1][feature].value_counts()
    healthy=heart[heart["target"]==0][feature].value_counts()
    df = pd.DataFrame([sick,healthy])
    df.index = ['Sick','Healthy']
    df.plot(kind='bar',stacked=True, figsize=(10,5))

bar_chart("sex")#maior parte de pessoas saudáveis é homem (curioso rs)
bar_chart("cp")#maioria doente com dor no peito nível 3
bar_chart("fbs") #maioria doente sem açucar no sangue em jejum
bar_chart("exang")#maioria doente sem agina induzida por exercício
bar_chart("restecg")#maioria dos doentes apresentou anormalidade em ST-T ondas
bar_chart("slope")#maioria dos doentes com downsloping

def hist_kde(feature,color):
    plt.figure(figsize=(10,5))
    sns.distplot(heart[feature],bins=30,kde=True,color=color)
    plt.show()

hist_kde("age","red")#distribuiçao da idade (pico em torno dos 60)
############################interessante seria plotar a distribuiçao também por sexo


male=heart[heart["sex"]==1]["age"]
female=heart[heart["sex"]==0]["age"]


plt.figure(figsize=(10,5))
sns.distplot(male,bins=30,kde=True,color="yellow")
plt.show()

plt.figure(figsize=(10,5))
sns.distplot(female,bins=30,kde=True,color="pink")
plt.show()

######################################################################################

hist_kde("chol","blue")
hist_kde("oldpeak","orange")

################# Predictions #######################################################################################

X= heart.drop('target',axis=1)
y=heart['target']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)

scaler = MinMaxScaler()# escala as features entre 0 e 1.

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(X_test_scaled)

# Test options and evaluation metric
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

seed = 7
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


from sklearn import metrics

lr=LogisticRegression().fit(X_train,y_train)
prob_lr=lr.predict_proba(X_train)

lda=LinearDiscriminantAnalysis().fit(X_train,y_train)
prob_lda=lda.predict_proba(X_train)

knn=KNeighborsClassifier().fit(X_train,y_train)
prob_knn=knn.predict_proba(X_train)

cart=DecisionTreeClassifier().fit(X_train,y_train)
prob_cart=cart.predict_proba(X_train)

gnb=GaussianNB().fit(X_train,y_train)
prob_gnb=gnb.predict_proba(X_train)

svm=SVC(probability=True).fit(X_train,y_train)
prob_svm=svm.predict_proba(X_train)


#Compute the ROC curve: true positives/false positives

tpr_lr,fpr_lr,thresh_lr=metrics.roc_curve(y_train,prob_lr[:,0])
tpr_lda,fpr_lda,thresh_lda=metrics.roc_curve(y_train,prob_lda[:,0])
tpr_knn,fpr_knn,thresh_knn=metrics.roc_curve(y_train,prob_knn[:,0])
tpr_cart,fpr_cart,thresh_cart=metrics.roc_curve(y_train,prob_cart[:,0])
tpr_gnb,fpr_gnb,thresh_gnb=metrics.roc_curve(y_train,prob_gnb[:,0])
tpr_svm,fpr_svm,thresh_svm=metrics.roc_curve(y_train,prob_svm[:,0])

#Area under Curve (AUC)
from sklearn.metrics import auc

roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_lda = auc(fpr_lda, tpr_lda)
roc_auc_knn = auc(fpr_knn, tpr_knn)
roc_auc_cart = auc(fpr_cart, tpr_cart)
roc_auc_gnb = auc(fpr_gnb, tpr_gnb)
roc_auc_svm = auc(fpr_svm, tpr_svm)

#Plotting the ROC curves


plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_lr, tpr_lr, label='LR, ROC curve (area = %0.2f)' % roc_auc_lr)
plt.plot(fpr_lda, tpr_lda, label='LDA, ROC curve (area = %0.2f)' % roc_auc_lda)
plt.plot(fpr_knn, tpr_knn, label='KNN, ROC curve (area = %0.2f)' % roc_auc_knn)
plt.plot(fpr_cart, tpr_cart, label='CART, ROC curve (area = %0.2f)' % roc_auc_cart)
plt.plot(fpr_gnb, tpr_gnb, label='NB, ROC curve (area = %0.2f)' % roc_auc_gnb)
plt.plot(fpr_svm, tpr_svm, label='SVC, ROC curve (area = %0.2f)' % roc_auc_svm)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# Make predictions on validation dataset
print("--------------------------")
print("LogisticRegression Report")
print("--------------------------")
predictions_lr = lr.predict(X_test)
print("accuracy =",accuracy_score(y_test, predictions_lr))
print("confusion matrix",confusion_matrix(y_test, predictions_lr))
print(classification_report(y_test, predictions_lr))

print("--------------------------")
print("LinearDiscriminantAnalysis Report")
print("--------------------------")
predictions_lda = lda.predict(X_test)
print("accuracy =",accuracy_score(y_test, predictions_lda))
print("confusion matrix",confusion_matrix(y_test, predictions_lda))
print(classification_report(y_test, predictions_lda))

print("--------------------------")
print("KNeighborsClassifier Report")
print("--------------------------")
predictions_knn = knn.predict(X_test)
print("accuracy =",accuracy_score(y_test, predictions_knn))
print("confusion matrix",confusion_matrix(y_test, predictions_knn))
print(classification_report(y_test, predictions_knn))

print("--------------------------")
print("DecisionTreeClassifier Report")
print("--------------------------")
predictions = cart.predict(X_test)
print("accuracy =",accuracy_score(y_test, predictions))
print("confusion matrix",confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

print("--------------------------")
print("GaussianNB Report")
print("--------------------------")
predictions_gnb = gnb.predict(X_test)
print("accuracy =",accuracy_score(y_test, predictions_gnb))
print("confusion matrix",confusion_matrix(y_test, predictions_gnb))
print(classification_report(y_test, predictions_gnb))

print("--------------------------")
print("SVC Report")
print("--------------------------")
predictions_svm = svm.predict(X_test)
print("accuracy =",accuracy_score(y_test, predictions_svm))
print("confusion matrix",confusion_matrix(y_test, predictions_svm))
print(classification_report(y_test, predictions_svm))

#Conclusao: O melhor método é o LDA! acurácia = 0.