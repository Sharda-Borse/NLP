# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 21:52:12 2019

@author: ShardaBorse
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#read the text from diffrent file and merge it to dataset dataframe
df1 = pd.read_table('apple-computers.txt', names=('TextData','OutputLabel'))
df1['OutputLabel']='computer-company'
df1.head()


df2 = pd.read_table('apple-fruit.txt', names=('TextData','OutputLabel'))
df2['OutputLabel']='fruit'
df2.head()

dataset=df1.append(df2,ignore_index =True,verify_integrity =True)




import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, dataset['TextData'].size):
    textData = re.sub('[^a-zA-Z]', ' ', str(dataset['TextData'][i]))
    textData = textData.lower()
    textData = textData.split()
    ps = PorterStemmer()
    textData = [ps.stem(word) for word in textData if not word in set(stopwords.words('english'))]
    textData = ' '.join(textData)
    corpus.append(textData)


    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values


# splitting datasetinto trainset and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# fitting naive bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

#predicting test results
y_pred=classifier.predict(X_test)

# confusion matrix 
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(accuracy_score(y_test,y_pred))



# fitting SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)


#predicting test results
y_pred=classifier.predict(X_test)

# confusion matrix 
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(accuracy_score(y_test,y_pred))




# fitting SVM Kernel
from sklearn.svm import SVC
classifier = SVC(kernel = 'sigmoid', random_state = 0)
classifier.fit(X_train, y_train)


#predicting test results
y_pred=classifier.predict(X_test)

# confusion matrix 
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(accuracy_score(y_test,y_pred))

# logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test,y_pred))


# Fitting K- Nearest Neighbors Classifier (KNN) to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test,y_pred))




# Fitting Decision Tree Classifier (DTC) to the Training set

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test,y_pred))


# Fitting Random Forest Classifier (RFC) to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(accuracy_score(y_test,y_pred))



#Test user inputs 


corpusTest=[]
n=int(input("Enter the number of queries you want to know the category it belongs to : "))
for i in range(n):
    text=input("Enter query : ")
    text=re.sub('[^a-zA-Z]',' ',str(text))
    text=text.lower()
    text=text.split()
    ps=PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpusTest.append(text)

print(classifier.predict(cv.transform(corpusTest).toarray()))    


