# -*- coding: utf-8 -*-
"""
===========================================#
# Title:  Review Analysis using NLP and Naive Bayes

# Date:   7 Jan 2020

@author: Eshika Mahajan
#==========================================#
"""

############################### Natural Language Processing #######################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#we convert csv to tsv bcoz we want tab seperated reviews as comma can be used in one review also

############################### Functions for evaluation parameters ################

def summing_array(cm):
    sum_Arr=0
    for i in range(0,2):
        for k in range(0,2):
            sum_Arr+=cm[i][k]
    return sum_Arr   

def imp_param(cm):
    
    TP=cm[0][0]#True Positive
    TN=cm[1][1]#True Negative
    FP=cm[0][1]#false positive
    FN=cm[1][0]#false negative
    
    #accuracy
    Accuracy=(TP+TN)/(summing_array(cm))
    Accuracy=Accuracy*100
    print('Accuracy is ',Accuracy)
    #Precision
    Precision=(TP / (TP + FP))
    print('Precision is ',Precision)
    
    #recall
    Recall = TP / (TP + FN)
    print('Recall is ',Recall)
    #f1 score
    F1_Score = (2 * Precision * Recall )/ (Precision + Recall)
    print('F1_score is ',F1_Score)
    
    
########################################### DATA PREPROCESSING PART ################################
# Cleaning the texts
import re   #re libary works weven without nltk

import nltk
nltk.download('stopwords')  #stopwards is the collection of the words like 'the,is,am,are' etc...
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
#In above line i have subtracted the unwanted aplhanumeric characters etc  from my first review so that only meaningful and imp words remain in 
#my review and all the other operations are performed on these set of meaningful words rather than on all words.
#^ signifies that all the words except a-z and A-z
review = review.lower() # converted the words into lowercase.
review=review.split()

'''
This is removing the stopword from first review

'''

review = [word for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)

'''
STEMMING
in this process we take the root of EVERY word individually which converts the various tenses of the word
to the same tense 
example:
    loving---> love
    loved--> love
'''
review=review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
    
'''

doing the above procedure for the whole tsv file.

'''  
review_list = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    review_list.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) 
#max_features is used to create a sparse matrix of those many features and the less frequent words are removed from the matrix
X = cv.fit_transform(review_list).toarray()
y = dataset.iloc[:, 1].values    
    


########################################### USING NAIVE BAYES ################################
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)    
    

print('\n\n########## ANALAYSIS AS PER NAIVE BAYES ##########\n')
summing_array(cm)
imp_param(cm)
print('\n####################################################\n') 


################################################## USING RANDOM FOREST #################

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Fitting Random Forest
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting the Test set results
y_pred_rf = regressor.predict(X_test_rf)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(y_test_rf, y_pred_rf)    

print('########## ANALAYSIS AS PER RANDOM FOREST ##########\n')
summing_array(cm_rf)
imp_param(cm_rf)
print('\n####################################################\n')  

 
################################################## USING Decision Tree #################
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train_dc, X_test_dc, y_train_dc, y_test_dc = train_test_split(X, y, test_size = 0.45, random_state = 0)

# Fitting Random Forest
from sklearn.tree import DecisionTreeClassifier
regressor_dc = DecisionTreeClassifier(random_state = 0)
regressor_dc.fit(X, y)


# Predicting the Test set results
y_pred_dc = regressor_dc.predict(X_test_dc)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_dc = confusion_matrix(y_test_dc, y_pred_dc)    

print('########## ANALAYSIS AS PER DECISION TREE ##########')
summing_array(cm_dc)
imp_param(cm_dc)
print('\n####################################################\n')


####################################################################################################


 