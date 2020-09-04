import matplotlib

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score




"""Implementation of ML Algorithms¶
To predict the mobile phone prices, we are going to apply below algorithms respectively on the training and validation dataset. After that, we are going to choose the best model for our data set and create target values for test dataset.

Logistic regression
Decision tree
Random forest
KNN



Logistic Regression


lr = LogisticRegression(multi_class = 'multinomial', solver = 'sag',  max_iter = 10000)
lr.fit(x_train, y_train)


y_pred_lr = lr.predict(x_valid)

confusion_matrix = metrics.confusion_matrix(y_valid, y_pred_lr)
confusion_matrix


acc_lr = metrics.accuracy_score(y_valid, y_pred_lr)
acc_lr
0.73





Decision Tree¶
Decision tree is one of the most popular supervised learning algorithm that is mostly used in classification problems.


dt = DecisionTreeClassifier(random_state=101)
dt_model = dt.fit(x_train, y_train)


y_pred_dt = dt.predict(x_valid)


dt_model

print(metrics.confusion_matrix(y_valid, y_pred_dt))
[[89 11  0  0]
 [ 7 74 19  0]
 [ 1  9 80 10]
 [ 0  0 12 88]]


print(metrics.classification_report(y_valid, y_pred_dt))

   precision    recall  f1-score   support

           0       0.92      0.89      0.90       100
           1       0.79      0.74      0.76       100
           2       0.72      0.80      0.76       100
           3       0.90      0.88      0.89       100

   micro avg       0.83      0.83      0.83       400
   macro avg       0.83      0.83      0.83       400
weighted avg       0.83      0.83      0.83       400

How do we interpret the numbers in classification report?
Precision and recall metrices should be high as possible. They gives us accuracy level out of different classes.
We use f1-score to compare models that have different precision and recall levels.'





acc_dt = metrics.accuracy_score(y_valid, y_pred_dt)
acc_dt


0.8275



Random Forest¶
Random forest is a type of ensemble method of machine learning. It deals with overfitting problem and increase accuracy compared to a simple decision tree model.

rf = RandomForestClassifier(n_estimators = 100, random_state=101, criterion = 'entropy', oob_score = True) 
model_rf = rf.fit(x_train, y_train)


y_pred_rf = rf.predict(x_valid)


print(metrics.confusion_matrix(y_valid, y_pred_rf))
[[91  9  0  0]
 [ 3 91  6  0]
 [ 0  7 85  8]
 [ 0  0  6 94]]
 
 
 
 
 
 pd.crosstab(y_valid, y_pred_rf, rownames=['Actual Class'], colnames=['Predicted Class'])


Predicted Class	0	1	2	3
Actual Class				
0	91	9	0	0
1	3	91	6	0
2	0	7	85	8
3	0	0	6	94


acc_rf = metrics.accuracy_score(y_valid, y_pred_rf)
acc_rf


Out[218]:
0.9025

We see that accuracy score in random forest model is higher than decision tree accuracy score.












K-Nearest Neighbors (KNN)¶
'K' is the number of nearest training points which we classify them using the majority vote.


model_knn = KNeighborsClassifier(n_neighbors=3)  
model_knn.fit(x_train, y_train)


KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=3, p=2,
           weights='uniform')

y_pred_knn = model_knn.predict(x_valid)

print(metrics.confusion_matrix(y_valid, y_pred_knn))
[[94  6  0  0]
 [ 3 96  1  0]
 [ 0  3 92  5]
 [ 0  0  7 93]]

print(accuracy_score(y_valid, y_pred_knn))
0.9375
In the first try, we did not know the optimum 'k'.
Let's find the optimum 'k' value via Grid Search method and run knn model with this 'k'.







Conclusion¶
In [308]:
models = ['logistic regression', 'decision tree', 'random forest', 'knn']
acc_scores = [0.73, 0.83, 0.90, 0.95]

plt.bar(models, acc_scores, color=['lightblue', 'pink', 'lightgrey', 'cyan'])
plt.ylabel("accuracy scores")
plt.title("Which model is the most accurate?")
plt.show()
"""




data = pd.read_csv("train.csv")

y = data['price_range']
x = data.drop('price_range', axis = 1)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 101, stratify = y)


model_knn = KNeighborsClassifier(n_neighbors=9)
model_knn.fit(x_train, y_train)


y_pred_knn = model_knn.predict(x_valid)

pickle.dump(model_knn,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


test_data = pd.read_csv("test.csv")

print(test_data)
