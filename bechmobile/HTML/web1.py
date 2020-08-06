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