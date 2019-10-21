#random forest

import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


avocado=pd.read_csv("avocado.csv")
print(avocado.head())

x=pd.DataFrame([avocado['Total Volume'],
                avocado['Total Bags'],
                avocado['AveragePrice'],
                avocado['Small Bags'],
                avocado['Large Bags'],
                avocado['XLarge Bags']]).T
y=avocado['type']

#切割成75%訓練集，25%測試集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

cl = tree.DecisionTreeClassifier(max_depth = 3)
clf=cl.fit(x_train,y_train)

randomforest=RandomForestClassifier()
randomforest.fit(x_train,y_train)
predicted=randomforest.predict(x_test)
print(randomforest)

#Accuracy of random forest
print("Accuracy:",clf.score(x_test,y_test))
