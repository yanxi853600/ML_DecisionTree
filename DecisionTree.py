
import pandas as pd
#from sklearn.tree import DecisionTreeClassifier
#import matplotlib.pyplot as plt
#from sklearn import preprocessing,tree
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
#import pydot
#from sklearn.tree import export_graphviz
from sklearn import tree

#download dataset
avocado=pd.read_csv("avocado.csv")

x=pd.DataFrame([avocado["Total Volume"],
                avocado["Total Bags"],
                avocado["AveragePrice"],
                avocado["Small Bags"],
                avocado["Large Bags"],
                avocado["XLarge Bags"],]).T


y=avocado["type"]

#切割成75%訓練集，25%測試集
xTrain,xTest,yTrain,yTest=train_test_split(x,y,random_state=1)

#決策樹分類器進行分類
dtree=tree.DecisionTreeClassifier(max_depth = 4)
clf=dtree.fit(xTrain,yTrain)

#Accuracy of dtree
predicted=dtree.predict(xTest)
print(dtree)
print("Accuracy:",clf.score(xTest,yTest))

with open("test.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              impurity = True,
                              feature_names = list(xTrain),
                              class_names = ['conventional','organic'],
                              rounded = True,
                              filled= True )
     
from subprocess import check_call
check_call(['dot','-Tpng','test.dot','-o','test.png'])


from PIL import Image
img = Image.open("test.png")