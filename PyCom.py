#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:17:40 2022

@author: ehdiehkhaledian
"""

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=1)


import pandas as pd
import pickle
df = pd.read_csv("data.csv")

# L = len(df)//2
# df_1 = df.iloc[:L,:]
# df_2 = df.iloc[L+1:,:]

# df_1.to_csv("Data1.csv")
# df_2.to_csv("Data2.csv")


df["target"] = [0 if df["class"][i]=="tested_negative" else 1 for i in range(len(df))]
del df["class"]

X= df
y= df["target"]
del X["target"]


#Fitting model with training data
clf.fit(X, y)
pickle.dump(clf, open('model.pkl','wb'))
