from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

def evaluate_prob(model,data):
    #概率预测
    x,label = data
    logits = model.predict_proba(x)
    pred = logits
    return pred

def evaluate_class(model,data):
    #分类预测
    x,label = data
    logits = model.predict(x)
    pred = logits
    return pred

def trainmodel( data ):
    #随机森林分类
    x,y = data
    model = RandomForestClassifier(n_estimators = 70,max_depth=28,min_samples_leaf=10,random_state = 66)
    model.fit ( x , y )
    return model
