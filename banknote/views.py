from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def home(request):
    return render(request,'index.html')


def result(request):
    df=pd.read_csv('BankNote_Authentication.csv')
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
    classifier=RandomForestClassifier()
    classifier.fit(x_train,y_train)
    if request.method == 'POST':
        variance=int(request.POST['variance'])
        skewness=int(request.POST['skewness'])
        curtosis=int(request.POST['curtosis'])
        entropy=int(request.POST['entropy'])
        y_pred=classifier.predict([[variance,skewness,curtosis,entropy]])
        if y_pred==0:
            y_pred='Authenticate'
            return render(request,'result.html',{'prediction':y_pred})
    
        else:
            y_pred='Duplicate'
            return render(request,'Duplicate.html',{'prediction':y_pred})