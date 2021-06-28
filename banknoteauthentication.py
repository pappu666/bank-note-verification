#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


import numpy as np


# In[4]:


df=pd.read_csv('C:\\Users\Pappu Gupta\\Desktop\\succesfull\\BankNote_Authentication.csv')


# In[5]:


df.head()


# In[6]:


x=df.iloc[:,:-1]


# In[8]:


y=df.iloc[:,-1]


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[12]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()


# In[13]:


classifier.fit(x_train,y_train)


# In[14]:


y_pred=classifier.predict(x_test)


# In[15]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)


# In[16]:


score


# In[17]:


import pickle


# In[18]:


pickle_out=open('classifier.pkl','wb')
pickle.dump(classifier,pickle_out)
pickle_out.close()


# In[20]:


classifier.predict([[2,3,4,1]])


# In[ ]:




