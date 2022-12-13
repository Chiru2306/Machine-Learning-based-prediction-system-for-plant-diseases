#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('Book.csv')
df.head()


# In[8]:


df.describe()


# In[ ]:





# In[23]:


print(df.shape)


# In[24]:


df.columns


# In[25]:


df.info()


# In[26]:


df.isnull().sum()


# In[27]:


df['disease'].value_counts()


# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('Book.csv')

a = df.drop(columns='disease',axis=1)
b = df['disease']
print(a)


# In[3]:


print(b)


# In[4]:


a_train,a_test,b_train,b_test = train_test_split(a,b,test_size=0.2,stratify=b,random_state = 2)
print(a.shape,a_train.shape,a_test.shape)


# In[5]:


model = LogisticRegression()
model.fit(a_train,b_train)


# In[6]:


a_train_prediction = model.predict(a_train)
training_data_accuracy = accuracy_score(a_train_prediction,b_train)
print(training_data_accuracy)


# In[7]:


a_test_prediction = model.predict(a_test)
testing_data_accuracy = accuracy_score(a_test_prediction,b_test)
print(testing_data_accuracy)


# In[8]:


input_data = (0.4,14,90,14)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)


# In[9]:


input_data = (0.3,13,90,13)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)


# In[10]:


input_data = (0.3,13,90,13)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)


# In[11]:


input_data = (0.4,14,90,14)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)


# In[12]:


importance = model.coef_[0]

for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[13]:


scaler = StandardScaler()
scale = scaler.fit(a_train)
a_train = scale.transform(a_train)
a_test = scale.transform(a_test)


# In[14]:


confusion_matrix(b_test,a_test_prediction)


# In[15]:


matrix = classification_report(b_test,a_test_prediction)
print(matrix)


# In[38]:


a_train,a_test,b_train,b_test = train_test_split(a,b,test_size=0.2,random_state = 0)
sc = StandardScaler()
a_train = sc.fit_transform(a_train)
a_test = sc.transform(a_test)
lb = LabelEncoder()
b_train = lb.fit_transform(b_train)
b_test = lb.fit_transform(b_test)


# In[39]:


b_train_cat = tf.keras.utils.to_categorical(b_train)
b_test_cat = tf.keras.utils.to_categorical(b_test)


# In[40]:


a_train.shape, a_test.shape, b_train_cat.shape, b_test_cat.shape


# In[59]:


import tensorflow as tf
nnmodel = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8,activation = 'relu',input_dim = 4),
    tf.keras.layers.Dense(15,activation = 'relu'),
    tf.keras.layers.Dense(9,activation = 'relu'),
    tf.keras.layers.Dense(4,activation = 'softmax'),
])
nnmodel.summary()


# In[60]:


nnmodel.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics=['accuracy'])
hist = nnmodel.fit(a_train,b_train_cat,batch_size = 100,epochs= 100)


# In[61]:


b_pred = nnmodel.predict(a_test)


# In[49]:


np.argmax(b_test_cat, axis=1)


# In[47]:


np.argmax(b_pred,axis=1)


# In[62]:


m = tf.keras.metrics.Accuracy()
m.update_state(np.argmax(b_test_cat, axis=1), np.argmax(b_pred,axis=1))
m.result().numpy()


# In[ ]:




