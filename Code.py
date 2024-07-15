#!/usr/bin/env python
# coding: utf-8

# # Assignment 04

# Importing libraries

# In[1]:


import numpy as np
import tensorflow as tf
import tensorflow as tf
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.layers import Dense,SimpleRNN
from keras.models import Sequential
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# Reading text file

# In[2]:


with open("names.txt",'r') as f:
    data=f.readlines()
data=[text.strip('\n')+'.' for text in data]


# #Creating Dataset

# In[3]:


def create_dataset(data,ngram):
    X=[]
    Y=[]
    for text in data:
        pointer=0
        while pointer+ngram<len(text):
            X.append(text[pointer:pointer+ngram])
            Y.append(text[pointer+ngram])
            pointer+=1

    char_to_number={char:ind for ind,char in enumerate(sorted(set(Y)))}

    # converting text data to one hot encoding

    X = [[to_categorical(char_to_number[charector],27) for charector in  text_data] for text_data in X]
    X = np.array(X)

    # converting result to one hot encoding

    Y = [to_categorical(char_to_number[charector],27) for charector in Y]
    Y = np.array(Y)

    X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))

    final_data=np.concatenate([X,Y],axis=1)

    np.random.shuffle(final_data)

    partition_1=int(final_data.shape[0]*0.9)
    partition_2=partition_1+int(final_data.shape[0]*0.05)

    return np.split(final_data,[partition_1,partition_2]),char_to_number


# In[4]:


def create_model(ngram):

    def perplexity_loss(y_true, y_pred):
        cross_entropy = K.categorical_crossentropy(y_true, y_pred)
        perplexity = K.pow(np.e, cross_entropy)
        return perplexity

    model = Sequential([
        Dense(128,input_shape=(27*ngram,),activation='relu'),
        Dense(64,activation='relu'),
        Dense(27,activation='softmax'),
    ])

    model.compile(optimizer='adam',loss=perplexity_loss,metrics=['accuracy'])

    return model  


# # 2Gram model 

# In[5]:


early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)

(train,test,val),char_to_number=create_dataset(data,2)

model_2gram = create_model(2)
gram_2_history = model_2gram.fit(train[:,:27*2],train[:,27*2:],
                  validation_data=[val[:,:27*2],val[:,27*2:]],
                  epochs=30,callbacks=[early_stopping])


# #3 gram model

# In[6]:


(train,test,val),char_to_number=create_dataset(data,3)

model_3gram = create_model(3)
gram_3_history = model_3gram.fit(train[:,:27*3],train[:,27*3:],
                  validation_data=[val[:,:27*3],val[:,27*3:]],
                  epochs=30,callbacks=[early_stopping])


# In[7]:


early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)

def create_RNN_model(ngram):

    def perplexity_loss(y_true, y_pred):
        cross_entropy = K.categorical_crossentropy(y_true, y_pred)
        perplexity = K.pow(np.e, cross_entropy)
        return perplexity

    model = Sequential([
        SimpleRNN(128,input_shape=(ngram,27),activation='relu'),
        Dense(27,activation='softmax'),
    ])

    model.compile(optimizer='adam',loss=perplexity_loss,metrics=['accuracy'])

    return model  


# In[8]:


ngram = 3
(train,test,val),char_to_number=create_dataset(data,ngram)
model_2_rnn_gram = create_RNN_model(ngram)
model_2_rnn_gram.fit(train[:,:27*ngram].reshape(train.shape[0],ngram,27),train[:,27*ngram:],
          validation_data=[val[:,:27*ngram].reshape(val.shape[0],ngram,27),val[:,27*ngram:]],
          epochs=30,callbacks=[early_stopping])


# In[14]:


import matplotlib.pyplot as plt


plt.plot([x for x in range(len(gram_2_history.history['loss']))],gram_2_history.history['loss'],c='r')
plt.title("Epochs vs perplexity_loss for 2gram")
plt.x_label="Epochs"
plt.y_label="Loss"
plt.show()
plt.plot([x for x in range(len(gram_2_history.history['accuracy']))],gram_2_history.history['accuracy'],c='y')
plt.title("Epochs vs accuracy for 2gram")
plt.x_label="Epochs"
plt.y_label="Loss"
plt.show()
plt.plot([x for x in range(len(gram_3_history.history['loss']))],gram_3_history.history['loss'],c='r')
plt.title("Epochs vs perplexity_loss for 3gram")
plt.x_label="Epochs"
plt.y_label="Loss"
plt.show()
plt.plot([x for x in range(len(gram_3_history.history['accuracy']))],gram_3_history.history['accuracy'],c='y')
plt.title("Epochs vs accuracy for 3gram")
plt.x_label="Epochs"
plt.y_label="Loss"
plt.show()


# In[16]:


with open("logs.txt" , 'w') as f:
  for ind,history in enumerate([gram_2_history,gram_3_history]):
    for data in ['loss','accuracy']:
      for epoch,value in enumerate(history.history[data]):
        f.writelines(f"""For dataset_{ind+1} 
                        \t\t\t For epoch_{epoch} \t{data}= {round(value,2)}\n""")


# In[ ]:




