# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:51:49 2021

@author: makn0023
"""

import os
import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
from nltk.stem import PorterStemmer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
# Binary Relevance
from sklearn.multiclass import OneVsRestClassifier
# Performance metric
from sklearn.metrics import f1_score
import nltk
import pickle
nltk.download('stopwords');
nltk.download('punkt')
from nltk import word_tokenize
from nltk import punkt
stop_words = set(stopwords.words("english"))
nltk.download('wordnet')
wnl = WordNetLemmatizer()
nltk.download('vader_lexicon')

path = 'C:\\\\Users\\\\makn0023\\\\Desktop\\\\challenge\\\\package\\\\challenge-env\\\\Challenge'
os.chdir(path)

dataf = pd.read_csv('C:\\\\Users\\\\makn0023\\\\Desktop\\\\challenge\\\\package\\\\challenge-env\\\\Challenge\\\\train.csv')
datat = pd.read_csv('C:\\\\Users\\\\makn0023\\\\Desktop\\\\challenge\\\\package\\\\challenge-env\\\\Challenge\\\\test.csv')

dataf.info()
def removeNull(wrdList):
     for wrd in wrdList :
         if wrd == '':
             wrdList.remove(wrd)
     return wrdList
 
def checkEmpty(stringVal):
    if not stringVal:
        return True
    else :
        return False
stemming = PorterStemmer()
list(dataf)

def data(df):
    df['synopsis']= df['synopsis'].str.replace('[^a-zA-Z#]+',' ')
    df['synopsis'] = df['synopsis'].str.lower()
    df['synopsis'] = df['synopsis'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
    df['synopsis']=df.synopsis.apply(lambda row: word_tokenize(row))
    df['synopsis']
    df['synopsis'] = df['synopsis'].apply(lambda x: [item for item in x if item not in stop_words])
    df['synopsis'] = df['synopsis'].apply(lambda x: [wnl.lemmatize(i) for i in x ])
    df['synopsis'] = df['synopsis'].apply(lambda x: removeNull(x))
    df['synopsis1'] = df['synopsis'].apply(lambda x: checkEmpty(x))
    df[df['synopsis1']==True]
    df['synopsis1'] = df[df['synopsis1']==False]
    df = df.drop(["synopsis1"],axis=1)
    df['synopsis']=df['synopsis'].apply(lambda x: ' '.join([word for word in x]))
    df.reset_index(drop=True,inplace=True)
    df.synopsis.head(100)
    return df

dataf['genres_new']=dataf.genres.apply(lambda row: word_tokenize(row))
dataclean = data(dataf)

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(dataclean['genres_new'])

# transform target variable
y = multilabel_binarizer.transform(dataclean['genres_new'])

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=1000000)

xtrain, xval, ytrain, yval = train_test_split(dataclean['synopsis'], y, test_size=0.2, random_state=9)

xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

lr = LogisticRegression()
clf = OneVsRestClassifier(lr)

clf.fit(xtrain_tfidf, ytrain)
#y_pred = clf.predict(xval_tfidf)


#y_pred[2]
#multilabel_binarizer.inverse_transform(y_pred)[2]
#f1_score(yval, y_pred, average="micro")


# predict probabilities
y_pred_prob = clf.predict_proba(xval_tfidf)
t = 0.3 # threshold value
y_pred_new = (y_pred_prob >= t).astype(int)
f1_score(yval, y_pred_new, average="micro")

a=multilabel_binarizer.inverse_transform(y_pred_new)
a

dfa = pd.DataFrame(a)
dfa['Geners'] = dfa[dfa.columns[0:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
)
dfa=dfa.drop([0,1,2,3,4,5], axis=1)
dataclean['Geners Predicted'] = dfa['Geners'].copy()
dataclean = dataclean.drop(['genres_new'],axis=1)

#Test Data
dataft = data(datat)

xvalft = dataft['synopsis'].copy()
xval_tfidft = tfidf_vectorizer.transform(xvalft)

ypreda = clf.predict(xval_tfidft)
y_pred_probt = clf.predict_proba(xval_tfidft)
y_pred_newt = (y_pred_probt >= t).astype(int)
abcd=multilabel_binarizer.inverse_transform(y_pred_newt)
abcd

dummy = pd.DataFrame(abcd)
dummy['Geners'] = dummy[dummy.columns[0:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)),
    axis=1
)
dummy=dummy.drop([0,1,2,3,4,5], axis=1)
dataft['Geners Predicted'] = dummy['Geners'].copy()
#

#dataft[['movie_id','year','synopsis']].groupby(['Geners Predicted']).size() #the test predicted data
ad = dataft.groupby(["movie_id","Geners Predicted"], sort=True)["Geners Predicted"].count()
ad = ad.head(5)
ad.to_csv('predicted.csv')
pickle.dump(clf, open('C:\\\\Users\\\\makn0023\\\\Desktop\\\\challenge\\\\package\\\\challenge-env\\\\Challenge\\\\Trainedmodel\\\\modelml.pkl','wb'))

modelml = pickle.load(open('C:\\\\Users\\\\makn0023\\\\Desktop\\\\challenge\\\\package\\\\challenge-env\\\\Challenge\\\\Trainedmodel\\\\modelml.pkl','rb'))
#print(model.predict([["""Cruel But Necessary is the story of Betty Munson's strange journey of self-discovery and soul-awakening in the traumatic years following the revelation, on videotape, of her husband's infidelity. Her marriage over, struggling to raise her teen-age son alone, Betty becomes driven to discover other secrets that may surround her and so she videotapes every aspect of her life during the gradual disintegration of her comfortable upper middle-class existence. Sometimes used as an eavesdropping device, other times as a confessional, Betty's camera dispassionately records the layers of family and personal dynamics. The film is seen entirely from the viewpoint of Betty's video camera resulting in a "surveillance tape" that is a kind of voyeurism of the absurd"""]]))
##################################################################################
#Deep Learning

#from keras.models import Sequential
#from keras.layers import Dropout,Flatten
##from keras.layers.core import Dense, Activation, Dropout
#from keras.layers import *
#from keras import backend as K
#from keras.callbacks import ReduceLROnPlateau
#from keras.layers.recurrent import LSTM

#max_len = 50000000   #length of sequence
#batch_size = 256
#epochs = 10
#max_features = 64  # (number of words in the vocabulary) + 1
#X = dataclean[['movie_id','year','synopsis']]
#y = dataclean['genres'].values
#x_traind, x_testd, y_traind, y_testd = train_test_split(X,y, test_size=0.2, random_state=42)

#lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)
#label_num = len(y_traind[0])

#def swish_activation(x):
 #   return (K.sigmoid(x) * x)

#model = Sequential()

#model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=x_traind.shape)
#model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
#model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
#model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
#model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(64, activation=swish_activation))
#model.add(Dropout(0.4))
#model.add(Dense(label_num , activation='sigmoid'))

#model.compile(loss='binary_crossentropy',
#              optimizer='adam' ,
#              metrics=['accuracy'])
#batch_size = 128
#epochs = 14
#model.summary()
#model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])
#steps_per_epoch = 16
#validation_steps = len(()) // batch_size


#history = model.fit(x_traind, y_traind, batch_size=batch_size,
#                    steps_per_epoch=X.shape[0] // batch_size,
#                    callbacks=[lr_reduce],
#                    validation_data=(x_testd, y_testd),
#                    epochs = epochs, verbose = 2)



#model = Sequential()
#model.add(Embedding(input_dim = max_features, output_dim = 64, mask_zero = True, input_length = max_len))
#model.add(embedding_layer)
#model.add(Dropout(0.3))
#model.add(LSTM(64, return_sequences = True))
#model.add(Dropout(0.3))
#model.add(LSTM(64))
#model.add(Dropout(0.3))
#model.add(Dense(40, activation = 'sigmoid'))

##print(model.summary())
#rmsprop = optimizers.RMSprop(lr = 0.01, decay = 0.0001)
#model.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics=['accuracy'])



#from tensorflow.keras.layers import Input, Dense, Activation,Dropout
#from tensorflow.keras.models import Model

#X_traind, X_testd, y_traind, y_testd = train_test_split(X, y, test_size=0.20, random_state=42)

#input_layer = Input(shape=(X.shape[1],))
#dense_layer_1 = Dense(15, activation='relu')(input_layer)
#dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
#output = Dense(y.shape[0], activation='sigmoid')(dense_layer_2)

#model = Model(inputs=input_layer, outputs=output)
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
#model.summary()
#history = model.fit(X_traind, y_traind, batch_size=8, epochs=50, verbose=1, validation_split=0.2)
