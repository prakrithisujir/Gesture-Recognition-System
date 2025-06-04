import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten,Conv2D, MaxPool2D
from keras.utils import np_utils
import pandas as pd
import time

keras.regularizers.l1(0.01)
keras.regularizers.l2(0.01)
keras.regularizers.l1_l2(l1=0.01, l2=0.01)
# define model

def plot(history):#,current_time):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # using now() to get current time
    #current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    plt.savefig(f'Model-Accuracy.png')#+str(current_time)+'.svg')
    #plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    plt.savefig(f'Model-Loss.png')#+str(current_time)+'.svg')
    #plt.show()

def train(filename,num_epochs):
    df=pd.read_csv(filename)
    x=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    #x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=42)
    x.shape

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    x = x.reshape((x.shape[0], x.shape[1], n_features))
    x.shape
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=42)
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(x.shape[1], n_features)))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu',))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Conv1D(filters=16, kernel_size=1, activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=8, kernel_size=1, activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=4, kernel_size=1, activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(5,activation='softmax'))
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=int(num_epochs),verbose=1)
    #millisec = int(round(time.time() * 1000))
    #model.save('model'+str(millisec)+'.keras')
    plot(history)#,millisec)
    return(model)