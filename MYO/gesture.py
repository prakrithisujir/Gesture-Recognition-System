import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from numpy import var
from math import sqrt
from math import log
from math import exp
import os
import glob
import matplotlib.pyplot as plt #used for visualization purposes in this tutorial.
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten,Conv2D, MaxPool2D
import np_utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

model = tf.keras.models.load_model('model.h5')
model.summary()
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

def generate(sample_number):
    print("Generating...")
    df = pd.read_csv('xtest.csv')
    test_sample=df.iloc[sample_number,:-1].to_numpy()
    test_sample_label=df.iloc[sample_number,-1]
    test_sample = (np.expand_dims(test_sample,0))
    predictions_single = probability_model.predict(test_sample)
    print("Actual Gesture:",test_sample_label )
    print("Predicted Gesture:",np.argmax(predictions_single[0]))
    x_axis=['Point','Middle Finger Extension','Grip','Pinch','Rest']
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(test_sample[0])
    plt.subplot(2,1,2)
    plt.bar(x_axis,predictions_single[0],color ='maroon',width = 0.4)
    plt.tight_layout()
    plt.savefig('gesture.png')
    return test_sample_label,np.argmax(predictions_single[0])