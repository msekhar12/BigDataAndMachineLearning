#################################
##Program Name: program.py     ##
##Python version: 3.4          ##
##Author: Sekhar Mekala        ##
#################################

#Importing all the required packahes
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split


import numpy as np
from keras.utils import np_utils
import tensorflow as tf
# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout


#Please see the document "writeupdocx.docx", for a brief explanation of the 
#neural network parameters chosen.
#For a detailed explanation of the complete training process, please visit 
#https://github.com/cuny-sps-msda-data622-2017fall/project-msekhar12/blob/master/MNIST%20Project.ipynb
#We will be using the following options to train the network:
#epochs=150 (Number of training episodes)
#batch_size = 1000 (Batch size for stochastic gradient descent)
#First layer nodes = 128 (Number of nodes in the first layer)
#Second layer nodes = 40 (Number of nodes in the second layer)
#Last layer nodes = 10 (Number of nodes in the final layer)
#Activation = 'relu' (Activation function)
#Dropout = 0.2 (Drop 20% of the connections randomly at each layer for each training episode.)
#Dropout will help us to avoid overtraining some connections (resulting in heavy weighted edges, resulting in biased outcomes)


def get_data():
       
        #Get data from sklearn.datasets
        mnist = fetch_mldata('MNIST original')
        
        #Prepare the independent(X) and depenednt(y) variaobes:
        X, y = mnist["data"], mnist["target"]
        
        #Split the data into train and test sets.
        #Initial 60K records will be training and the last 10K records (out of 70K records) will be test data
        X_train, X_test, y_train, y_test = X[:60000], X[60000:],y[:60000], y[60000:]

        #Set the random seed to reproduce the results
        np.random.seed(100)

        #Shffle the training data randomly, as the initial 60K data has been ordered by class (digits 0 to 9)
        observations_count = X_train.shape[0]
        shuffle_index = np.random.permutation(observations_count)
        X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

        #Scale the data using std. scaler. Fit only on training data.
        std_scaler = StandardScaler()
        std_scaler.fit(X_train)

        #One-hot encoding of target labels. Fit only on training data.
        label_binarizer = LabelBinarizer()
        label_binarizer.fit(y_train)

        #Transform the training data to standardize the data, and also binarize the training data's target variable.
        X_train_transformed = std_scaler.transform(X_train)
        y_train_transformed = label_binarizer.transform(y_train)
        
        
        #Transform the test data using the models std. scaler and binarize models developed for training data
        X_test_transformed = std_scaler.transform(X_test)
        y_test_transformed = label_binarizer.transform(y_test)
        
        return X_train_transformed, y_train_transformed, X_test_transformed, y_test_transformed


def train_model(X_train_transformed, y_train_transformed):
        #Creating the object
        model = Sequential()

        #Adding initial layer
        model.add(Dense(128, input_dim=784))

        #Adding the activation function for 1st layer's nodes
        model.add(Activation('relu'))

        #Adding dropout for edges between layer-1 and 2
        model.add(Dropout(0.2))

        #Adding second layer with 40 nodes
        model.add(Dense(40))

        #Adding the activation function for second layer's nodes
        model.add(Activation('relu'))

        #Adding dropout for edges between layer-2 and 3
        model.add(Dropout(0.2))

        #Adding the final layer with 10 nodes, as we have 10 classes to predict
        model.add(Dense(10))

        #Adding activation function of the final layer
        #This has to be softmax, to get true probabilities for the classes
        model.add(Activation('softmax'))

        #Compiling the model. Specifying the cost function, optimizer and metrics
        model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #Uncomment the following statement to display model summary
        #model.summary()

        #Starting the training
        model.fit(X_train_transformed, y_train_transformed, epochs=150, batch_size=1000, verbose=0)
        
        return model

def get_test_score(X_test_transformed, y_test_transformed,model):
        #Getting the accuracy of the model using test data
        print("Test data accuracy ",model.evaluate(X_test_transformed, y_test_transformed)[-1])
        
def main():
        #Get the test and training data:
        X_train_transformed, y_train_transformed, X_test_transformed, y_test_transformed = get_data()
        
        #Build the model
        model = train_model(X_train_transformed, y_train_transformed)
        
        #Get the test score (accuracy)
        get_test_score(X_test_transformed, y_test_transformed,model)



##Boiler plate syntax        
if __name__ == '__main__':
    main()        