'''
Created on Nov 14, 2017

@author: Gad Frey
'''
import deepneuralnet as net
import numpy as np
from tflearn.data_utils import image_preloader
from tflearn.data_flow import ArrayFlow
model = net.model
#X is data to train model. 
#Y is the category
X, Y = image_preloader(target_path='./train', image_shape=(300, 300),
 mode='folder', grayscale=False, categorical_labels=True, normalize=True)#Loads the image file to variables X
X = np.reshape(X, (-1, 300, 300, 3))#Reshape the array like structure
#Note. Question does this only reshape the  
#W is data to train model.
W, Z = image_preloader(target_path='./validate', image_shape=(300, 300),
 mode='folder', grayscale=False, categorical_labels=True, normalize=True)
W = np.reshape(W, (-1, 300, 300, 3))
model.fit(X, Y, n_epoch=50, validation_set=(W,Z), show_metric=True,batch_size=10)#This where the data is sent to the deep neural net
model.save('./ZtrainedNet/final-model.tfl')