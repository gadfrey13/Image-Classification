import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.metrics import Accuracy
acc = Accuracy()
network = input_data(shape=[None, 300, 300, 3])
# Conv layers ------------------------------------ Alexnet
network = conv_2d(network, 64, 3, strides=1, activation='relu')
network = max_pool_2d(network, 2, strides=2)#Cuts the dimension in half
network = conv_2d(network, 64, 3, strides=1, activation='relu')
network = max_pool_2d(network, 2, strides=2)#Cuts the dimension in half
network = conv_2d(network, 64, 3, strides=1, activation='relu')
network = conv_2d(network, 64, 3, strides=1, activation='relu')
network = conv_2d(network, 64, 3, strides=1, activation='relu')
network = max_pool_2d(network, 2, strides=2)#Cuts the dimension in half
# Fully Connected Layers -------------------------
network = fully_connected(network, 1024, activation='tanh')#Number of neurons is 1024
network = dropout(network, 0.5)#Randomly remove half of the neurons
network = fully_connected(network, 1024, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='momentum',
 loss='categorical_crossentropy', learning_rate=0.001, metric=acc)
model = tflearn.DNN(network,tensorboard_verbose=3,tensorboard_dir="logs")