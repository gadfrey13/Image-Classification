import deepneuralnet as net
import numpy as np
from tflearn.data_utils import image_preloader
model = net.model
path_to_model = './ZtrainedNet/final-model.tfl'
model.load(path_to_model)#Loads the save training deep neural network
#X is data to feed to train model
#Y is Targets (labels) to feed to train model.
X, Y = image_preloader(target_path='./validate', image_shape=(300,300), mode='folder',
 grayscale=False, categorical_labels=True, normalize=True)
X = np.reshape(X, (-1, 300, 300, 3))#Guest this change the X and Y into buffers
for i in range(0, len(X)):
 iimage = X[i]#Image
 icateg = Y[i]#Category correspond to the image
 result = model.predict([iimage])[0]#Returns the predicted probabilities. What is this for [0]
 prediction = result.tolist().index(max(result))#Grab the bigger of the two values 
 reality = icateg.tolist().index(max(icateg))
 if prediction == reality:
     print("image %d CORRECT " % i, end='')
 else:
     print("image %d WRONG " % i, end='')
 print(result)
 
 #Loss equals error
 #Accuracy how much of the training data it has gotten correct
 #Looks at testing set
 #