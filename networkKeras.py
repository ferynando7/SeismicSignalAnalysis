#1.  (Input) -> [batch_size, 24, 1, 1] >> Apply 32 filter of [5x5]
#2.  (Convolutional layer 1) -> [batch_size, 24, 1, 32]
#3.  (ReLU 1) -> [?, 24, 1, 32]
#4.  (Max pooling 1) -> [?, 14, 1, 32]
#5.  (Convolutional layer 2) -> [?, 14, 1, 64]
#6.  (ReLU 2) -> [?, 14, 1, 64]
#7.  (Max pooling 2) -> [?, 7, 1, 64]
#8.  [fully connected layer 3] -> [1x150]
#9.  [ReLU 3] -> [1x150]
#10. [Drop out] -> [1x150] (optional)
#11. [fully connected layer 4] -> [1x10]



#To manage csv files
import pandas as pd
from math import floor
import glob
import os
import numpy as np
from datetime import datetime
from sklearn.utils import class_weight
#For tensorboard usage
from keras import callbacks

#This is simply a linear stack of neural network layers, and it's perfect for the type of feed-forward CNN we're building.
from keras.models import Sequential

#Next, let's import the "core" layers from Keras. These are the layers that are used in almost any neural network
from keras.layers import Dense, Dropout, Activation, Flatten

#Then, we'll import the CNN layers from Keras. These are the convolutional layers that will help us efficiently train on seismic data
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D

#Finally, we'll import some utilities. This will help us transform our data later
from keras.utils import np_utils, to_categorical, plot_model

from keras.regularizers import l2


#width = 4504 # We expect data with 24 colums for input and 1 for output
#Prepare data
mycsvdir = '../Windowed/BMAS/'

csvfiles = glob.glob(os.path.join(mycsvdir, '*.csv'))

# loop through the files and read them in with pandas
dataframes = []  # a list to hold all the individual pandas DataFrames
for csvfile in csvfiles:
    df = pd.read_csv(csvfile, header=None)
    dataframes.append(df)   

for csvfile in csvfiles:
    df = pd.read_csv(csvfile, header=None)
    dataframes.append(df)   

for csvfile in csvfiles:
    df = pd.read_csv(csvfile, header=None)
    dataframes.append(df)   

for csvfile in csvfiles:
    df = pd.read_csv(csvfile, header=None)
    dataframes.append(df)   


# # concatenate them all together
data = np.concatenate(dataframes, axis=0)
#data = np.array(data, dtype=float)
# dataNoHeader = np.array(data[1:,:], dtype=float) #get rid of header row
# filtered = dataNoHeader[dataNoHeader[:,-1] != 0]
filtered = data
np.random.shuffle(filtered)
ncols = len(filtered[0])
nrows = len(filtered)
width = ncols-1
train_lenght = floor(0.9*nrows)
test_lenght = nrows - train_lenght
x = filtered[:,:(ncols-1)]
y = np.reshape(filtered[:,-1],(-1,1)) 



#toTest = dataNoHeader[dataNoHeader[:,-1] != 0]

x_train = x[:train_lenght,:].reshape((train_lenght,width,1))
x_test = x[train_lenght:,:].reshape((test_lenght,width,1))
#x_test = toTest[:,:(ncols-1)].reshape((len(toTest),width,1))

y_train = to_categorical(y[:train_lenght,:],num_classes=9) #.reshape((1,train_lenght,1))
y_test = to_categorical(y[train_lenght:,:],num_classes=9)  #.reshape((1,test_lenght,1))
#y_test = to_categorical(np.reshape(toTest[:,-1],(-1,1)), num_classes=9) 


logdirFit = "tensorboard/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logdirEval = "tensorboard/logs/eval/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = callbacks.TensorBoard(log_dir=logdirFit, histogram_freq=1)
evaluate_callback = callbacks.TensorBoard(log_dir=logdirEval, histogram_freq=1)

#Define model architecture

model = Sequential()



######### CONVOLUTIONAL LAYER 1 and RELU

#, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)
model.add(Conv1D(16, 7, activation='relu', input_shape=(width,1)))
# , kernel_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)
# ######### MAX POOLING 1
model.add(MaxPooling1D(3))
#model.add(Dropout(0.3))
# # ######### CONVOLUTIONAL LAYER 2 and RELU
model.add(Conv1D(32,7, activation='relu'))

# # ######### MAX POOLING 2
#model.add(GlobalAveragePooling1D())
model.add(MaxPooling1D(3))
#model.add(Dropout(0.3))

# # ######### CONVOLUTIONAL LAYER 3 and RELU
model.add(Conv1D(32,7, activation='relu'))

# # ######### MAX POOLING 3
model.add(MaxPooling1D(3))
#model.add(Dropout(0.6))
######### FULLY CONNECTED LAYER 3

# Flattening last layer
model.add(Flatten())


# ######### RELU 3
model.add(Dense(200, activation='relu', input_shape=(width,)))

# ######### SOFTMAX LAYER
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))

print(model.summary())

#Now all we need to do is define the loss function and the optimizer, and then we'll be ready to train it.

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


#To fit the model, all we have to do is declare the batch size and number of epochs to train for, then pass in our training data.
classes = np.transpose(y)[0]
class_weights = class_weight.compute_class_weight('balanced'
                                               ,np.unique(classes)
                                               ,classes)

print(class_weights)
training_history = model.fit(
    x_train, 
    y_train, 
    batch_size=128, 
    nb_epoch=50, 
    verbose=2,
    shuffle=True,
    validation_split=0.2,
#    validation_data=(x_test,y_test),
    class_weight=class_weights,
    callbacks=[tensorboard_callback])

 	
score = model.evaluate(
    x_test, 
    y_test, 
    verbose=1)
#print(model.metrics_names)
print(score)




# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")