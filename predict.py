import glob
import os
import pandas as pd
import numpy as np
from keras.models import model_from_json, model_to_json

mycsvdir = '../ToPredict/BMAS/'

csvfiles = glob.glob(os.path.join(mycsvdir, '*.csv'))

# loop through the files and read them in with pandas
dataframes = []  # a list to hold all the individual pandas DataFrames
for csvfile in csvfiles:
    df = pd.read_csv(csvfile, header=None)
    dataframes.append(df)   

# # concatenate them all together
data = np.concatenate(dataframes, axis=0)
#data = np.array(data, dtype=float)
# dataNoHeader = np.array(data[1:,:], dtype=float) #get rid of header row
# filtered = dataNoHeader[dataNoHeader[:,-1] != 0]
x = data




# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.predict(x, verbose=0)


score = [np.argmax(a, axis=None, out=None) for a in score]


with open('results.csv', 'r+') as f:
    text = f.read()
    finalData = np.hstack(text, score)
    f.write(finalData)
    f.truncate()
