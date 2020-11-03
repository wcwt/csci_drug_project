import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

file_out = "labels.txt"
test_path = "../source_file/csci_data/SR-ARE-test/"
size = 70*325

def dataloader(path):
    pickle_in = path + "names_onehots.pickle"
    label_in = path + "names_labels.txt"
    # load data from file
    with open (pickle_in,"rb") as f:
        obj = pickle.load(f)
    feature,name = obj['onehots'],obj['names']
    label_data = np.loadtxt(label_in,dtype=str,delimiter=',')
    toxic_label = np.array(label_data[:,1],dtype=int) # get {0,1} form name
    return feature,name,toxic_label

def create_model():
        # Define Sequential model with 3 layers
        model = keras.Sequential(
            [
                layers.Flatten(input_shape = (70,325)),
                layers.Dense(size, activation="relu", name="layer1"),
                layers.Dense(32, activation="relu",name="layer2"),
                layers.Dense(2, activation="softmax", name="output"),
            ]
        # https://www.tensorflow.org/api_docs/python/tf/keras/losses for loss function
        )

        model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
        return model

model = create_model()
model.load_weights('modle/modle.w')
model.summary()
test_feature,test_name,test_toxic_label = dataloader(test_path)
for feature in test_feature:
    predict = model.predict(feature)
    print(predict)
"""

with open (file_out,"w+") as f:
    for d in predict:
        f.write(f"{int(d)}\n")
"""
