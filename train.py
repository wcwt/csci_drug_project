import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

path = "../source_file/csci_data/SR-ARE-train/"
pickle_in = path + "names_onehots.pickle"
label_in = path + "names_labels.txt"

def dataloader():
    # load data from file
    with open (pickle_in,"rb") as f:
        obj = pickle.load(f)
    feature,name = obj['onehots'],obj['names']
    label_data = np.loadtxt(label_in,dtype=str,delimiter=',')
    toxic_label = label_data[:,0] # get {0,1} form name
    return feature,name,toxic_label

def main():
    feature,name,toxic_label = dataloader()
    # Define Sequential model with 3 layers
    model = keras.Sequential(
        [
            layers.Dense(2, activation="relu", name="layer1"),
            layers.Dense(3, activation="relu", name="layer2"),
            layers.Dense(4, name="layer3"),
        ]
    )
    # Call model on a test input
    x = tf.ones((3, 3))
    y = model(x)


if __name__ == "__main__":
    main()
