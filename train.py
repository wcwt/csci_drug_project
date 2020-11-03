import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

import numpy as np
import pickle

path = "../source_file/csci_data/SR-ARE-train/"
pickle_in = path + "names_onehots.pickle"
label_in = path + "names_labels.txt"

def dataloader():
    # load data part
    with open (pickle_in,"rb") as f:
        obj = pickle.load(f)
    feature,name = obj['onehots'],obj['names']
    label_data = np.loadtxt(label_in,dtype=str,delimiter=',')
    toxic_label = label_data[:,0] # get {0,1} form name
    return feature,name,toxic_label

def main():
    feature,name,toxic_label = dataloader()
    print(feature[0])


if __name__ == "__main__":
    main()
