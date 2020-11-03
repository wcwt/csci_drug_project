import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

train_path = "../source_file/csci_data/SR-ARE-train/"
test_path = "../source_file/csci_data/SR-ARE-test/"

def dataloader(mode = "train"):
    path = train_path if mode == "train" else test_path
    pickle_in = path + "names_onehots.pickle"
    label_in = path + "names_labels.txt"
    # load data from file
    with open (pickle_in,"rb") as f:
        obj = pickle.load(f)
    feature,name = obj['onehots'],obj['names']
    label_data = np.loadtxt(label_in,dtype=str,delimiter=',')
    toxic_label = np.array(label_data[:,1],dtype=int) # get {0,1} form name
    return feature,name,toxic_label

model = create_model()

# Restore the weights
model.load_weights('./')

test_feature,test_label = dataloader("test")
# Evaluate the model
loss,acc = model.evaluate(test_feature, test_label, verbose=2)
