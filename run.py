import numpy as np
import pickle
import os

file_out = "labels.txt"
train_path = "../source_file/csci_data/SR-ARE-train/"
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

train_feature,train_name,train_toxic_label = dataloader(train_path)
test_feature,test_name,test_toxic_label = dataloader(test_path)

l = []
for i in range(len(test_feature)-1):
    print(test_feature[i].shape) 
