import numpy as np
import pickle

train_path = "../source_file/csci_data/SR-ARE-train/"
test_path = "../source_file/csci_data/SR-ARE-test/"

size = 70*325

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

train_feature,train_name,train_toxic_label = dataloader("train")
test_feature,test_name,test_toxic_label = dataloader("test")

print(train_toxic_label.sum()/len(train_toxic_label))
