import numpy as np
import pickle

train_file = "../source_file/csci_data/SR-ARE-train/names_onehots.pickle"

with open (train_file,"rb") as f:
    obj = pickle.load(f)

feature,name = obj['onehots'],obj['names']

#print(feature[0])
pos = 0
neg = 0
for label in name:
    if label[-1] == '1':
        pos += 1
    else:
        print(label)
