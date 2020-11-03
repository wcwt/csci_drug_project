import numpy as np
import pickle

train_file = "../source_file/csci_data/SR-ARE-train/names_onehots.pickle"

with open (train_file,"rb") as f:
    obj = pickle.load(f)

feature,name = obj['onehots'],obj['names']

print(feature)
