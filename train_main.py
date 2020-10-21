import pickle
import numpy as np
import torch

train_file = "../source_file/csci_data/SR-ARE-train/names_onehots.pickle"

with open (train_file,"rb") as f:
    obj = pickle.load(f)

structure,name = obj['onehots'],obj['names']

#print(len(structure))
#print(len(name))

print(structure[3])
