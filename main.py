import pickle
#import numpy as np


train_file = "SR-ARE-train/names_onehots.pickle"

with open (train_file,"rb") as f:
    obj = pickle.load(f)

structure,name = obj['onehots'],obj['names']

print(structure)