import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

train_file = "../source_file/csci_data/SR-ARE-train/names_onehots.pickle"

with open (train_file,"rb") as f:
    obj = pickle.load(f)

structure,name = obj['onehots'],obj['names']

print(name)
