import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

train_file = "../source_file/csci_data/SR-ARE-train/names_onehots.pickle"

with open (train_file,"rb") as f:
    obj = pickle.load(f)

feature,name = obj['onehots'],obj['names']

batch_size = 32

"""
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
"""
