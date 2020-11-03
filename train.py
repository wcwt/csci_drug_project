import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

def create_model():
        # Define Sequential model with 3 layers
        model = keras.Sequential(
            [
                layers.Flatten(input_shape = (70,325)),
                layers.Dense(size, activation="relu", name="layer1"),
                layers.Dense(32, activation="relu",name="layer2"),
                layers.Dense(2, activation="softmax", name="output"),
            ]
        # https://www.tensorflow.org/api_docs/python/tf/keras/losses for loss function
        )
        return model

def train(train_feature,train_label,test_feature,test_label):
    model = create_model()

    model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

    model.fit(train_feature,train_label,epochs=1,shuffle=True,batch_size=10)

    test_loss,test_acc = model.evaluate(test_feature,test_label,verbose=10)

    model.save('modle.md')

def main():
    train_feature,train_name,train_toxic_label = dataloader(train_path)
    test_feature,test_name,test_toxic_label = dataloader(test_path)
    train(train_feature[:10],train_toxic_label[:10],test_feature,test_toxic_label)

if __name__ == "__main__":
    main()
