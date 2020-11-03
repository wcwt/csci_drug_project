import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

def train(train_feature,train_label,test_feature,test_label):
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
    model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])

    model.fit(train_feature,train_label,epochs=10,shuffle=True)

    test_loss,test_acc = model.evaluate(test_feature,test_label,verbose=10)
    print(f"loss = {test_loss}, acc = {test_acc}")
    with open("model.pk","wb+") as f
        pickle.dump(model,f)

def main():
    train_feature,train_name,train_toxic_label = dataloader("train")
    test_feature,test_name,test_toxic_label = dataloader("test")
    train(train_feature,train_toxic_label,test_feature,test_toxic_label)

if __name__ == "__main__":
    main()
