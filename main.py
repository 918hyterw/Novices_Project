import os,numpy,tensorflow
import numpy as np
from tensorflow import keras
from keras import layers
import sklearn

from data_load import *
from train import *
from Prediction import *
from Evolution_result import *
from model import *
from display_image import *

# main 
if __name__ == "__main__":
    # data path
    non_nude_data_path_1 = r"D:\PROJECT\non_nude_dataset.npz"
    non_nude_data_path_2 = r"D:\PROJECT\non_nude_dataset_1.npz"
    nude_data_path = r"D:\PROJECT\nude_dataset.npz"

    # load data
    data = load_data(non_nude_data_path_1,non_nude_data_path_2,nude_data_path)
    print("\nShape of Image data and label data: ",np.array(data[0]).shape, np.array(data[1]).shape)

    # show one image
    show(data[0][0])

    # model
    model_path = make_model()

    # train the model
    model = train(model_path,data,epochs=2,batch_size=32,optimizer=keras.optimizers.Adam(1e-3),loss="binary_crossentropy")

    # accuracy check
    evolutation_function(model_path,data[0],data[1])