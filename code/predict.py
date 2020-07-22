import tensorflow as tf
from tensorflow import keras
from loader import load_data
from generator import Generator
import numpy as np


def makepredictions(modelpath,outputfile):

    train_dir = "afhq/val"
    image_num = 1500
    val_split = 0
    
    X_test_filenames, y_test = load_data(train_dir, image_num, val_split)
    
    batch_size = 11
    
    print(np.argmax(y_test,axis=1))
    
    test_generator = Generator(X_test_filenames, y_test, batch_size)
    
    reconstructed_model = keras.models.load_model(modelpath)
    
    prediction = reconstructed_model.predict_generator(test_generator)
    
    np.savetxt(outputfile,np.vstack((np.arange(1500),np.argmax(y_test,axis=1),np.array(prediction[:,0]),np.array(prediction[:,1]),np.array(prediction[:,2]))).T)

makepredictions("logs_and_models/7layer/model0","7layer_predictions.txt")
makepredictions("logs_and_models/finalrun/model1","2layer_predictions.txt")
