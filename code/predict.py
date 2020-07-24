import tensorflow as tf
from tensorflow import keras
from loader import load_data
from generator import Generator
import numpy as np


def makepredictions(modelpath,outputfile,training_dir):

    train_dir = training_dir
    image_num = 16000
    val_split = 0
    
    X_test_filenames, y_test = load_data(train_dir, image_num, val_split)
    
    batch_size = 11
    
    print(np.argmax(y_test,axis=1))

    test_generator = Generator(X_test_filenames, y_test, batch_size)
    
    reconstructed_model = keras.models.load_model(modelpath)
    
    prediction = reconstructed_model.predict_generator(test_generator)
    
    print(X_test_filenames)
    print(y_test)
    print(prediction)
    
    np.savetxt(outputfile,np.vstack((np.arange(len(y_test)),np.argmax(y_test,axis=1),np.array(prediction[:,0]),np.array(prediction[:,1]),np.array(prediction[:,2]))).T)

#makepredictions("logs_and_models/7layer/model0","predictions/7layer_predictions_test.txt","afhq/val")
#makepredictions("logs_and_models/7layer/model0","predictions/7layer_predictions_train.txt","afhq/train")
#makepredictions("logs_and_models/finalrun/model1","predictions/2layer_predictions_test.txt","afhq/val")
#makepredictions("logs_and_models/finalrun/model1","predictions/2layer_predictions_train.txt","afhq/train")
#makepredictions("logs_and_models/dnn/model0","predictions/dnn_predictions_train.txt","afhq/train")
#makepredictions("logs_and_models/dnn/model0","predictions/dnn_predictions_test.txt","afhq/val")


