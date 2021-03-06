import tensorflow as tf
from tensorflow import keras
from loader import load_data
from generator import Generator
import numpy as np
import matplotlib.pyplot as plt

# Erstell Dateien mit dem Output des Netzwerkes bei modelpath auf den daten in training_dir
def makepredictions(modelpath,outputfile,training_dir):

    train_dir = training_dir
    image_num = 16000
    val_split = 0
    
    X_test_filenames, y_test = load_data(train_dir, image_num, val_split)
    
    batch_size = 11
    
    print(np.argmax(y_test,axis=1))

    test_generator = Generator(X_test_filenames, y_test, batch_size)
    
    reconstructed_model = keras.models.load_model(modelpath)
    
    reconstructed_model.summary()

    prediction = reconstructed_model.predict_generator(test_generator)

    ## drei größten fehler von dogs in wildlife
    #    label = np.argmax(y_test,axis=1)
    #    wildlife = prediction[:,1]
    #    wilddogs = wildlife[label==2]
    #    dogfiles = X_test_filenames[label==2]
    #    indices = wilddogs.argsort()[-3:][::-1]
    #    print(indices)
    #    print(wilddogs[indices])
    #    print(dogfiles[indices])
    
    np.savetxt(outputfile,np.vstack((np.arange(len(y_test)),np.argmax(y_test,axis=1),np.array(prediction[:,0]),np.array(prediction[:,1]),np.array(prediction[:,2]))).T)

makepredictions("logs_and_models/model1","predictions/7layer_predictions_test.txt","afhq/val")
makepredictions("logs_and_models/ergebnisse/model1","predictions/7layer_predictions_train.txt","afhq/train")
makepredictions("logs_and_models/ergebnisse/model2","predictions/2layer_predictions_test.txt","afhq/val")
makepredictions("logs_and_models/ergebnisse/model2","predictions/2layer_predictions_train.txt","afhq/train")
makepredictions("logs_and_models/ergebnisse/model3","predictions/dnn_predictions_train.txt","afhq/train")
makepredictions("logs_and_models/ergebnisse/model3","predictions/dnn_predictions_test.txt","afhq/val")


