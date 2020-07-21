import numpy as np
from loader import load_data
import matplotlib.pyplot as plt


#########################################
######### 1) Input Data Plots ###########
#########################################

image_num = 16000
val_split = 0.3
train_dir = "afhq/train"

X_train_filenames, X_val_filenames, y_train, y_val = load_data(train_dir, image_num, val_split)

X_test, Xempyt, y_test, yempty = load_data("afhq/val",1555,0)


print(y_train.shape)
print(y_val.shape)
print(y_test.shape)
print(np.argmax(y_train,axis=1))

y_train = np.argmax(y_train,axis=1)
y_val = np.argmax(y_val,axis=1)
y_test = np.argmax(y_test,axis=1)

print(np.bincount(y_train))
print(np.bincount(y_val))
print(np.bincount(y_test))

y_train = np.bincount(y_train)
y_val = np.bincount(y_val)
y_test = np.bincount(y_test)

labels = ['cat','dog','wildlife']

plt.subplots_adjust(wspace=0.5,hspace=1)

plt.subplot(131)
plt.title('Trainings Datensatz')
plt.bar(labels,y_train,color=['red','green','blue'])

plt.subplot(132)
plt.title('Validierungs Datensatz')
plt.bar(labels,y_val,color=['red','green','blue'])

plt.subplot(133)
plt.title('Test Datensatz')
plt.bar(labels,y_test,color=['red','green','blue'])

#plt.show()
plt.close()



#########################################
######### 2) Loss und Metric ############
#########################################

##### Tensorboard chart download ?

# .csv von tensorboard downloaden
#w, s, acc = np.genfromtxt(".csv",delimiter=",",unpack=True)
#
#print(s)
#print(acc)
#
#plt.plot(s,acc,"b-")
#plt.ylabel("Accuracy")
#plt.xlabel("Epoch")
#plt.show()



#########################################
############# 3) NN output ##############
#########################################

import tensorflow as tf
from tensorflow import keras
from loader import load_data
from generator import Generator


def makepredictions(modelpath,outputfile):

    train_dir = "afhq/val"
    image_num = 1500
    val_split = 0
    
    X_test_filenames, Xempty, y_test, yempty = load_data(train_dir, image_num, val_split)
    
    batch_size = 11
    
    print(np.argmax(y_test,axis=1))
    
    test_generator = Generator(X_test_filenames, y_test, batch_size)
    
    reconstructed_model = keras.models.load_model(modelpath)
    
    prediction = reconstructed_model.predict_generator(test_generator)
    
    np.savetxt(outputfile,np.vstack((np.arange(1500),np.argmax(y_test,axis=1),np.array(prediction[:,0]),np.array(prediction[:,1]),np.array(prediction[:,2]))).T)

makepredictions("model0","predictions.txt")
