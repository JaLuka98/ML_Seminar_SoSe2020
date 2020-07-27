from loader import load_data
from generator import Generator

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import matplotlib.pyplot as plt

from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras.backend import clear_session
from tensorflow.compat.v1.keras.backend import get_session

from tensorflow.keras import regularizers

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import numpy as np

import gc

from tensorflow.keras.callbacks import TensorBoard

import time

import os

class Testsequence:

    def __init__(self, testname, epochs):
        self.models = []
        self.text_file = open("test_" + testname + ".log", "w")
        self.text_file.write('Log file for test ' + testname + '\n')
        self.epochs = epochs

    def add_dnn(self, numdense, units,pooling):
        
        model = Sequential()
        model.add(MaxPooling2D((pooling,pooling),input_shape=(512,512,3)))
        model.add(Flatten())
        for i in range(0,numdense):
            model.add(Dense(units[i], activation='relu'))

        model.add(Dense(3, activation='softmax'))

        self.models.append(model)
    # numconvlayer: Anzahl convolutional layer (int)
    # numfilters: Anzahl filter pro conv layer (array int)
    # numdense: Anzahl dense layer (int)
    # numnodes: Anzahl unit pro dense layer (array int)
    # pooling: pooling size
    # dopout: Dropout nach jedem conv und dense layer (array double) dorpout = 0 für kein dropout
    # lam1: lambda für l1 regularisierung
    # lam2: lambda für l2 regularisierung
    # lastreglayers: anzahl der layer die mit l1 und l2 regularisiert werden (von hinten gezählt)
    def add_model(self, numconvlayer, numfilters, numdense, numnodes, pooling, dropout, lam1, lam2, lastreglayers):

        model = Sequential()
        model.add(Conv2D(numfilters[0], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(512, 512, 3)))
        model.add(MaxPooling2D((pooling, pooling)))

        for i in range(1,numconvlayer):
            if(i >= lastreglayers):
                model.add(Conv2D(numfilters[i], (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=lam1,l2=lam2)))
            else:
                model.add(Conv2D(numfilters[i], (3, 3), activation='relu'))
            if not dropout[i] == 0:
                model.add(Dropout(dropout[i]))
            model.add(MaxPooling2D((pooling, pooling)))
        
        model.add(Flatten())

        for i in range(0,numdense):
            if(i >= lastreglayers):
                model.add(Dense(numnodes[i], activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l1_l2(l1=lam1,l2=lam2)))
            else:
                model.add(Dense(numnodes[i], activation='relu', kernel_initializer='he_uniform'))
            if not dropout[i+numconvlayer] == 0:
                model.add(Dropout(dropout[i]))
            
        model.add(Dense(3, activation='softmax'))

        self.models.append(model)

    def summary(self, model):
        self.models[model].summary()

    def reset_keras(self):
            sess = get_session()
            clear_session()
            sess.close()
            sess = get_session()
            
            try:
                del classifier 
            except:
                pass

            print(gc.collect())
            
            # use the same config as you used to create the session
            config = ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            config.gpu_options.allow_growth = True
            set_session(InteractiveSession(config=config))

    def compileall(self):
        for model in self.models:
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def trainall(self, traingenerator, valgenerator, safe):
        i = 0
        for model in self.models:
            self.text_file.write("Model " + str(i) + "\n")
            
            model.summary(print_fn=lambda x: self.text_file.write(x + '\n\n'))

            logdir = os.path.join("logs","tensorboard",str(i))
            tensorboard_callback = TensorBoard(logdir)

            start = time.time()
            history = model.fit_generator(traingenerator, steps_per_epoch=300, 
                                          validation_data=valgenerator, validation_steps=300, epochs=self.epochs, verbose=1, 
                                          callbacks=[tensorboard_callback])
            end = time.time()

            if(safe == True):
                model.save("model"+str(i))

            self.text_file.write('time for training: ' + str(end-start) + '\n')


            self.text_file.write("==============================\n")
            self.text_file.write("==============================\n")
            self.text_file.write("==============================\n")

            np.savetxt(str(i)+"history_loss.txt",np.vstack((np.array(history.history['accuracy']),np.array(history.history['val_accuracy']),history.history['loss'],history.history['val_loss'])).T)

            i+=1
            self.reset_keras()

        self.text_file.close()
