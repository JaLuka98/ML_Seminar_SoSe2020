from loader import load_data
from generator import Generator

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
import matplotlib.pyplot as plt

from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras.backend import clear_session
from tensorflow.compat.v1.keras.backend import get_session

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

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

        model.add(Dense(3, activation='sigmoid'))

        self.models.append(model)

    def add_model(self, numconvlayer, numfilters, numdense, numnodes, pooling):

        model = Sequential()
        model.add(Conv2D(numfilters[0], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(512, 512, 3)))
        model.add(MaxPooling2D((pooling, pooling)))

        for i in range(1,numconvlayer):
            model.add(Conv2D(numfilters[i], (3, 3), activation='relu'))
            model.add(MaxPooling2D((pooling, pooling)))
        
        model.add(Flatten())

        for i in range(0,numdense):
            model.add(Dense(numnodes[i], activation='relu', kernel_initializer='he_uniform'))
            
        model.add(Dense(3, activation='sigmoid'))

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
            config.gpu_options.per_process_gpu_memory_fraction = 0.75
            config.gpu_options.allow_growth = True
            set_session(InteractiveSession(config=config))

    def compileall(self):
        for model in self.models:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def trainall(self, traingenerator, valgenerator, safe, safemodel):
        i = 0
        for model in self.models:
            self.text_file.write("Model " + str(i) + "\n")
            
            model.summary(print_fn=lambda x: self.text_file.write(x + '\n\n'))

            logdir = os.path.join("logs","tensorboard",str(i))
            tensorboard_callback = TensorBoard(logdir)

            start = time.time()
            history = model.fit_generator(traingenerator, steps_per_epoch=128, 
                                          validation_data=valgenerator, validation_steps=128, epochs=self.epochs, verbose=1, 
                                          callbacks=[tensorboard_callback])
            end = time.time()

            if(safe == True and i == safemodel):
                model.save("my_model")

            self.text_file.write('time for training: ' + str(end-start) + '\n')

            i+=1
            self.text_file.write("==============================\n")
            self.text_file.write("==============================\n")
            self.text_file.write("==============================\n")

            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['training', 'validation'], loc='lower right')
            #plt.show()
            #plt.savefig("model" + str(i) + ".pdf")
            plt.close()
            self.reset_keras()

        self.text_file.close()
