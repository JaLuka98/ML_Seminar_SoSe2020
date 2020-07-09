from loader import load_data
from generator import Generator
from testsequence import Testsequence
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

train_dir = "afhq/train"
image_num = 10000
val_split = 0.3

X_train_filenames, X_val_filenames, y_train, y_val = load_data(train_dir, image_num, val_split)

batch_size = 5

my_training_batch_generator = Generator(X_train_filenames, y_train, batch_size)
my_validation_batch_generator = Generator(X_val_filenames, y_val, batch_size)


seq = Testsequence("Test1",1)

#Alles konstant Anzahl conv layer hoch
seq.add_model(1,64,2,32,5)
seq.add_model(2,64,2,32,3)
seq.add_model(3,64,2,32,2)
seq.add_model(4,64,2,32,2)

#Alles konstant Anzahl filter variieren
seq.add_model(2,16,2,32,3)
seq.add_model(2,32,2,32,3)
seq.add_model(2,64,2,32,3)
seq.add_model(2,96,2,32,3)

# dense layer variieren
seq.add_model(2,64,1,32,3)
seq.add_model(2,64,2,32,3)
seq.add_model(2,64,3,32,3)

# Anzahl dense units
seq.add_model(2,64,2,16,3)
seq.add_model(2,64,2,32,3)
seq.add_model(2,64,2,64,3)


seq.compileall()

seq.trainall(my_training_batch_generator,my_validation_batch_generator)

