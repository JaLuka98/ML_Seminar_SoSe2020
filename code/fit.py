from loader import load_data
from generator import Generator

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(512, 512, 3)))
#model.add(Conv2D(512, (3, 3), activation='relu', input_shape=(512, 512, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(3, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

train_dir = "afhq/train"
image_num = 10000
val_split = 0.3

X_train_filenames, X_val_filenames, y_train, y_val = load_data(train_dir, image_num, val_split)

print(y_val)
print(y_train)

batch_size = 16

my_training_batch_generator = Generator(X_train_filenames, y_train, batch_size)
my_validation_batch_generator = Generator(X_val_filenames, y_val, batch_size)

history = model.fit_generator(my_training_batch_generator, steps_per_epoch=128,
	validation_data=my_validation_batch_generator, validation_steps=128, epochs=30, verbose=1)
