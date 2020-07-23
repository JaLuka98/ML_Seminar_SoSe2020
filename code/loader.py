import numpy as np
import os

import keras
from keras.utils import to_categorical

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import os

def load_data(train_dir, image_num, val_split):
    filenames = []
    labels = np.array([[42]])

    #train_dir = "afhq/train"


    labels_counter = -1

    for subdir, dirs, files in os.walk(train_dir):
        print("subdir: " + str(subdir))
        print("dirs: " + str(dirs))


        filenames_counter = 0

        if subdir.endswith("cat"):
            labels_counter=0
        if subdir.endswith("dog"):
            labels_counter=2
        if subdir.endswith("wild"):
            labels_counter=1

        print(labels_counter)

        for file in files:
            full_path = os.path.join(subdir, file)
            filenames.append(full_path)
            labels = np.append(labels, [[labels_counter]],axis=0)
            #labels[filenames_counter, 0] = labels_counter
            filenames_counter = filenames_counter + 1
            if filenames_counter >= (image_num-2)/3:
                break

    labels = np.delete(labels,0,0)

    y_labels_one_hot = to_categorical(labels)

    filenames_shuffled, y_labels_one_hot_shuffled = shuffle(filenames, y_labels_one_hot)

    filenames_shuffled_numpy = np.array(filenames_shuffled)

    if val_split != 0.0:
        X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(filenames_shuffled_numpy, y_labels_one_hot_shuffled, test_size=val_split, random_state=1)
        return X_train_filenames, X_val_filenames, y_train, y_val
    elif val_split==0.0:
        return filenames_shuffled_numpy, y_labels_one_hot_shuffled
