from loader import load_data

import numpy as np
import skimage.io
import skimage
#import cv2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


train_dir = "afhq/train"
image_num = 1000
val_split = 0.3

X_train_filenames, X_val_filenames, y_train, y_val = load_data(train_dir, image_num, val_split)

#print(X_train_filenames)
#print(X_val_filenames)
#print(y_train)
#print(y_val)

X_train = np.array([skimage.io.imread(str(file_name)) for file_name in X_train_filenames])
X_val = np.array([skimage.io.imread(str(file_name)) for file_name in X_val_filenames])
print(X_train.shape)
print(X_val.shape)

scale_percent = 50 # percent of original size
width = int(512 * scale_percent / 100)
height = int(512 * scale_percent / 100)
dim = (width, height)
# resize image
X_train = np.array([skimage.transform.rescale(image, 0.25, multichannel=True, anti_aliasing=True) for image in X_train])
X_val = np.array([skimage.transform.rescale(image, 0.25, multichannel=True, anti_aliasing=True) for image in X_val])
print(X_train.shape)
print(X_val.shape)
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
print(X_train.shape)
print(X_val.shape)

# Reverse one hot
y_train = np.argmax(y_train, axis=1)
y_val = np.argmax(y_val, axis=1)

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_val)

print('Accuracy: %.2f' % accuracy_score(y_val, y_pred))
print("Precision: %.2f" % precision_score(y_val, y_pred, average='weighted'))
print("Recall: %.2f" % recall_score(y_val, y_pred, average='weighted'))
print('Classification Report:\n', classification_report(y_val, y_pred))
