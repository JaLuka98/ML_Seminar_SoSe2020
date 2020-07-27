#import time
# See https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
# Needs to be executed in a py36 environment: https://stackoverflow.com/questions/57186629/install-opencv-with-conda (because of opencv issues)

import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

from sklearn.decomposition import PCA

import imutils
from imutils import paths
import cv2
import os
import sys

from mlxtend.plotting import plot_decision_regions
import seaborn as sns


def get_label_from_path(path):
	# Get rid of dir and extension, only filename remains
	filename = str(path).split(os.path.sep)[-1].split(".")[0]
	if "cat" in filename:
		label = 0
	elif "dog" in filename:
		label = 1
	elif "wild" in filename:
		label = 2
	else:
		print("INFO: Critical error. Image is neither cat, nor dog nor wild. Exiting...")
		sys.exit()

	return label


def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# 1: Source array, 2: channels to be used (all, since only single image at a time)
	# 3: Dont use a mask, how should the binning be
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 256, 0, 256, 0, 256])
	# Will become (8,8,8) hist
	# 256 because [0,256] actually means half open interval ,256) :/
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()


def gs_color_hists(path_get, path_store, path_labels):
	# Get and store color hist
	imagePaths = list(paths.list_images(path_get))

	X = []
	y = []

	# loop over the input images
	for (i, imagePath) in enumerate(imagePaths):
		if i%100==0: print("INFO: Processing image", i, "...")
		# load the image and extract the class label
		image = cv2.imread(imagePath)
		label = get_label_from_path(path=imagePath)
		y.append(label)
		hist = extract_color_histogram(image)
		X.append(hist)

	X = np.asarray(X)
	y = np.asarray(y)

	print("[INFO] features matrix: {:.2f}MB".format(
		X.nbytes / (1024 * 1000.0)))

	np.save(file=path_store, arr=X)
	np.save(file=path_labels, arr=y)

# Set this variable if you want to extract the color histograms from the pictures...
# ...and store them and the pca transformed data on disk.
# You can set store = False, then the data wont be saved to disk.
store = True

if store:
	gs_color_hists(path_get="../afhq/train", path_store="./color_hists_train", path_labels="./labels_train")
	gs_color_hists(path_get="../afhq/val", path_store="./color_hists_test", path_labels="./labels_test")

X = np.load(file="./color_hists_train.npy")
y = np.load(file="./labels_train.npy")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

pca = PCA(n_components=0.95, svd_solver = 'full')
pca.fit(X_train)
X_train = pca.transform(X_train)
X_val = pca.transform(X_val)
X = pca.transform(X) # for boundary regions
print(X_train.shape)
if store: np.save(file='X_pca_train', arr=X)

X_test = np.load(file="./color_hists_test.npy")
X_test = pca.transform(X_test)
if store: np.save(file='X_pca_test', arr=X_test)

neigh = KNeighborsClassifier(n_neighbors=25)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_val)

print('Accuracy: %.2f' % accuracy_score(y_val, y_pred))
print("Precision: %.2f" % precision_score(y_val, y_pred, average='weighted'))
print("Recall: %.2f" % recall_score(y_val, y_pred, average='weighted'))
print('Classification Report:\n', classification_report(y_val, y_pred))

sns.distplot(X[:,0][y==0], hist=False, label='cat')
sns.distplot(X[:,0][y==1], hist=False, label='dog')
sns.distplot(X[:,0][y==2], hist=False, label='wild')
plt.xlabel('First principal component')
plt.ylabel('KDE')
plt.savefig('kde_sns.pdf')
plt.clf()

sns.distplot(X[:,1][y==0], hist=False, label='cat')
sns.distplot(X[:,1][y==1], hist=False, label='dog')
sns.distplot(X[:,1][y==2], hist=False, label='wild')
plt.xlabel('Second principal component')
plt.ylabel('KDE')
plt.savefig('kde_sns_2.pdf')
