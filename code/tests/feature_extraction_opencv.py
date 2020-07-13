#import time
# See https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
# Needs to be executed in a py36 environment: https://stackoverflow.com/questions/57186629/install-opencv-with-conda (because of opencv issues)

import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
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
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# 1: Source array, 2: channels to be used (all, since only single image at a time)
	# 3: Dont use a mask, how should the binning be
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	# Will become (8,8,8) hist
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()



#print("[INFO] describing images...")
imagePaths = list(paths.list_images("../afhq/train"))

X = []
y = []

random.shuffle(imagePaths) # inplace shuffling to get examples of all classes

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	if i%100==0: print("INFO: Processing image", i, "...")
	# load the image and extract the class label
	image = cv2.imread(imagePath)
	label = get_label_from_path(path=imagePath)
	y.append(label)
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
	#pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	X.append(hist)

X = np.asarray(X)
y = np.asarray(y)

print("[INFO] features matrix: {:.2f}MB".format(
	X.nbytes / (1024 * 1000.0)))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

pca = PCA(n_components=0.95, svd_solver = 'full')
pca.fit(X_train)
X_train = pca.transform(X_train)
X_val = pca.transform(X_val)
X = pca.transform(X) # for boundary regions
print(X_train.shape)
#
#print(X_train[0])
#
#colors = ['navy', 'turquoise', 'red']
#lw = 2
#
## funzt noch nicht
#for color, i, target_name in zip(colors, [0, 1, 2], ['cat','dog','wild']):
#    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], c=color, alpha=.8, lw=lw,
#                label=target_name)
#plt.legend(loc='best', shadow=False, scatterpoints=1)
#plt.title('PCA of afhq color histograms dataset')
#plt.show()


neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_val)

print('Accuracy: %.2f' % accuracy_score(y_val, y_pred))
print("Precision: %.2f" % precision_score(y_val, y_pred, average='weighted'))
print("Recall: %.2f" % recall_score(y_val, y_pred, average='weighted'))
print('Classification Report:\n', classification_report(y_val, y_pred))

sns.distplot(X[:,0][y==0], hist=False, label='cat')
sns.distplot(X[:,0][y==1], hist=False, label='dog')
sns.distplot(X[:,0][y==2], hist=False, label='wild')
plt.savefig('kde_sns_test.pdf')
plt.clf()

sns.distplot(X[:,1][y==0], hist=False, label='cat')
sns.distplot(X[:,1][y==1], hist=False, label='dog')
sns.distplot(X[:,1][y==2], hist=False, label='wild')
plt.savefig('kde_sns_test2.pdf')

#plot_decision_regions(X=X[0:200,:], y=y[0:200], clf=neigh, legend=2)
#plt.show()

#img = cv2.imread('../afhq/train/cat/flickr_cat_000015.jpg')
#print(img.shape)

#hist = extract_color_histogram(img)

#plt.figure(figsize=(4.2, 4))
#for i in range(0,max_patches):
#    plt.subplot(9, 9, i + 1)
#    plt.imshow(data[i,:,:].reshape(patch_size), cmap=plt.cm.gray, interpolation='nearest')
#    plt.xticks(())
#    plt.yticks(())
#    plt.savefig('patches_test3.pdf')
#
#plt.clf()
#plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
#plt.savefig('cat3.pdf')
