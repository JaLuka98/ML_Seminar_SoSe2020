#import time

import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread

from sklearn import datasets
#from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d

rng = np.random.RandomState(0)

patch_size = (50, 50)
max_patches = 25

img = imread('./afhq/train/dog/flickr_dog_000020.jpg', as_gray=True)

data = extract_patches_2d(img, patch_size, max_patches, random_state=rng)

plt.figure(figsize=(4.2, 4))
for i in range(0,max_patches):
    plt.subplot(9, 9, i + 1)
    plt.imshow(data[i,:,:].reshape(patch_size), cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.savefig('patches_test3.pdf')

plt.clf()
plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
plt.savefig('cat3.pdf')
