import numpy as np
import os

import tensorboard_to_csv
from loader import load_data

import matplotlib.pyplot as plt
#########################################
######### 1) Input Data Plots ###########
#########################################

image_num = 16000
val_split = 0.3
train_dir = "afhq/train"

X_train_filenames, X_val_filenames, y_train, y_val = load_data(train_dir, image_num, val_split)

X_test, y_test= load_data("afhq/val",1555,0.0)


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


plt.savefig('plots/barplot_input.pdf')
plt.clf()

#########################################
###### 2) Loss und Metric vs Epoch ######
#########################################

# Get the raw tensorboard logs from these paths
paths = ['logs_and_models/finalrun/logs/tensorboard/1',
         'logs_and_models/finalrun/logs/tensorboard/0',
         'logs_and_models/7layer/logs/tensorboard/0']

# Convert the tensorboard data to csv if it does not exist already
for path in paths:
    if not os.path.exists(path+'/csv'):
        tensorboard_to_csv.to_csv(dpath=path)

# Load the csv and make numpy arrays for the histories to plot them
# Note: x-axis is same for all plots, so it does not matter from where we import the epoch numbers
# First: 2 Layer. Note: csv behaves a little strange, this is why we have this clumsy syntax. But it works!
epoch_two, acc_train_two, acc_val_two = np.split(ary=np.genfromtxt(paths[0]+'/csv/epoch_accuracy.csv', delimiter=','), indices_or_sections=3, axis=-1)
epoch_two, loss_train_two, loss_val_two = np.split(ary=np.genfromtxt(paths[0]+'/csv/epoch_loss.csv', delimiter=','), indices_or_sections=3, axis=-1)
# Now: 5 Layer
epoch_five, acc_train_five, acc_val_five = np.split(ary=np.genfromtxt(paths[1]+'/csv/epoch_accuracy.csv', delimiter=','), indices_or_sections=3, axis=-1)
epoch_five, loss_train_five, loss_val_five = np.split(ary=np.genfromtxt(paths[1]+'/csv/epoch_loss.csv', delimiter=','), indices_or_sections=3, axis=-1)
# Finally: 7 Layer
epoch_seven, acc_train_seven, acc_val_seven = np.split(ary=np.genfromtxt(paths[2]+'/csv/epoch_accuracy.csv', delimiter=','), indices_or_sections=3, axis=-1)
epoch_seven, loss_train_seven, loss_val_seven = np.split(ary=np.genfromtxt(paths[2]+'/csv/epoch_loss.csv', delimiter=','), indices_or_sections=3, axis=-1)

# Sadly, in the csv produced the top row is not commented out, resulting in nans
# We need to manually remove them
#epoch_two = epoch_two[1:], epoch_five = epoch_five[1:], epoch_seven = epoch_seven[1:]


#########################################
########### 3) Network output ###########
#########################################

i, label, cat, dog, wildlife = np.genfromtxt('7layer_predictions.txt',unpack=True)


plt.subplots_adjust(wspace=0.5,hspace=0.5)

### Reihenfolg wild=0 - dog=1 - cat=2

plt.subplot(131)
plt.hist(cat[label==2],label="Cat prediction",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.hist(cat[label==1],label="Dog prediction",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.hist(cat[label==0],label="Wildlife prediction",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

plt.subplot(132)
plt.hist(dog[label==2],label="Cat prediction",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.hist(dog[label==1],label="Dog prediction",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.hist(dog[label==0],label="Wildlife prediction",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

plt.subplot(133)
plt.hist(wildlife[label==2],label="Cat prediction",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.hist(wildlife[label==1],label="Dog prediction",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.hist(wildlife[label==0],label="Wildlife prediction",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])


plt.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
plt.legend(bbox_to_anchor=(-0.1,1))
plt.show()

