import numpy as np
import os

import tensorboard_to_csv
from loader import load_data

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
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
# We dont need to worry, though, because matplotlib ignores nans

plt.subplots_adjust(wspace=0.5,hspace=1)

ylim_acc = [0.75,1.02]
ylim_loss = [-0.025,0.475]

plt.subplot(221)
plt.title('Two Layers')
plt.plot(epoch_two, acc_train_two, label='train')
plt.plot(epoch_two, acc_val_two, label='val')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(ylim_acc)
plt.grid()

plt.subplot(222)
plt.title('Seven Layers')
plt.plot(epoch_seven, acc_train_seven, label='train')
plt.plot(epoch_seven, acc_val_seven, label='val')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(ylim_acc)
plt.grid()

plt.subplot(223)
plt.title('Two Layers')
plt.plot(epoch_two, loss_train_two, label='train')
plt.plot(epoch_two, loss_val_two, label='val')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(ylim_loss)
plt.grid()

plt.subplot(224)
plt.title('Seven Layers')
plt.plot(epoch_seven, loss_train_seven, label='train')
plt.plot(epoch_seven, loss_val_seven, label='val')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(ylim_loss)
plt.grid()

#plt.tight_layout() # Ist das besser oder schlechter so?
#plt.show()
plt.clf()


#########################################
########### 3) Network output ###########
#########################################

i, label, cat, dog, wildlife = np.genfromtxt('7layer_predictions.txt',unpack=True)

print(wildlife[label==0][0:30])


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
#plt.show()
plt.clf()

#########################################
########### 4) Overtraining check #######
#########################################


###



#########################################
########### 5) Performance Plot #########
#########################################

import itertools
import matplotlib.cm as cm

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# with confusion matrix
#confusion_matrix(y_true, y_pred)
