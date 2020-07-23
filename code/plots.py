import numpy as np
import os

import tensorboard_to_csv
from loader import load_data

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# For plot_confusion_matrix
import itertools
import matplotlib.cm as cm

# Acknowledgement to Olaf Nackenhorst Exercise 4 ML Seminar summer term 2020, TU Dortmund
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          colorbar=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    if colorbar: plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib 3.1.1 broke inverted axes with fixed ticks :(
    # Maybe this is not necessary in other version. Check it at the end!
    plt.ylim([2.5,-0.5])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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
plt.savefig('plots/acc_and_loss.pdf')
plt.clf()


#########################################
########### 3) Network output ###########
#########################################

i, label, cat, wildlife, dog = np.genfromtxt('7layer_predictions_test.txt',unpack=True)

## wildlife = 1
## cat = 0
## dog = 2

plt.subplots_adjust(wspace=0.5,hspace=0.5)

plt.subplot(131)
plt.title("Cat output node")
plt.hist(cat[label==0],label="Cat images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.hist(cat[label==2],label="Dog images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.hist(cat[label==1],label="Wildlife images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

plt.subplot(132)
plt.title("Dog output node")
plt.hist(dog[label==0],label="Cat images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.hist(dog[label==2],label="Dog images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.hist(dog[label==1],label="Wildlife images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

plt.subplot(133)
plt.title("Wildlife output node")
plt.hist(wildlife[label==0],label="Cat images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.hist(wildlife[label==2],label="Dog images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.hist(wildlife[label==1],label="Wildlife images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])


plt.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
plt.legend(bbox_to_anchor=(-0.1,1))
plt.savefig("plots/Network-output.pdf")
plt.clf()

#########################################
########### 4) Overtraining check #######
#########################################


###



#########################################
########### 5) Performance Plot #########
#########################################

# Hier noch ein mal, weil ich nichts bei Punkt 3 verändern will. Mergen soll einfach sein
i, label_2, cat_2, wildlife_2, dog_2 = np.genfromtxt('2layer_predictions_test.txt',unpack=True)
i, label_7, cat_7, wildlife_7, dog_7 = np.genfromtxt('7layer_predictions_test.txt',unpack=True)

# We relabel everything so that we have cat,dog,wild!
# Otherwise I dont see a way to plot the confusion matrix like this
wildlife = label_2 == 1
dog = label_2 == 2
label_2[wildlife] = 2
label_2[dog] = 1

wildlife = label_7 == 1
dog = label_7 == 2
label_7[wildlife] = 2
label_7[dog] = 1

#Y_pred_2 = np.vstack((dog_2, wildlife_2, cat_2)).T # Now it is in the common convention (1500,3)
Y_pred_2 = np.vstack((cat_2, dog_2, wildlife_2)).T # Now it is in the common convention (1500,3)
# Number of examples, number of classes
Y_cls_2 = np.argmax(Y_pred_2, axis = 1)
conf_mat_2 = confusion_matrix(label_2, Y_cls_2)

# Do the same for the seven layer model
#Y_pred_7 = np.vstack((dog_7, wildlife_7, cat_7)).T # Now it is in the common convention (1500,3)
Y_pred_7 = np.vstack((cat_7, dog_7, wildlife_7)).T # Now it is in the common convention (1500,3)
# Number of examples, number of classes
Y_cls_7 = np.argmax(Y_pred_7, axis = 1)
conf_mat_7 = confusion_matrix(label_7, Y_cls_7)

# plot the confusion matrices side by side
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(121)
plot_confusion_matrix(conf_mat_2, classes = ['cat', 'dog', 'wildlife'], normalize=False, title='Two Layers', colorbar=False)
plt.subplot(122)
plot_confusion_matrix(conf_mat_7, classes = ['cat', 'dog', 'wildlife'], normalize=False, title='Seven Layers', colorbar=False)
plt.tight_layout()
plt.savefig('plots/confusion_matrix.pdf')
plt.clf()

##################################################
########### 6) Understanding performance #########
##################################################

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

def lin(x,a,b):
    return a*x+b

def plot_roc_curves(label, Y_pred, title='ROC Curves'):
    # Binarize the output
    Y_one_hot = label_binarize(label, classes=[0, 1, 2])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    x_plot = np.linspace(0.001,1,1000)
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(Y_one_hot[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                label='ROC class {0} (AUC = {1:0.3f})'
                ''.format(i, roc_auc[i]))
        plt.plot(x_plot, lin(x_plot, 1,0), 'k--')
        plt.xlim([0.001, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xscale('log')
        plt.title(title)
        plt.tight_layout()
        plt.legend(loc="best")
        plt.grid()


#plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.figure(figsize=(9,4.5))
plt.subplot(121)
plot_roc_curves(label_2, Y_pred_2, title='Two Layers')
plt.subplot(122)
plot_roc_curves(label_7, Y_pred_7, title='Seven Layers')
plt.savefig('plots/ROC_Curves.pdf')
