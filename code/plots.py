import numpy as np
import os

import tensorboard_to_csv
from loader import load_data

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

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

i, label, cat, wildlife, dog = np.genfromtxt('predictions/7layer_predictions_train.txt',unpack=True)
i_test, label_test, cat_test, wildlife_test, dog_test = np.genfromtxt('predictions/7layer_predictions_test.txt',unpack=True)

## wildlife = 1
## cat = 0
## dog = 2

plt.figure(figsize=(14,5))
plt.subplots_adjust(wspace=0.3,hspace=0.35)


plt.subplot(131)
plt.title("Cat output node")
plt.hist(cat[label==0],label="Cat images train",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],alpha=0.5,color="red",density=True)
plt.hist(cat[label==2],label="Dog images train",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],alpha=0.5,color="blue",density=True)
plt.hist(cat[label==1],label="Wildlife images train",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],alpha=0.5,color="green",density=True)

counts_cat,bin_edges = np.histogram(cat_test[label_test==0],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],density=True)
counts_dog,bin_edges = np.histogram(cat_test[label_test==2],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],density=True)
counts_wild,bin_edges = np.histogram(cat_test[label_test==1],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],density=True)

bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.

plt.plot(bin_centres,counts_cat,'ro',alpha=0.8, label="Cat images test")
plt.plot(bin_centres,counts_dog,'bo',alpha=0.8, label="Dog images test")
plt.plot(bin_centres,counts_wild,'go',alpha=0.8, label="Wildlife images Test")

plt.yscale('log')


plt.subplot(132)
plt.title("Dog output node")
plt.hist(dog[label==0],label="Cat images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],alpha=0.5,color="red",density=True)
plt.hist(dog[label==2],label="Dog images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],alpha=0.5,color="blue",density=True)
plt.hist(dog[label==1],label="Wildlife images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],alpha=0.5,color="green",density=True)

counts_cat,bin_edges = np.histogram(dog_test[label_test==0],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],density=True)
counts_dog,bin_edges = np.histogram(dog_test[label_test==2],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],density=True)
counts_wild,bin_edges = np.histogram(dog_test[label_test==1],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],density=True)

plt.plot(bin_centres,counts_cat,'ro',alpha=0.8, label="Cat images test")
plt.plot(bin_centres,counts_dog,'bo',alpha=0.8, label="Dog images test")
plt.plot(bin_centres,counts_wild,'go',alpha=0.8, label="Wildlife images Test")

plt.yscale('log')


plt.subplot(133)
plt.title("Wildlife output node")
plt.hist(wildlife[label==0],label="Cat images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],alpha=0.5,color="red",density=True)
plt.hist(wildlife[label==2],label="Dog images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],alpha=0.5,color="blue",density=True)
plt.hist(wildlife[label==1],label="Wildlife images",histtype="step",bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],alpha=0.5,color="green",density=True)

counts_cat,bin_edges = np.histogram(wildlife_test[label_test==0],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],density=True)
counts_dog,bin_edges = np.histogram(wildlife_test[label_test==2],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],density=True)
counts_wild,bin_edges = np.histogram(wildlife_test[label_test==1],bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],density=True)

plt.plot(bin_centres,counts_cat,'ro',alpha=0.8, label="Cat images test")
plt.plot(bin_centres,counts_dog,'bo',alpha=0.8, label="Dog images test")
plt.plot(bin_centres,counts_wild,'go',alpha=0.8, label="Wildlife images Test")

plt.yscale('log')

plt.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
plt.legend(bbox_to_anchor=(-0.1,1))
plt.savefig("plots/Network-output.pdf")
plt.clf()

#########################################
########### 4) Overtraining check #######
#########################################

# Function that takes an array and swaps all ones and twos with each other
def resort_labels(labels):
    ones = labels == 1
    twos = labels == 2
    labels[ones] = 2
    labels[twos] = 1
    return labels


i, label_train_2, cat_train_2, wildlife_train_2, dog_train_2 = np.genfromtxt('predictions/2layer_predictions_train.txt',unpack=True)
i, label_train_7, cat_train_7, wildlife_train_7, dog_train_7 = np.genfromtxt('predictions/7layer_predictions_train.txt',unpack=True)

i, label_2, cat_2, wildlife_2, dog_2 = np.genfromtxt('predictions/2layer_predictions_test.txt',unpack=True)
i, label_7, cat_7, wildlife_7, dog_7 = np.genfromtxt('predictions/7layer_predictions_test.txt',unpack=True)

# We relabel everything so that we have cat,dog,wild!
label_train_2 = resort_labels(label_train_2)
label_2 = resort_labels(label_2)

label_train_7 = resort_labels(label_train_7)
label_7 = resort_labels(label_7)

# Ab jetzt also: Cat=0, dog=1, wild=2

def plot_cumulative_distribution(type, dists, labels, binning, classcode, classname):
    if type=='Two':
        c = 'orange'
    elif type=='Seven':
        c = 'crimson'

    train_val = dists[0]
    test = dists[1]
    label_train_val = labels[0]
    label_test = labels[1]
    plt.hist(train_val[label_train_val==classcode], bins=binning, density=True, histtype='step',
             cumulative=True, label=type+' Layers', color=c, linestyle='-')
    plt.hist(test[label_test==classcode], bins=binning, density=True, histtype='step',
             cumulative=True, color=c, linestyle='--')
    plt.title('True ' + classname + 's')
    plt.xlim([0,1])
    plt.ylim(1.5*1e-3,1)
    plt.yscale('log')
    plt.xlabel(classname + ' output node')
    plt.ylabel('ECDF (normed)')
    plt.legend(loc='upper center')
    plt.tight_layout()

binning = np.linspace(0,1,100)
plt.figure(figsize=(9,3))
plt.subplot(131)
plot_cumulative_distribution(type='Two', dists=[cat_train_2, cat_2], labels=[label_train_2, label_2],
                             binning=binning, classcode=0, classname='cat')
plot_cumulative_distribution(type='Seven', dists=[cat_train_7, cat_7], labels=[label_train_7, label_7],
                             binning=binning, classcode=0, classname='cat')
plt.grid()
plt.subplot(132)
plot_cumulative_distribution(type='Two', dists=[dog_train_2, dog_2], labels=[label_train_2, label_2],
                             binning=binning, classcode=1, classname='dog')
plot_cumulative_distribution(type='Seven', dists=[dog_train_7, dog_7], labels=[label_train_7, label_7],
                             binning=binning, classcode=1, classname='dog')
plt.grid()
plt.subplot(133)
plot_cumulative_distribution(type='Two', dists=[wildlife_train_2, wildlife_2], labels=[label_train_2, label_2],
                             binning=binning, classcode=2, classname='wild animal')
plot_cumulative_distribution(type='Seven', dists=[wildlife_train_7, wildlife_7], labels=[label_train_7, label_7],
                             binning=binning, classcode=2, classname='wild animal')
plt.grid()
plt.savefig('plots/cumulative.pdf')
plt.clf()

#########################################
########### 5) Performance Plot #########
#########################################

Y_pred_2 = np.vstack((cat_2, dog_2, wildlife_2)).T # Now it is in the common convention (1500,3)
# Number of examples, number of classes
Y_cls_2 = np.argmax(Y_pred_2, axis = 1)
conf_mat_2 = confusion_matrix(label_2, Y_cls_2)

# Do the same for the seven layer model
Y_pred_7 = np.vstack((cat_7, dog_7, wildlife_7)).T # Now it is in the common convention (1500,3)
# 1500: Number of examples, 3: number of classes
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
plt.clf()

######################################################
########### 7) Vergleich mit Alternative #############
######################################################

# We load the saved data, fit the knn and do prediction on the unseen test set
X_train = np.load(file="alternative/X_pca_train.npy")
y_train = np.load(file="alternative/labels_train.npy")

X_test = np.load(file="alternative/X_pca_test.npy")
y_test = np.load(file="alternative/labels_test.npy")

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=25)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)

report_knn = classification_report(y_test, y_pred, output_dict=True)
report_2 = classification_report(label_2.astype(int), Y_cls_2, output_dict=True)
print(label_2.shape)
report_7 = classification_report(label_7.astype(int), Y_cls_7, output_dict=True)

# We also load the dnn predictions as a final alternative method
i, label_dnn, cat_dnn, wildlife_dnn, dog_dnn = np.genfromtxt('predictions/dnn_predictions_test.txt',unpack=True)

# We relabel everything so that we have cat,dog,wild!
label_dnn = resort_labels(label_dnn)
Y_pred_dnn = np.vstack((cat_dnn, dog_dnn, wildlife_dnn)).T
Y_cls_dnn = np.argmax(Y_pred_dnn, axis = 1)
report_dnn = classification_report(label_dnn.astype(int), Y_cls_dnn, output_dict=True)

def plot_bars(classname, legend, report_dnn=report_dnn, report_knn=report_knn, report_2=report_2, report_7=report_7):
    # Transform name of class to label
    label = '0'
    if classname=='cats':
        label = '0'
    elif classname=='dogs':
        label = '1'
    elif classname=='wildlife':
        label = '2'

    # set width of bar
    barWidth = 0.25

    # set height of bars
    # Note that 0.0 is there because for some reason the labels were saved as floats for the neural networks
    bars1 = [report_knn[label]['precision'], report_dnn[label]['precision'], report_2[label]['precision'], report_7[label]['precision']]
    bars2 = [report_knn[label]['recall'], report_dnn[label]['recall'], report_2[label]['recall'], report_7[label]['recall']]
    bars3 = [report_knn[label]['f1-score'], report_dnn[label]['f1-score'], report_2[label]['f1-score'], report_2[label]['f1-score']]

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Precision')
    plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Recall')
    plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label=r'F$_1$ score')
    plt.grid(alpha=0.5)
    # Add xticks on the middle of the group bars
    plt.title(classname, fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], [ 'kNN', 'DNN', 'Two\nLayers', 'Seven\nLayers'])
    plt.xlim([-0.25,3.75])
    plt.ylim([0.0,1.35])
    plt.hlines(1,-4,4, linestyles='dashed')
    if legend==True: plt.legend()
    plt.tight_layout()

plt.figure(figsize=(9,3))
plt.subplot(131)
plot_bars(classname='cats', legend=False)
plt.subplot(132)
plot_bars(classname='dogs', legend=True)
plt.subplot(133)
plot_bars(classname='wildlife', legend=False)
plt.savefig('plots/comparison.pdf')
