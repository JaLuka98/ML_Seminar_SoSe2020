import keras
import numpy as np
from skimage.io import imread
from tensorflow.keras.utils import Sequence

# Data generator zum dynamischen Laden der Bilder in Batches
class Generator(Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
  # Länge der Trainingsdaten (Anzahl der Batches)
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  # Lädt Bilder anhand der Pfade aus dem Trainingsarray
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([imread(str(file_name)) for file_name in batch_x])/255.0, np.array(batch_y)
