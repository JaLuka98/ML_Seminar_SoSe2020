import numpy as np
import matplotlib.pyplot as plt

i, label, cat, wildlife, dog = np.genfromtxt('predictions/7layer_predictions_test.txt',unpack=True)

print(max(wildlife[label==2]))
