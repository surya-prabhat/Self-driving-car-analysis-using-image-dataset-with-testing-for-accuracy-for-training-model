import os
from itertools import islice
from scipy import pi
import numpy as np
import matplotlib.pyplot as plt


Data_folder = 'D:\driving_dataset\driving_dataset'
Train_file = os.path.join(Data_folder , 'data.txt')
Limit = None

split = 0.8
X = []
y = []

with open(Train_file) as fp:
    for line in islice(fp, Limit):
         path, angle = line.strip().split()
         full_path = os.path.join(Data_folder, path)
         X.append(full_path)
         y.append(float(angle)*pi/100)

y = np.array(y)
print("Successful")

split_index = int(len(y)*0.8)
train_y = y[:split_index]
test_y = y[split_index :]
plt.hist(train_y, bins= 50, color = "blue" , histtype= 'step')
plt.hist(test_y, color= "red", bins= 50, histtype= 'step')
train_mean_y =  np.mean(train_y)
np = np.mean(np.square(test_y - train_mean_y))
print(np)
