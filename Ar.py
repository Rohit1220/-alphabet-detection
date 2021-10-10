import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score as ac

x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
n_classes = len(classes)
spc = 5
fig = plt.figure(figsize=(n_classes*2,(1+spc*2)))
index = 0
for i in classes:
  inbs = np.flatnonzero(y == i)
  inbs = np.random.choice(inbs,spc,replace = False)
  j = 0
  for ids in inbs:
    pltids = j*n_classes + index +1
    fig = plt.subplot(spc,n_classes,pltids)
    fig = sns.heatmap(np.reshape(x[ids],(22,30)),cmap = plt.cm.gray,xticklabels = False, yticklabels = False, cbar= False)
    fig = plt.axis('off')
    j += 1
  index += 1
print(len(x[0]))
print(len(x))
print(x[0])
print(y[0])
x_train, x_test , y_train , y_test = tts(x,y,random_state = 9,train_size = 7500, test_size = 2500)
x_train_scale = x_train/255.0
x_test_scale = x_test/255.0
model = lr(solver = 'saga',multi_class= 'multinomial').fit(x_train_scale,y_train)
y_predict = model.predict(x_test_scale)
print("accuracy",ac(y_test,y_predict))