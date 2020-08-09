# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 02:48:50 2019

@author: Himanshu
"""

from glob import glob
import time
from adjusting import adjust, dataset_new, optical_flow, diff, optical_flow_diff, optical_flow_acc
from cnnlstm import cnnlstm
from joblib import dump, load
import numpy as np
from keras.utils import plot_model
import matplotlib.pyplot as plt
from timedistributed import cnn_lstm2


#for reproducing the results
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# Training/nofights
#Hockey dataset

img_mask = 'Dataset/crowd_nofights/*.avi'
names = glob(img_mask)
train_nofights = optical_flow(names)
train_nofights1 = diff(names)
train_nofights2 = optical_flow_diff(train_nofights)
train_nofights3 = optical_flow_acc(train_nofights2)
#train_nofights = adjust(names)

# Training/fights
img_mask = 'Dataset/crowd_fights/*.avi'
names = glob(img_mask)
train_fights = optical_flow(names)
train_fights1 = diff(names)
train_fights2 = optical_flow_diff(train_fights)
train_fights3 = optical_flow_acc(train_fights2)
#train_fights = adjust(names)
# dataset(fights, nofights)'
s = time.time()
data = dataset_new(train_fights3, train_nofights3, frames = 38)
print(time.time() - s)
train, target = data[0], data[1]



del(names, img_mask, data, train_nofights, train_fights)
#dump(train, 'hog_data.joblib')
#dump(target, 'hog_test.joblib')
'''
# Architecture of cnnlstm
arch = cnnlstm(38, 1, 64, 64)
classifier = KerasClassifier(build_fn = cnnlstm(40, 1, 64, 64), batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = train, y = target, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()

# Fitting the model
history = arch.fit(train, target, validation_split=0.25, batch_size = 4, epochs = 200, verbose = 1)

dump(arch, 'classifier_paatani.joblib')
#plots
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')

#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
# Saving the weights
dump(arch, 'classifier_denoise_layer.joblib')

#train2 = train[:700]

arch.fit(train, target[:700], batch_size = 10, epochs = 100, verbose = 1)

dump(arch, 'classifier.joblib')

#predicting
classifier = load('classifier_5_layer.joblib')

#testing phase
img_mask = '*.mp4'
names = glob(img_mask)
img_mask = 'Dataset/Testing/fights/*.avi'
names.extend(glob(img_mask))

test_data = optical_flow(img_mask)
test_data = np.reshape(test_data, (40, 40,1, 64, 64))
test_labels = [0]*20 + [1]*20 #actual test labels

check = classifier.predict(test_data)
answer = [int(i) for i in check]

correct_pred = 0 #if the category is correctly classified 
                 #then this will be incremented by one
        
for i in range(len(answer)):
    if answer[i] == test_labels[i]:
        correct_pred = correct_pred + 1

accuracy =  correct_pred / 40

if int(check):
    print('Violence Detected')
else:
    print('No violence is detected')'''
    
    
import numpy as np
train2 = np.reshape(train, (242, 60, 64, 64, 1))
arch = cnn_lstm2(60, 64, 64, 1)
arch = arch.fit(train, target, validation_split=0.25, batch_size = 4, epochs = 50, verbose = 1)

dump(arch, 'classifier_denoise_layer_timedistributed.joblib')