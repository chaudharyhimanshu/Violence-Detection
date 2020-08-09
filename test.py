# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 02:46:10 2020

@author: Himanshu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
cap = cv2.VideoCapture('videoplayback.mp4')
ret, frame1 = cap.read()
frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
new2 = np.zeros((5, 360, 640))
n = 0
prev = frame1
x = 0
while(1):
    ret, frame2 = cap.read()
    if ret:
        if n == 1000 or n==999 or n == 1001 or n == 1002 or n == 998:
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            new2[x] = next
            x += 1
        n += 1
    else:
        break
    
    
plt.subplot(2, 2, 1)
plt.imshow(new2[0], cmap = 'Greys_r')
plt.subplot(2, 2, 2)
plt.imshow(new2[1], cmap = 'Greys_r')
plt.subplot(2, 2, 3)
plt.imshow(new2[2], cmap = 'Greys_r')
plt.subplot(2, 2, 4)
plt.imshow(new2[3], cmap = 'Greys_r')

a = new2[0, 0:50, 400:500]
b = new2[1, 0:50, 400:500]
c = new2[2, 0:50, 400:500]
d = new2[3, 0:50, 400:500]

plt.subplot(2, 2, 1)
plt.hist(np.reshape(a, (-1,1)))
plt.subplot(2, 2, 2)
plt.hist(np.reshape(b, (-1,1)))
plt.subplot(2, 2, 3)
plt.hist(np.reshape(c, (-1,1)))
plt.subplot(2, 2, 4)
plt.hist(np.reshape(d, (-1,1)))

plt.subplot(2, 2, 1)
plt.imshow(a, cmap = 'Greys_r')
plt.subplot(2, 2, 2)
plt.imshow(b, cmap = 'Greys_r')
plt.subplot(2, 2, 3)
plt.imshow(c, cmap = 'Greys_r')
plt.subplot(2, 2, 4)
plt.imshow(d, cmap = 'Greys_r')

sample1 = b - a
sample2 = c - a
sample3 = d - c

plt.hist(np.reshape(sample1, (-1,1)))
plt.hist(np.reshape(sample2, (-1,1)))
plt.hist(np.reshape(sample3, (-1,1)))


e = new2[1] - new2[0]
f = new2[2] - new2[1]
g = new2[3] - new2[2]
h = new2[4] - new2[3]

plt.hist(np.reshape(e, (-1,1)))
plt.hist(np.reshape(f, (-1,1)))
plt.hist(np.reshape(g, (-1,1)))
plt.hist(np.reshape(h, (-1,1)))


e[e < 0] = 0
plt.imshow(e, cmap = 'Greys_r')
e[f < 0] = 0
plt.imshow(f, cmap = 'Greys_r')
e[e < 0] = 0
plt.imshow(e, cmap = 'Greys_r')
e[e < 0] = 0
plt.imshow(e, cmap = 'Greys_r')



plt.subplot(2, 2, 1)
plt.imshow(new2[1] - new2[0], cmap = 'Greys_r')
plt.subplot(2, 2, 2)
plt.imshow(new2[2] - new2[1], cmap = 'Greys_r')
plt.subplot(2, 2, 3)
plt.imshow(new2[3] - new2[2], cmap = 'Greys_r')
plt.subplot(2, 2, 4)
plt.imshow(new2[4] - new2[3], cmap = 'Greys_r')





import cv2
import numpy as np
import matplotlib.pyplot as plt
cap = cv2.VideoCapture('xyz.mkv')
ret, frame1 = cap.read()
frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
out = cv2.VideoWriter('outpy11.avi',cv2.VideoWriter_fourcc('M','J','P','G'),
                          int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(3)),int(cap.get(4))), 0)



prev = frame1
while(1):
    ret, frame2 = cap.read()
    if ret:
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        v = next.astype(int) - prev.astype(int)
        v[np.where((v > -10) & (v < 10))] = 0
        v = np.array(v, dtype = np.uint8)
        out.write(v)
        prev = next
    else:
        break
    
cap.release()
out.release()



for i in range(101):
#plt.imshow(frame1, cmap = 'Greys_r')
    ret, frame1 = cap.read()
frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
ret, frame2 = cap.read()
frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
plt.imshow(frame2, cmap = 'Greys_r')
plt.imshow(frame2-frame1, cmap = 'Greys_r')
#after 100 frames
sample = frame2 - frame1
sample = sample[250:,550:]
sample2 = frame1[250:,550:]
sample3 = frame2[250:,550:]

plt.imshow(sample2, cmap = 'Greys_r')
plt.imshow(sample3.astype(int) - sample2.astype(int), cmap = 'Greys_r')
x = np.array(frame2.astype(int) - frame1.astype(int), dtype = np.uint8)
y = frame2.astype(int) - frame1.astype(int)
y[np.where((y > -8) & (y < 8))] = 0
plt.imshow(np.uint8(y), cmap = 'Greys_r')
y[(y > -10) & (y < 10)] = 0
y = np.uint8(np.log1p(y))
y = np.array(y, dtype = np.uint8)
