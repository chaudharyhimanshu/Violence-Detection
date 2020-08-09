# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:07:00 2019

@author: Himanshu
"""
'''Functions of this:
1. This file converts the video into gray
2. Then resize it to 288 * 350 pixels and 40 frames
'''

import cv2
import numpy as np
from sklearn.utils import shuffle 
new_h = 64
new_w = 64
#for acceleration
def acc(diff):
    if diff + 128 < 0:
        acc = 0
    elif diff + 128 < 255:
        acc = diff + 128
    else:
        acc = 255
    return acc

def elu_diff(diff):
    diff = diff / 40
    return diff



def elu(diff):
    if diff + 128 < 0:    #Difference between every pixel of two consecutive frames 
        acc =  0.2 * diff
    else:
        acc = diff
    return acc
        






def optical_flow(names):
    op_flow = np.zeros(shape = (len(names), 40, new_h, new_w))
    f = 0
    for it, name in enumerate(names):
        cap = cv2.VideoCapture(name)
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 40:
                continue
        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        i = 0
        while(1):
            if i < 40:
                ret, frame2 = cap.read()
                next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                resize = cv2.resize(gray, (new_h, new_h))
                prvs = next
                op_flow[f, i] = resize
                i = i + 1
            else:
                break
        f = f+1
        print(it)
        cap.release()
    return op_flow[:f]


def optical_flow_video(name):
    cap = cv2.VideoCapture(name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'),
                          10, (frame_width,frame_height))
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    while(1):
        ret, frame2 = cap.read()
        if ret:
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prvs = next
            out.write(rgb)
        else:
            break
    cap.release()
    out.release()


def new_video(name):
    cap = cv2.VideoCapture(name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'),
                          10, (frame_width,frame_height))
    ret, frame1 = cap.read()
    frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    n = 1
    while(1):
        ret, frame2 = cap.read()
        if ret:
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            v = np.subtract(next, frame1)
            plt.imshow(v,  cmap='Greys_r')
            plt.imsave(str(n) + '.png', v,  cmap='Greys_r')
            n += 1
        else:
            break
    cap.release()


def optical_flow_diff(train):
    new_image = np.zeros(shape = [len(train),39, new_h, new_w])
    vecfunc = np.vectorize(elu)
    for i in range(len(train)):
        for j in range(1, 40):
            ac = np.subtract(train[i, j], train[i, j-1])
            ac = vecfunc(ac)
            new_image[i, j-1] = ac
        print(i)
    data = np.reshape(new_image, (train.shape[0], 39, 1, 64, 64))
    return data


def optical_flow_acc(train):
    new_image = np.zeros(shape = [len(train),38, new_h, new_w])
    vecfunc = np.vectorize(elu)
    for i in range(len(train)):
        for j in range(1, 39):
            ac = np.subtract(train[i, j], train[i, j-1])
            ac = vecfunc(ac)
            new_image[i, j-1] = ac
        print(i)
    data = np.reshape(new_image, (train.shape[0], 38, 1, 64, 64))
    return data



def diff(names):
    new_h = 64
    new_w = 64
    new_image = np.zeros(shape = [len(names),39, new_h, new_w])
    vecfunc = np.vectorize(elu_diff) #acc
    for it, name in enumerate(names):
        vidcap = cv2.VideoCapture(name)
        if int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) < 40:
                continue
        i = j = 0
        s, imag = vidcap.read()
        imag = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
        resize_1 = cv2.resize(imag, (new_w, new_h))
        while True:
            success,image = vidcap.read()
            if 0<=i<39:
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                resize = cv2.resize(image, (new_w, new_h))
                ac = np.subtract(resize, resize_1)
                ac = vecfunc(ac)
                resize_1 = resize
                new_image[it,j] = ac
                j = j + 1
            elif i >40:
                break
            i = i + 1
        
    return new_image


#basic preprocessing of video

def adjust(names): # names have path of all videos that needs adjusting
    new_h = 64  # pixel_y will be resized to 64
    new_w = 64  # pixel_x will be resized to 64
    new_videos = np.zeros(shape = [len(names),
                                   50, new_h, new_w]) # contains adjusted videos 
    f = 0
    for it, name in enumerate(names):
        vidcap = cv2.VideoCapture(name)
        if int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) < 50:
            continue
        i = j = 0
        while True:
            success,frame = vidcap.read()
            if 0<=i<60:
                im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                im = np.float32(im) / 255.0
                #gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
                #gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
                #mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
                resize = cv2.resize(mag, (64, 64))
                new_videos[f, j] = resize
                j = j + 1
            elif i >60:
                break
            i = i + 1
        f = f + 1
    new_videos = new_videos[:f]
    return new_videos

def dataset_new(fights, nofights, frames):
    target = np.zeros(shape = (len(nofights), 1))
    target2 = np.ones(shape = (len(fights), 1))
    target = np.append(target, target2, axis = 0)
    del target2
    data = np.append(nofights, fights, axis = 0)
    del fights, nofights
    data, target = shuffle(data, target, random_state=0)
    data = np.reshape(data, (data.shape[0], frames, 1, 64, 64))
    return [data, target]


'''def dataset(fights, nofights): # all videos containing fights and nofights
    new_h = 200  # pixel_y will be resized to 64
    new_w = 320  # pixel_x will be resized to 64
    limit = fights.shape[0] + nofights.shape[0] # total number of videos
                                                # i.e., fights + nofights
    target = np.zeros(shape = (limit,1)) # classes of training data 
    data = np.zeros(shape = [limit, 60, new_h, new_w]) # will contain fights and
                                                        # no fights videos mixed randomly
    
    count1 = count2 = 0 # keeps track of data in fights and nofights respectively
    for i in range(limit):
        if fights.shape[0] and nofights.shape[0]:
            group = np.random.randint(0,9) % 2 # 0 means nofights, 1 means fights
            which = np.random.randint(0,min(fights.shape[0],
                                            nofights.shape[0])) # randomly selects the
                                                                # video index to be appended
            if group == 1:
                data[i] = fights[which]
                count1 = count1 + 1
                fights = np.delete(fights, which, 0)
                target[i] = 1
            else:
                data[i] = nofights[which]
                count2 = count2 + 1
                nofights = np.delete(nofights, which, 0)
                target[i] = 0
        elif nofights.shape[0]: # if there are still videos left in nofights
            data = np.append(data, nofights, axis = 0)
            target = np.append(target, np.zeros(shape = (nofights.shape[0],
                                                         1)))
            del(nofights)
            break
        else: # if there are still videos left in fights
            data = np.append(data, fights, axis = 0)
            target = np.append(target, np.ones(shape = (fights.shape[0], 1)))
            del(fights)
            break
    
    data = np.reshape(data, (data.shape[0], 60, 1, new_h, new_w))
    return [data, target]

def single(name):
    new_h = 64
    new_w = 64
    new_image = np.zeros(shape = [40, new_h, new_w])
    name = 'Dataset/Testing/fights\\newfi46.avi'
    vidcap = cv2.VideoCapture(name)
    i = j = 0
    while True:
        success,image = vidcap.read()
        if 0<=i<40:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            resize = cv2.resize(image, (new_w, new_h))
            new_image[j] = resize
            j = j + 1
        elif i >40:
            break
        i = i + 1
    new_image = np.reshape(new_image, (1, 40, 1, 64, 64))
    return new_image

def flip(video):
    new_image = np.zeros(shape = [40, 64, 64])
    for i in range(40):
        new_image[i] = video[i].T
    return np.reshape(new_image, (1, 40, 64, 64))

'''
def hog_dis():
    im = cv2.imread('barbara.jpg')
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = np.float32(im) / 255.0
    
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
    hog = cv2.HOGDescriptor()
    h = hog.compute(im)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)