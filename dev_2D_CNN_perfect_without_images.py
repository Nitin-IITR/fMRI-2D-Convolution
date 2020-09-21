import nmrglue as ng
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import nilearn
from nilearn import plotting
from nilearn import image
from nilearn import datasets
import nibabel as nib
import matplotlib.image as mpimg
import h5py
import imageio
import scipy.misc as spmi
import nibabel as nib
from nilearn.image import get_data
import os
import random
from random import seed
from keras.utils import normalize



##################################################################################
# Convert float4 to unit8
def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

#imgu8 = convert(img16u, 0, 255, np.uint8)

##################################################################################

##################################################################################
# Healthy subjects
files1 = os.listdir( r'D:\fmri dev data\Child' )
mystring1=r'D:\fmri dev data\Child\\'

filenames1=[ mystring1+s for s in files1];

#Schizophrenia subject
files2 = os.listdir( r'D:\fmri dev data\Adult' )
mystring2=r'D:\fmri dev data\Adult\\'

filenames2=[ mystring2+s for s in files2];

##################################################################################
    
def train_x():    
    for i in range(21):
        img1 = nib.load(filenames1[i])
        img2 = nib.load(filenames2[i])
        img_data1 = img1.get_fdata()
        img_data2 = img2.get_fdata()
        for j in range(28):
            brain_slice1=[]
            brain_slice2=[]
            for k in range(6):
                for l in range(8,36):
#                    rescale_img = convert(img_data[:,:,l,k+12*j], 0, 255, np.float32)
                    rescale_img1=normalize(img_data1[:,:,l,k+6*j])
                    rescale_img2=normalize(img_data2[:,:,l,k+6*j])
                    brain_slice1.append(rescale_img1.reshape(50,59,1))
                    brain_slice2.append(rescale_img2.reshape(50,59,1))
            brain_slice =  brain_slice1+brain_slice2     
            random.seed(j)
            random.shuffle(brain_slice)        
            img_array=np.array(brain_slice)
            yield img_array



def train_y():
    for i in range(21):  
        for j in range(28):
            a1=np.zeros(168)
            a2=np.ones(168)
            a=np.concatenate((a1,a2))
            random.seed(j)
            random.shuffle(a) 
            yield a


def train():
    while True:
        for (i,j) in zip(train_x(),train_y()): 
            yield (i,j)

        
##################################################################################                
                
def test_x():    
    for i in range(21,30):
        img1 = nib.load(filenames1[i])
        img2 = nib.load(filenames2[i])
        img_data1 = img1.get_fdata()
        img_data2 = img2.get_fdata()
        for j in range(21):
            brain_slice1=[]
            brain_slice2=[]
            for k in range(8):
                for l in range(8,36):
#                    rescale_img = convert(img_data[:,:,l,k+12*j], 0, 255, np.float32)
                    rescale_img1=normalize(img_data1[:,:,l,k+8*j])
                    rescale_img2=normalize(img_data2[:,:,l,k+8*j])
                    brain_slice1.append(rescale_img1.reshape(50,59,1))
                    brain_slice2.append(rescale_img2.reshape(50,59,1))
            brain_slice =  brain_slice1+brain_slice2     
            random.seed(j)
            random.shuffle(brain_slice)        
            img_array=np.array(brain_slice)
            yield img_array



def test_y():
    for i in range(21,30):
        for j in range(21):
            a1=np.zeros(224)
            a2=np.ones(224)
            a=np.concatenate((a1,a2))
            random.seed(j)
            random.shuffle(a) 
            yield a
        
def test():
    while True:
        for (i,j) in zip(test_x(),test_y()): 
            yield (i,j)

#################################################################################
##################################################################################    
#################################################################################
#####################     Deep Learning        ###############################     
    
    
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers



classifier = Sequential()

# Convolution and Maxpooling layer
classifier.add(Conv2D(filters=32,kernel_size=(3,3) ,padding='same',activation='relu',input_shape=(50,59,1)))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(filters=64,kernel_size=(3,3) ,padding='same',activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Flattening
classifier.add(Flatten())

#Fully Connected layer
classifier.add(Dense(256, activation='relu'))

classifier.add(Dense(128, activation='relu'))

classifier.add(Dense(1, activation='sigmoid'))

#Compiling
classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])


classifier.summary()
################################################################################
################################################################################


classifier.fit_generator(train(),steps_per_epoch=588,epochs=10,workers=1,
                         use_multiprocessing=False,
                         validation_data=test(),validation_steps=189)



################################################################################

from keras.models import load_model

classifier.save('schizo_model1.h5')  # creates a HDF5 file 'my_model.h5'
#del classifier  # deletes the existing model

classifier.save_weights('schizo_model_weights1.h5')
################################################
from keras.models import load_model

# returns a compiled model
# identical to the previous one
classifier = load_model('schizo_model.h5')

################################################
################################################

#classifier.evaluate_generator(test(),steps=45)    
    
    
    
    
    