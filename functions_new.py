
import os
from scipy import ndimage, misc
from skimage.measure import label, regionprops, regionprops_table
import numpy as np
import os
import matplotlib.pylab as plt
from skimage.transform import rescale, resize, downscale_local_mean
import torch
import numpy as numpy


'''standard preprocessing, i.e. only standardization and expansion of dimensions, we use that for sag and cor, axial patches are already normalized'''
def preprocessing_no_patching(X):
    liste=[]
    for i in range(len(X)):
        m = (X[i]-np.mean(X[i]))/np.std(X[i])
        liste.append(m)
    X = np.asarray(liste)
    return X


''' '''




def create_boxes_sag_cor(X,thalamus):
    image = X[:,int(thalamus)-32:int(thalamus)+32,: ]
    return image
 
def preprocessing_coronal_sagittal(X, Y, thalamus=60, mode = "sag"):
    ''' input X, T1, patches,
         Y GT patches, 
         thalamus training index of middle slices training and val same, 
         N_augmentations preprocessing number,
         N_training_data, how many volumes are in the training,
         mode : can either be sag or cor'''
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    if mode == "cor":
        X_train = np.moveaxis(X,0,1)
        Y_train = np.moveaxis(Y,0,1)
        X_res = create_boxes_sag_cor(X_train,thalamus)
        Y_res = create_boxes_sag_cor(Y_train,thalamus)
        X_res = preprocessing_no_patching(X_res)

    if mode == "sag":    
        X_train = np.moveaxis(X,2,0)
        Y_train = np.moveaxis(Y,2,0)
        X_res = create_boxes_sag_cor(X_train,thalamus)
        Y_res = create_boxes_sag_cor(Y_train,thalamus)
        X_res = preprocessing_no_patching(X_res)



    X_train=X_res
    Y_train=Y_res
    return X_train[:,:,:],Y_train[:,:,:]

