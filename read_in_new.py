# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 11:10:19 2023

@author: nadja
"""

#from skimage import io
import numpy as np
import pydicom
import os
from skimage.transform import resize
import argparse
import nibabel as nib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='read in')

parser.add_argument('--input_directory', type=str, 
                    help='directory for input files', default = "C://Users//nadja//Documents//PLIC Segmentation//PLIC_pytorch//Babies//" )
parser.add_argument('--output_directory', type=str, 
                    help='directory for output files', default = "C://Users//nadja//Documents//PLIC Segmentation//PLIC_pytorch//Babies//" )

args = parser.parse_args()
input_path = "/home/nadja/nadja/PLIC/Babies/new/"
output_path = "/home/nadja/nadja/PLIC/Babies/training"

relevant_files = []
Names = []
babies = os.listdir(input_path)
for file in babies:
    newpath = input_path + "/" + file
    length = len(os.listdir(newpath))
    ### check whether labels are included in folder
    if length == 3:
        relevant_files.append(newpath)
        Names.append(file)
'''--------------------------------read in T1----------------------------------------'''
def read_in(input_path):
    ds=[]
    Patient=[]
    Patches = []
    length = []
    Spacings=[]
    T1_orig=[]
    Voxels=[]
  
    PathDicom_t1 =  input_path 
    for j in sorted(os.listdir(PathDicom_t1))[:-1]: 
        if j != "F0000000":
            header = pydicom.dcmread( input_path +  j, force = True)   
           # print(header)
            #print(header.ImageType)
            if 'PROJECTION IMAGE' not in header.ImageType:
                ds1 = pydicom.dcmread( input_path + j, force = True)     
                Pixel_spacings = ds1.PixelSpacing
                Thickness  = ds1.SliceThickness
                ds1 = ds1.pixel_array
                length.append(j)
                ds.append(ds1)
                T1_orig.append(ds1)
            else:
                print("reference image was removed")
    Spacings = list([Thickness, Pixel_spacings[0],Pixel_spacings[1]])
    Voxels = list([len(length), ds1.shape[0], ds1.shape[1]])
    ds = np.asarray(ds)     
    if ds.shape[0] > 190 or ds.shape[0]<90:
        print("We resize the input to shape (100x192x192)")
        ds = resize(ds, (100, 192, 192))

    else:
        ds = resize(ds, (np.shape(ds)[0], 192, 192 ))    

        
    Patient.append(np.asarray(ds))
    Patches.append(np.asarray(ds)[:, 65:129, 65:129])

    ds=[]
    Patches = np.asarray(np.concatenate(Patches,axis=0))    
        
    for i in range(len(Patches)):
        Patches[i] = Patches[i] - np.mean(Patches[i])
        Patches[i] = Patches[i]/np.std(Patches[i])    
    T1=np.concatenate(Patient, axis=0)
    T1=np.asarray(T1)
    
    indices = []
    for i in range(len(Patient)):
        length = np.shape(Patient[i])[0]
        indices.append(length)
  
    '''Grundstruktur und slices sortiert lassen'''
    liste = []
    liste.append(T1[:indices[0]])
    for i in range(len(Patient)-1):
        index1 = int(np.sum(indices[:i]))
        index2 = int(np.sum(indices[:i])+indices[i+1])
        P = T1[index1:index2]
        P=np.asarray(P)
        liste.append(P)   

    return T1, Patches, indices, Voxels, Spacings, np.asarray(T1_orig), header


def read_in_masks(inputdir):
    # List all files in the input directory
    files = os.listdir(inputdir)
    
    # Initialize a list to hold the arrays
    masks = []

    # Iterate through the files
    for file in files:
        # Check if the file is a NIfTI file
        if file.endswith('.nii.gz') or file.endswith('.nii'):
            # Load the NIfTI file
            file_path = os.path.join(inputdir, file)
            nifti_data = nib.load(file_path)
            # Convert the NIfTI data to a NumPy array and append to the list
            dat = (nifti_data.get_fdata())
            dat = np.transpose(dat, (2,1,0))
            if dat.shape[1] > 190 or ds.shape[0]<90:
                print("We resize the input to shape (100x192x192)")
                ds = resize(dat, (100, 192, 192))

            else:
                ds = resize(dat, (np.shape(dat)[0], 192, 192 ))    


    return np.round(dat[:-1,:,:]) 




for file in Names:
    labels = read_in_masks(input_path+"/" + file)
    T1 = read_in(input_path+ "/" + file + "/T1/DCM0/")[0]
    np.savez_compressed(output_path + "/Baby"+ file + ".npz",  T1 = read_in(input_path+"/" + file + "/T1/DCM0/")[0], indices = read_in(input_path+"/" +file  + "/T1/DCM0/")[2], T1_patches = read_in(input_path+"/" +file + "/T1/DCM0/")[1], masks = labels, mask_patches = labels[:, 65:129, 65:129])




for i in range(len(T1)):
    plt.subplot(1,2,1)
    plt.imshow(labels[i, 65:129, 65:129])
    plt.subplot(1,2,2)
    plt.imshow(T1[i, 65:129, 65:129])
    plt.savefig('/home/nadja/nadja/PLIC/Babies/plots/baby_' + str(i) )