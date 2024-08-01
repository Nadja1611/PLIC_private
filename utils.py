import os
import numpy as np
import torch
from skimage.transform import resize



def is_image_file(filename):
    extensions = ['.npz']
    return any(filename.endswith(extension) for extension in extensions)

def make_dataset(dir):
    img_paths = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    for (root, dirs, files) in sorted(os.walk(dir)):
        for filename in files:
            if is_image_file(filename):
                img_paths.append(os.path.join(root, filename))
    return img_paths


def normalize_scan(img):
    # Find the minimum and maximum values along the dimensions (1, 2)
    min_vals = img.amin(dim=(1, 2), keepdim=True)
    max_vals = img.amax(dim=(1, 2), keepdim=True)

    # Normalize the tensors
    normalized_tensors = (img- min_vals) / (max_vals - min_vals)
    return normalized_tensors



def resize_scan(scan, desired_width, desired_height):
    scan = resize(scan, (desired_height, desired_width))
    return scan


def preprocess_scan(clean, width, height):
    scan =np.copy(clean)
    resized_scan = resize_scan(scan, width, height)
    normalized_resized_scan = normalize_scan(resized_scan)
    return normalized_resized_scan

''' this function adds gaussian noise to our sinograms '''
def add_gaussian_noise(img, sigma):
    img_clone = np.copy(img)
    noise = np.random.normal(0, sigma, img_clone.shape)/100
    img_clone = img_clone + np.max(img)*noise
    return torch.tensor(img_clone)





# @title Define function to preprocess and save


