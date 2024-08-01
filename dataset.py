import os
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
import random
from utils import *
from functions_new import *

class RandomTransform:
    def __init__(self, noise_level=0.5, contrast_factor=0.1):
        self.noise_level = noise_level
        self.contrast_factor = contrast_factor

    def add_noise(self, img):
        """Add random Gaussian noise to an image."""
        noise = np.random.normal(0, self.noise_level, img.shape)
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 1)  # Ensure values are within [0, 1]
        return noisy_img

    def enhance_contrast(self, img):
        """Enhance the contrast of an image."""
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        enhancer = ImageEnhance.Contrast(img_pil)
        enhanced_img = enhancer.enhance(self.contrast_factor)
        return np.array(enhanced_img) / 255.0

    def __call__(self, img):
        """Randomly apply one of the three transformations."""
        transform_choice = random.choice(['noise', 'contrast', 'none'])
        if transform_choice == 'noise':
            print("Applying noise transformation")
            return self.add_noise(img)
        elif transform_choice == 'contrast':
            print("Applying contrast transformation")
            return self.enhance_contrast(img)
        else:
            print("No transformation applied")
            return img

class Babyloader(Dataset):
    def __init__(self, data_dir="D:/PLIC/Babies/", noise_type='gauss', noise_intensity=0.05, angles=512, train=True, transform=None):
        super(Babyloader, self).__init__()

        self.noise_intensity = noise_intensity
        self.noise_type = noise_type
        self.angles = angles
 
        if train:
            self.clean_dir = os.path.join(data_dir, 'training')
        else:
            self.clean_dir = os.path.join(data_dir, 'testing')

        self.clean_paths = sorted(self.make_dataset(self.clean_dir))
        self.random_transform = RandomTransform(noise_level=noise_intensity, contrast_factor=0.5)

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def make_dataset(self, dir):
        # Function to collect all file paths in the dataset directory
        paths = []
        for root, _, files in os.walk(dir):
            for file in files:
                if file.endswith(".npz"):
                    paths.append(os.path.join(root, file))
        return paths

    def normalize(self, image):
        for i in range(len(image)):
            image[i] = image[i] - np.min(image[i])
            image[i] = image[i] / ((np.max(image[i]) + 1e-5))
        return image   

    def __getitem__(self, index):
        clean_path = self.clean_paths[index]
        data = np.load(clean_path)
        #### read in image patches of T1 images
        img = data['T1_patches']
        ### read in label patches
        mask = torch.tensor(data['mask_patches'])
        ### read in full T1 images
        full_T1 = torch.tensor(data['T1'])
        ### read in full image masks
        full_mask = torch.tensor(data['masks'])
        indices = data['indices']
        """ In that case, we add the gaussian noise in the sinogram, then noise is uncorrelated in sino domain """
        clean = np.array([img.squeeze() for img in img], dtype='float32')
        clean_copy = np.copy(clean)
        clean = np.asarray(normalize_scan(torch.tensor(clean)))
        preprocessed = clean
        thalamus = int(np.round(clean.shape[0]/2))
        
        inputs_sag, mask_sag = preprocessing_coronal_sagittal(np.asarray(clean_copy), np.asarray(mask), thalamus=thalamus, mode = "sag")
        clean_sag = np.asarray(normalize_scan(torch.tensor(inputs_sag)))

        inputs_cor, mask_cor = preprocessing_coronal_sagittal(np.asarray(clean_copy), np.asarray(mask), thalamus=thalamus, mode = "cor")
        clean_cor = np.asarray(normalize_scan(torch.tensor(inputs_cor)))

        

        # Apply random transformation slice by slice for axial, coronal and sagittal slices
        transformed = np.stack([self.random_transform(slice) for slice in preprocessed])
        transformed_sag = np.stack([self.random_transform(slice) for slice in clean_sag])
        transformed_cor = np.stack([self.random_transform(slice) for slice in clean_cor])

        
        # Visualize the original and transformed images
        if index == 0:  # Only visualize for the first image for brevity
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(mask_cor[56], cmap='gray')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("Transformed Image ")
            plt.imshow(transformed_cor[56], cmap='gray')
            plt.colorbar()
            plt.show()


        return {'data_patch': torch.tensor(transformed), 'data_patch_cor': torch.tensor(transformed_cor), 'data_patch_sag': torch.tensor(transformed_sag), 'mask_patch': mask, 'mask_patch_sag': mask_sag, 'mask_patch_cor': mask_cor,  'data': full_T1, 'full_mask': full_mask, 'indices': indices}

    def __len__(self):
        return len(self.clean_paths)

# Create dataset and dataloader
dataset = Babyloader()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate over the dataset
for i, sample in enumerate(dataloader):
    # Only iterate a few samples to see the effect
    if i >= 5:
        break
