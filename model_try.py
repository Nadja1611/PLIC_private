
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import utils,models
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from tqdm import trange
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn import init, ReflectionPad2d, ZeroPad2d
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from unet_blocks import *
import random
import numpy as np
import matplotlib.pyplot as plt
import gc
import argparse
from functions_new import preprocessing_no_patching,  create_boxes_sag_cor, preprocessing_coronal_sagittal
from dataset import *




''' set argparse elements relevant for the training of the baby method '''
parser = argparse.ArgumentParser(description='Arguments for segmentation network.',add_help=False)
parser.add_argument('-l','--loss_variant', type=str, 
                    help='which loss variant should be used', default= "Dice")
parser.add_argument('-alpha','--alpha', type=float, 
                    help='how much noisier should z be than y', default= 0.5)
parser.add_argument('-lr','--learning_rate', type=str, 
                    help='which learning rate should be used', default= 1e-5)
parser.add_argument('-i', '--inputdir', type=str, 
                    help='directory for input files', default = "/home/nadja/nadja/PLIC/Data_npz" )
parser.add_argument('-batch_size', '--batch_size', type=int, 
                    help='batch size used for dataloader', default = 1 )
parser.add_argument('-N_training_data', '--N_training_data', type=int, 
                    help='how many volumes are used for training', default = 85 )
parser.add_argument('-N_augmentations', '--N_augmentations', type=int, 
                    help='what is the factor of increased volumes because of data augmentations', default = 3 )
args = parser.parse_args()


torch.manual_seed(0)

def split_volumes(x_new, indices):
    volumes = []
    start = 0
    for num_slices in indices:
        end = start + num_slices
        volume = x_new[start:end]
        volumes.append(volume)
        start = end
    return volumes



device = "cuda"



N_total = 98


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class Segmentation_of_PLIC():
    def __init__(
        self,
        learning_rate: float = 5e-5,
        device: str = 'cuda:0',
        inputs: str = "T1",
        sliceview: str= "axial"
    ):
        self.mean_FP=[]
        self.mean_FN=[]
        self.Recall= 0
        self.max_mean_dice=0
        self.max_mean_dice_sag=0
        self.max_mean_dice_cor=0

        self.mean_recall=[]
        self.learning_rate = 5e-5
        self.net = UNet(1,1)
        self.weight = 0.001
        #self.inputs = "DWI"
        self.segmentation_net = UNet(n_channels=1,n_classes=1).to(device)
        self.segmentation_net_sag = UNet(n_channels=1,n_classes=1).to(device)
        self.segmentation_net_cor = UNet(n_channels=1,n_classes=1).to(device)
        
        self.optimizer = optim.Adam(self.segmentation_net.parameters(), lr=self.learning_rate)
        self.optimizer_cor = optim.Adam(self.segmentation_net_cor.parameters(), lr=self.learning_rate)
        self.optimizer_sag = optim.Adam(self.segmentation_net_sag.parameters(), lr=self.learning_rate)

        self.device = 'cuda:0'
        self.Dice = 0
        self.Dice_isles = 0
        self.inputs = inputs
        self.mean_dice=[]
        self.mean_dice_isles=[]
        self.mean_dice_isles_sag = []
        self.mean_dice_sag=[]
        self.mean_dice_isles_cor = []
        self.mean_dice_cor = []
        self.FP= 0
        self.FN = 0
        self.mean_spec=[]
        self.Spec = 0
        self.alpha=0.7
        self.gamma=0.5
        self.delta=0.5
        self.data_val_sag=None
        self.data_sag=None
        self.data_cor = None
        self.data_val_cor=None
        self.batch_size = args.batch_size
        
        self.noise = "gauss"
        self.noise_intensity = 0.1


                # Dataset
        self.train_dataset = Babyloader(noise_type = self.noise, noise_intensity = self.noise_intensity, train=True)
        self.test_dataset = Babyloader(noise_type = self.noise, noise_intensity = self.noise_intensity, train=False)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
    def init_NW(self, device):
        if self.inputs == "T1":
            self.segmentation_net = UNet(n_channels=1,n_classes=1).to(device)    
            self.segmentation_net_cor = UNet(n_channels=1,n_classes=1).to(device)  
            self.segmentation_sag = UNet(n_channels=1,n_classes=1).to(device)       


        self.optimizer = optim.Adam(self.segmentation_net.parameters(), lr=self.learning_rate)
        self.optimizer_cor = optim.Adam(self.segmentation_net_cor.parameters(), lr=self.learning_rate)
        self.optimizer_sag = optim.Adam(self.segmentation_net_sag.parameters(), lr=self.learning_rate)



           




Segmenter = Segmentation_of_PLIC()
for epoch in range(1):
    for i, batch in enumerate(Segmenter.train_dataloader):
        
        Segmenter.segmentation_net.train()
        Segmenter.segmentation_net_cor.train()
        Segmenter.segmentation_net_sag.train()



        ### read in the T1-cubes of size (1, number slices, 64,64)
        inputs = batch['data_patch'].float().squeeze().unsqueeze(1).to(device)
        mask = batch['mask_patch'].squeeze().unsqueeze(1).to(device)
        T1 = batch["data"].squeeze()
        inputs_sag = batch['data_patch_sag'].float().squeeze().unsqueeze(1).to(device)
        inputs_cor = batch['data_patch_cor'].float().squeeze().unsqueeze(1).to(device)
        mask_sag = batch['mask_patch_sag'].float().squeeze().unsqueeze(1).to(device)
        mask_cor = batch['mask_patch_cor'].float().squeeze().unsqueeze(1).to(device)


        if epoch % 1000 == 0:
            with torch.no_grad():
                plt.subplot(1,2,1)
                plt.imshow(inputs[40][0].detach().cpu(), cmap='gray')
                plt.subplot(1,2,2)
                plt.imshow(mask[40][0].detach().cpu(), cmap='gray')
                plt.savefig(newpath + '/'+ "data" + ".png")
                plt.close()
        Segmenter.optimizer.zero_grad()
        Segmenter.optimizer_cor.zero_grad()
        Segmenter.optimizer_sag.zero_grad()

