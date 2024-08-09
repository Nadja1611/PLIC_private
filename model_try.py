import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, models
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
from functions_new import preprocessing_no_patching, create_boxes_sag_cor, preprocessing_coronal_sagittal
from dataset import *


''' set argparse elements relevant for the training of the baby method '''
parser = argparse.ArgumentParser(description='Arguments for segmentation network.', add_help=False)
parser.add_argument('-l', '--loss_variant', type=str,
                    help='which loss variant should be used', default="Dice")
parser.add_argument('-alpha', '--alpha', type=float,
                    help='how much noisier should z be than y', default=0.5)
parser.add_argument('-lr', '--learning_rate', type=str,
                    help='which learning rate should be used', default=1e-5)
parser.add_argument('-i', '--inputdir', type=str,
                    help='directory for input files', default="/home/nadja/nadja/PLIC/Data_npz")
parser.add_argument('-batch_size', '--batch_size', type=int,
                    help='batch size used for dataloader', default=1)
parser.add_argument('-N_training_data', '--N_training_data', type=int,
                    help='how many volumes are used for training', default=85)
parser.add_argument('-N_augmentations', '--N_augmentations', type=int,
                    help='what is the factor of increased volumes because of data augmentations', default=3)
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
        sliceview: str = "axial"
    ):
        self.mean_FP = []
        self.mean_FN = []
        self.Recall = 0
        self.max_mean_dice = 0
        self.max_mean_dice_sag = 0
        self.max_mean_dice_cor = 0

        self.mean_recall = []
        self.learning_rate = 5e-5
        self.net = UNet(1, 1)
        self.weight = 0.001
        # self.inputs = "DWI"
        self.segmentation_net = UNet(n_channels=1, n_classes=1).to(device)
        self.segmentation_net_sag = UNet(n_channels=1, n_classes=1).to(device)
        self.segmentation_net_cor = UNet(n_channels=1, n_classes=1).to(device)

        self.optimizer = optim.Adam(self.segmentation_net.parameters(), lr=self.learning_rate)
        self.optimizer_cor = optim.Adam(self.segmentation_net_cor.parameters(), lr=self.learning_rate)
        self.optimizer_sag = optim.Adam(self.segmentation_net_sag.parameters(), lr=self.learning_rate)

        self.device = 'cuda:0'
        self.Dice = 0
        self.Dice_isles = 0
        self.inputs = inputs
        self.mean_dice = []
        self.mean_dice_isles = []
        self.mean_dice_isles_sag = []
        self.mean_dice_sag = []
        self.mean_dice_isles_cor = []
        self.mean_dice_cor = []
        self.FP = 0
        self.FN = 0
        self.mean_spec = []
        self.Spec = 0
        self.alpha = 0.7
        self.gamma = 0.5
        self.delta = 0.5
        self.data_val_sag = None
        self.data_sag = None
        self.data_cor = None
        self.data_val_cor = None
        self.batch_size = args.batch_size

        self.noise = "gauss"
        self.noise_intensity = 0.1

        # Dataset
        self.train_dataset = Babyloader(noise_type=self.noise, noise_intensity=self.noise_intensity, train=True)
        self.test_dataset = Babyloader(noise_type=self.noise, noise_intensity=self.noise_intensity, train=False)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # For tracking and plotting loss and metrics
        self.train_losses = []
        self.val_losses = []
        self.val_dices = []

    def init_NW(self, device):
        if self.inputs == "T1":
            self.segmentation_net = UNet(n_channels=1, n_classes=1).to(device)
            self.segmentation_net_cor = UNet(n_channels=1, n_classes=1).to(device)
            self.segmentation_sag = UNet(n_channels=1, n_classes=1).to(device)

        self.optimizer = optim.Adam(self.segmentation_net.parameters(), lr=self.learning_rate)
        self.optimizer_cor = optim.Adam(self.segmentation_net_cor.parameters(), lr=self.learning_rate)
        self.optimizer_sag = optim.Adam(self.segmentation_net_sag.parameters(), lr=self.learning_rate)

    def tversky(self, tp, fn, fp):
        loss2 = 1 - ((torch.sum(tp) + 0.000001) / ((torch.sum(tp) + self.gamma * torch.sum(fn) + self.delta * torch.sum(fp) + 0.000001)))
        return loss2

    def dice_loss(self, segmentation_mask, output):
        weights = torch.stack([torch.tensor(1 - self.weight), torch.tensor(self.weight)]).to(self.device)
        output = torch.stack([output, 1 - output], axis=-1)
        segmentation_mask = torch.stack([segmentation_mask, 1 - segmentation_mask], axis=-1)

        output = torch.clip(output, min=1e-6)
        loss1 = -torch.sum(segmentation_mask * torch.log(output) * weights, axis=-1)
        loss1 = torch.mean(loss1)

        '''tversky preperation'''
        y_true_f = torch.flatten(segmentation_mask[:, :, :, :])
        y_pred_f = torch.flatten(output[:, :, :, :])
        fp = (1 - y_true_f) * y_pred_f
        fn = (1 - y_pred_f) * y_true_f
        tp = y_pred_f * y_true_f

        loss = (self.alpha * loss1) + (1 - self.alpha) * (self.tversky(tp, fn, fp))
        del(tp, fp, fn, y_true_f, y_pred_f)
        gc.collect()
        return loss

    def dice_coefficient(self, y_true, y_pred, epsilon=1e-6):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = (y_true_f * y_pred_f).sum()
        return (2. * intersection + epsilon) / (y_true_f.sum() + y_pred_f.sum() + epsilon)


Segmenter = Segmentation_of_PLIC()

for epoch in range(10):  # Increase the number of epochs for more meaningful training
    train_loss = 0.0
    val_loss = 0.0
    val_dice = 0.0
    Segmenter.segmentation_net.train()
    Segmenter.segmentation_net_cor.train()
    Segmenter.segmentation_net_sag.train()

    for i, batch in enumerate(Segmenter.train_dataloader):

        # Zero the parameter gradients
        Segmenter.optimizer.zero_grad()
        Segmenter.optimizer_cor.zero_grad()
        Segmenter.optimizer_sag.zero_grad()

        # Load inputs and masks
        inputs = batch['data_patch'].float().squeeze().unsqueeze(1).to(device)
        mask = batch['mask_patch'].squeeze().unsqueeze(1).to(device)
        inputs_sag = batch['data_patch_sag'].float().squeeze().unsqueeze(1).to(device)
        inputs_cor = batch['data_patch_cor'].float().squeeze().unsqueeze(1).to(device)
        mask_sag = batch['mask_patch_sag'].float().squeeze().unsqueeze(1).to(device)
        mask_cor = batch['mask_patch_cor'].float().squeeze().unsqueeze(1).to(device)

        # Forward pass
        output = F.sigmoid(Segmenter.segmentation_net(inputs))
        output_cor = F.sigmoid(Segmenter.segmentation_net_cor(inputs_cor))
        output_sag = F.sigmoid(Segmenter.segmentation_net_sag(inputs_sag))

        # Compute loss
        loss = Segmenter.dice_loss(mask, output)
        loss_cor = Segmenter.dice_loss(mask_cor, output_cor)
        loss_sag = Segmenter.dice_loss(mask_sag, output_sag)

        # Backward pass and optimization
        loss.backward()
        Segmenter.optimizer.step()

        loss_cor.backward()
        Segmenter.optimizer_cor.step()

        loss_sag.backward()
        Segmenter.optimizer_sag.step()

        train_loss += (loss.item() + loss_cor.item() + loss_sag.item())

    Segmenter.train_losses.append(train_loss / len(Segmenter.train_dataloader))

    with torch.no_grad():
        Segmenter.segmentation_net.eval()
        Segmenter.segmentation_net_cor.eval()
        Segmenter.segmentation_net_sag.eval()

        for i, batch in enumerate(Segmenter.test_dataloader):
            # Load inputs and masks
            inputs = batch['data_patch'].float().squeeze().unsqueeze(1).to(device)
            mask = batch['mask_patch'].squeeze().unsqueeze(1).to(device)
            inputs_sag = batch['data_patch_sag'].float().squeeze().unsqueeze(1).to(device)
            inputs_cor = batch['data_patch_cor'].float().squeeze().unsqueeze(1).to(device)
            mask_sag = batch['mask_patch_sag'].float().squeeze().unsqueeze(1).to(device)
            mask_cor = batch['mask_patch_cor'].float().squeeze().unsqueeze(1).to(device)

            # Forward pass
            output = F.sigmoid(Segmenter.segmentation_net(inputs))
            output_cor = F.sigmoid(Segmenter.segmentation_net_cor(inputs_cor))
            output_sag = F.sigmoid(Segmenter.segmentation_net_sag(inputs_sag))

            # Compute loss
            loss = Segmenter.dice_loss(mask, output)
            loss_cor = Segmenter.dice_loss(mask_cor, output_cor)
            loss_sag = Segmenter.dice_loss(mask_sag, output_sag)

            val_loss += (loss.item() + loss_cor.item() + loss_sag.item())

            # Compute Dice coefficient
            dice = Segmenter.dice_coefficient(mask, output)
            dice_cor = Segmenter.dice_coefficient(mask_cor, output_cor)
            dice_sag = Segmenter.dice_coefficient(mask_sag, output_sag)

            val_dice += (dice.item() + dice_cor.item() + dice_sag.item())

        Segmenter.val_losses.append(val_loss / len(Segmenter.test_dataloader))
        Segmenter.val_dices.append(val_dice / len(Segmenter.test_dataloader))

    print(f'Epoch {epoch + 1}/{10}, Training Loss: {Segmenter.train_losses[-1]:.4f}, Validation Loss: {Segmenter.val_losses[-1]:.4f}, Validation Dice: {Segmenter.val_dices[-1]:.4f}')


# Plotting the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(Segmenter.train_losses, label='Training Loss')
plt.plot(Segmenter.val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Plot')

plt.subplot(1, 2, 2)
plt.plot(Segmenter.val_dices, label='Validation Dice')
plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.title('Dice Coefficient Plot')

plt.show()
