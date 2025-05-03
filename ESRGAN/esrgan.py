"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan.py'
"""

'''
Adapted from: https://github.com/eriklindernoren/PyTorch-GAN

MOdified by ZeshengLiu to fit for FloodNet/RescueNet 
'''

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import skimage.metrics
from tqdm import tqdm

os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

def calc_psnr(sr, hr):
    '''
    Calculate PSNR between sr and hr
    param sr: super-resolved image
    param hr: high-resolution image
    return: PSNR value
    '''

    sr_img = sr.detach().cpu().numpy().transpose(1, 2, 0)
    hr_img = hr.detach().cpu().numpy().transpose(1, 2, 0)

    return skimage.metrics.peak_signal_noise_ratio(hr_img, sr_img, data_range=1.0)

def calc_ssim(sr, hr):
    '''
    Calculate SSIM between sr and hr
    param sr: super-resolved image
    param hr: high-resolution image
    return: SSIM value
    '''

    sr_img = sr.detach().cpu().numpy().transpose(1, 2, 0)
    hr_img = hr.detach().cpu().numpy().transpose(1, 2, 0)

    return skimage.metrics.structural_similarity(hr_img, sr_img, channel_axis=-1, data_range=1.0)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--total_iters", type=int, default=10000, help="number of iterations for training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=512, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=512, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=100, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
feature_extractor = FeatureExtractor().to(device)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

train_dataset = ImageDataset(phase='train', dataset_name = opt.dataset_name)
val_dataset = ImageDataset(phase='val', dataset_name = opt.dataset_name)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu,
)

iters_per_epoch = len(train_dataloader)
opt.n_epochs = math.ceil(opt.total_iters / iters_per_epoch)
opt.warmup_batches = iters_per_epoch
print(f"Adjusted n_epochs = {opt.n_epochs} to match total {opt.total_iters} iterations.")

# ----------
#  Training
# ----------

batches_done = 0
for epoch in tqdm(range(opt.epoch, opt.n_epochs)):
    for i, imgs in enumerate(train_dataloader):
        if batches_done >= opt.total_iters:
            break

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            print(f"[Epoch {epoch}] [Iter {batches_done}] [G pixel loss: {loss_pixel.item():.4f}]")
            batches_done += 1
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        # Total generator loss
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        print(f"[Epoch {epoch}] [Iter {batches_done}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}, content: {loss_content.item():.4f}, adv: {loss_GAN.item():.4f}, pixel: {loss_pixel.item():.4f}]")

        # --------------
        # Validation
        # --------------
        if batches_done % 500 == 0:
            generator.eval()
            psnr_total, ssim_total = 0, 0
            with torch.no_grad():
                for val_imgs in val_dataloader:
                    val_lr = val_imgs["lr"].to(device)
                    val_hr = val_imgs["hr"].to(device)

                    gen_val = generator(val_lr)
                    for b in range(gen_val.size(0)):
                        
                        pred = denormalize(gen_val[b].cpu())
                        truth = denormalize(val_hr[b].cpu())
                        psnr_total += calc_psnr(pred, truth)
                        ssim_total += calc_ssim(pred, truth)

            avg_psnr = psnr_total / len(val_dataloader)
            avg_ssim = ssim_total / len(val_dataloader)
            print(f"[Validation @ Iter {batches_done}] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")
            generator.train()



        if batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), f"saved_models/generator_{batches_done}.pth")
            torch.save(discriminator.state_dict(), f"saved_models/discriminator_{batches_done}.pth")
        
        batches_done += 1
    if batches_done >= opt.total_iters:
        break


# ------------------------------
# Final Evaluation and Saving
# ------------------------------
print("Training completed. Running final validation...")

generator.eval()
psnr_total, ssim_total = 0, 0
with torch.no_grad():
    for val_imgs in val_dataloader:
        val_lr = val_imgs["lr"].to(device)
        val_hr = val_imgs["hr"].to(device)

        gen_val = generator(val_lr)
        for b in range(gen_val.size(0)):
            pred = denormalize(gen_val[b].cpu())
            truth = denormalize(val_hr[b].cpu())
            psnr_total += calc_psnr(pred, truth)
            ssim_total += calc_ssim(pred, truth)

avg_psnr = psnr_total / len(val_dataloader)
avg_ssim = ssim_total / len(val_dataloader)
print(f"[Final Evaluation] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")

# Save final model
torch.save(generator.state_dict(), "saved_models/generator_final.pth")
torch.save(discriminator.state_dict(), "saved_models/discriminator_final.pth")
