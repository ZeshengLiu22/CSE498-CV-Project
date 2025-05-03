import os
import torch
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import GeneratorRRDB
from datasets import ImageDataset
import skimage.metrics

# -------- Settings --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "results/floodnet_sr"
os.makedirs(save_dir, exist_ok=True)

# Load pre-trained model
generator = GeneratorRRDB(channels=3, filters=64, num_res_blocks=23).to(device)
generator.load_state_dict(torch.load("saved_models_FloodNet_512/generator_final.pth", map_location=device))
generator.eval()

# Dataset & Dataloader
test_dataset = ImageDataset(phase='test', dataset_name='FloodNet')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 1)

# PSNR / SSIM helpers
def calc_psnr(sr, hr):
    sr_img = sr.detach().cpu().numpy().transpose(1, 2, 0)
    hr_img = hr.detach().cpu().numpy().transpose(1, 2, 0)
    return skimage.metrics.peak_signal_noise_ratio(hr_img, sr_img, data_range=1.0)

def calc_ssim(sr, hr):
    sr_img = sr.detach().cpu().numpy().transpose(1, 2, 0)
    hr_img = hr.detach().cpu().numpy().transpose(1, 2, 0)
    return skimage.metrics.structural_similarity(hr_img, sr_img, channel_axis=-1, data_range=1.0)

# Evaluation
total_psnr = 0
total_ssim = 0
count = 0

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader)):
        lr = data["lr"].to(device)
        hr = data["hr"].to(device)

        sr = generator(lr)
        sr_denorm = denormalize(sr[0].cpu())
        hr_denorm = denormalize(hr[0].cpu())

        psnr = calc_psnr(sr_denorm, hr_denorm)
        ssim = calc_ssim(sr_denorm, hr_denorm)

        total_psnr += psnr
        total_ssim += ssim
        count += 1

        # Get original filename and save
        lr_filename = os.path.basename(test_loader.dataset.lr_files[i])
        name, ext = os.path.splitext(lr_filename)
        save_path = os.path.join(save_dir, f"{name}-sr{ext}")
        save_image(sr_denorm, save_path)

# Final average scores
avg_psnr = total_psnr / count
avg_ssim = total_ssim / count
print(f"Finished evaluation on FloodNet test set")
print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
