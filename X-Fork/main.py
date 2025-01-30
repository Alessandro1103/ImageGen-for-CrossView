import tqdm
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import sys
from torch.optim import Adam
from torch import cuda
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generator import Generator
from discriminator import Discriminator
from dataset import ImageDataset
from edge_Concatenate import apply_canny, concatenate_images

# Funzione per ottenere la loss
def get_loss():
    return nn.L1Loss()

# Funzione per ottenere il dispositivo
def get_device():
    return "cuda" if cuda.is_available() else "cpu"

# Funzione per ottenere l'ottimizzatore
def get_optimizer(model, lr, b1, b2):
    return Adam(model.parameters(), lr=lr, betas=(b1, b2))

# Funzione per ottenere il DataLoader con split train/val/test
def get_data_loader(root_folder, batch_size=32, shuffle=True, num_workers=4, split_ratios=(0.7, 0.15, 0.15)):
    dataset = ImageDataset(root_folder)

    train_size = int(len(dataset) * split_ratios[0])
    val_size = int(len(dataset) * split_ratios[1])
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def main():
    device = get_device()
    train_loader, val_loader, test_loader = get_data_loader("./../CVUSA_subset", batch_size=32, shuffle=True, num_workers=4)

    generator = Generator().to(device)
    discriminator = Discriminator(input_channels=6).to(device)  # Discriminatore con 6 canali

    lr, b1, b2 = 0.0002, 0.5, 0.999
    g_optim = get_optimizer(generator, lr, b1, b2)
    d_optim = get_optimizer(discriminator, lr, b1, b2)

    loss = get_loss()
    n_epochs = 100

    for epoch in tqdm.trange(n_epochs):
        mean_g_loss = 0
        mean_d_loss = 0

        for imgs_sat, imgs_street in tqdm.tqdm(train_loader):
            batch_size = imgs_sat.shape[0]

            imgs_sat = imgs_sat.to(device)
            imgs_street = imgs_street.to(device)

            imgs_sat_np = imgs_sat.permute(0, 2, 3, 1).cpu().numpy()

            edges = np.array([apply_canny(img) for img in imgs_sat_np])
            imgs_sat_4ch_np = np.array([concatenate_images(img, edge) for img, edge in zip(imgs_sat_np, edges)])

            imgs_sat_4ch = torch.from_numpy(imgs_sat_4ch_np).float().permute(0, 3, 1, 2).to(device)
            print(f"Input Generator Shape: {imgs_sat_4ch.shape}")  # Debug

            real = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)

            # Train Generatore
            g_optim.zero_grad()
            fake_imgs, _ = generator(imgs_sat_4ch)
            fake_imgs = torch.nn.functional.interpolate(fake_imgs, size=(512, 512), mode="bilinear")
            g_loss = loss(discriminator(torch.cat((imgs_street, fake_imgs), dim=1)), real)
            g_loss.backward()
            g_optim.step()
            mean_g_loss += g_loss.item()

            # Train Discriminatore
            d_optim.zero_grad()
            fake_loss = loss(discriminator(torch.cat((imgs_street, fake_imgs.detach()), dim=1)), fake)
            real_loss = loss(discriminator(torch.cat((imgs_street, imgs_street), dim=1)), real)
            d_loss = (fake_loss + real_loss) / 2
            d_loss.backward()
            d_optim.step()
            mean_d_loss += d_loss.item()

        print(f"Epoch [{epoch+1}/{n_epochs}] | G Loss: {mean_g_loss/len(train_loader):.4f} | D Loss: {mean_d_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    main()
