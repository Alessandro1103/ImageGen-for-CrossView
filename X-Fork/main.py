import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch import cuda
from torch.utils.data import DataLoader, random_split
import os
import cv2

from generator import Generator
from discriminator import Discriminator
from dataset import ImageDataset

# Potremmo fare di meglio
def get_loss():
    """Returns L1 as loss."""
    return nn.L1Loss()


def get_device():
    """Returns CUDA if CUDA-supporting GPU available for use, else CPU."""
    return "cuda" if cuda.is_available() else "cpu"

def get_optimizer(model, lr, b1, b2):
    """Returns Adam optimizer."""
    return Adam(model.parameters(), lr=lr, betas=(b1, b2))

def get_data_loader(root_folder, batch_size=32, shuffle=True, num_workers=4, split_ratios=(0.7, 0.15, 0.15)):
    """Restituisce i DataLoader per train, val e test."""

    # Inizializza il dataset
    dataset = ImageDataset(root_folder)

    # Calcoliamo la dimensione di ogni split
    train_size = int(len(dataset) * split_ratios[0])
    val_size = int(len(dataset) * split_ratios[1])
    test_size = len(dataset) - train_size - val_size  # Il restante va nel test set

    # Divisione del dataset
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Creiamo i DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def main():

    device = get_device()
    train_loader, val_loader, test_loader = get_data_loader("./CVUSA_subset", batch_size=32, shuffle=True, num_workers=4)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    lr, b1, b2 = 0.0002, 0.5, 0.999
    g_optim = get_optimizer(generator, lr, b1, b2)
    d_optim = get_optimizer(discriminator, lr, b1, b2)

    curr_min_d_loss = float('inf')
    curr_min_g_loss = float('inf')

    loss = get_loss()

    n_epochs = 100

    for epoch in tqdm.trange(n_epochs):
        print(epoch)
        mean_g_loss = 0
        mean_d_loss = 0
        for imgs_sat, imgs_street in tqdm.tqdm(train_loader):
            batch_size = imgs_sat.shape[0] # anche imgs_street.shape[0] va bene
            real = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)
            imgs_sat = imgs_sat.to(device)
            imgs_street = imgs_street.to(device)
            
            # Train the generator
            g_optim.zero_grad()
            noise = torch.randn(batch_size, 4, 224, 1232).to(device)
            fake_imgs, _ = generator(noise)
            g_loss = loss(discriminator(fake_imgs), real)
            g_loss.backward()
            g_optim.step()        
            mean_g_loss += g_loss.item()
            
            # Train the discriminator
            d_optim.zero_grad()
            fake_loss = loss(discriminator(fake_imgs.detach()), fake)
            real_loss = loss(discriminator(imgs_street), real)
            d_loss = (fake_loss + real_loss) / 2
            d_loss.backward()
            d_optim.step()
            mean_d_loss += d_loss.item()

        mean_g_loss /= len(train_loader)
        mean_d_loss /= len(train_loader)

        print(f"Epoch [{epoch+1}/{n_epochs}] | G Loss: {mean_g_loss:.4f} | D Loss: {mean_d_loss:.4f}")

        if mean_d_loss < curr_min_d_loss:
            curr_min_d_loss = mean_d_loss
            torch.save(discriminator.state_dict(), 'discriminator.pth')
        
        if mean_g_loss < curr_min_g_loss:
            curr_min_g_loss = mean_g_loss
            torch.save(generator.state_dict(), 'generator.pth')

if __name__ == "__main__":
    main()