import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch import cuda
from torch.utils.data import DataLoader, random_split
import os
import cv2
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from generator import Generator
from discriminator import Discriminator
from dataset import ImageDataset

def get_device():
    """Returns CUDA if CUDA-supporting GPU available for use, else CPU."""
    return "cuda" if cuda.is_available() else "cpu"

def get_optimizer(model, lr, b1, b2):
    """Returns Adam optimizer."""
    return Adam(model.parameters(), lr=lr, betas=(b1, b2))

def get_data_loader(root_folder, batch_size=32, shuffle=True, num_workers=4, split_ratios=(1, 0, 0)):
    """Restituisce i DataLoader per train, val e test."""
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

    generator = Generator(input_channels=4, output_channels=3).to(device)
    discriminator = Discriminator(input_channels=6).to(device)

    lr, b1, b2 = 0.0002, 0.5, 0.999
    g_optim = get_optimizer(generator, lr, b1, b2)
    d_optim = get_optimizer(discriminator, lr, b1, b2)

    curr_min_d_loss = float('inf')
    curr_min_g_loss = float('inf')

    # Definiamo le funzioni di perdita
    adversarial_loss = nn.BCEWithLogitsLoss()  # Perdita avversaria
    l1_loss = nn.L1Loss()  # Perdita L1 per la coerenza delle immagini

    lambda_l1 = 100  # Peso della perdita L1

    # **Prealloca i tensori reali/fake per evitare ridondanza**
    real = torch.ones((train_loader.batch_size, 1), device=device)
    fake = torch.zeros((train_loader.batch_size, 1), device=device)

    n_epochs = 100

    for epoch in tqdm.trange(n_epochs):
        print(epoch)
        mean_g_loss = 0
        mean_d_loss = 0

        for imgs_sat, imgs_street in tqdm.tqdm(train_loader):
            batch_size = imgs_sat.shape[0]

            # Aggiorna i tensori reali/fake alla dimensione del batch attuale
            real_resized = real[:batch_size]
            fake_resized = fake[:batch_size]
            
            
            batch_size = imgs_sat.shape[0] # anche imgs_street.shape[0] va bene
            real = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)

            batch_size = imgs_sat.shape[0] # anche imgs_street.shape[0] va bene
            real = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)
            imgs_sat = imgs_sat.to(device)
            imgs_street = imgs_street.to(device)

            # **TRAINING DEL GENERATORE**
            g_optim.zero_grad()
            fake_sat, _ = generator(imgs_street)  # Generazione immagine sintetica

            resize_transform = torch.nn.functional.interpolate
            imgs_street = resize_transform(imgs_street, size=(512, 512), mode='bilinear', align_corners=False)
            
            # Concatenazione per il discriminatore
            real_input = torch.cat((imgs_street[:,:3,:,:], imgs_sat), dim=1)
            fake_input = torch.cat((imgs_street[:,:3,:,:], fake_sat), dim=1)

            # **Loss del generatore**
            pred_fake = discriminator(fake_input)  # Output del discriminatore su immagine sintetica
            g_loss_adv = adversarial_loss(pred_fake, real_resized)  # Il generatore cerca di "ingannare" il discriminatore
            g_loss_l1 = l1_loss(fake_sat, imgs_sat)  # Perdita L1 per evitare immagini sfocate
            g_loss = g_loss_adv + lambda_l1 * g_loss_l1  # Combiniamo entrambe le perdite

            g_loss.backward()
            g_optim.step()
            mean_g_loss += g_loss.item()

            # Calcoliamo le predizioni del discriminatore
            pred_real = discriminator(real_input)
            pred_fake = discriminator(fake_input)

            # **Loss del discriminatore**
            d_loss_real = adversarial_loss(pred_real, real_resized)
            d_loss_fake = adversarial_loss(pred_fake, fake_resized)
            d_loss = (d_loss_real + d_loss_fake) / 2  # Media delle due perdite

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