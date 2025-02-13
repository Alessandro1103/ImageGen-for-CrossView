import os
import sys
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch import cuda
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from dataset import ImageDataset
from XFork.generator import Generator

def get_device():
    return "cuda" if cuda.is_available() else "cpu"

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


def tensor_to_image(tensor):
    """Converte un tensore PyTorch in immagine NumPy normalizzata per plt.imshow()."""
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (C, H, W) → (H, W, C)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalizza tra 0 e 1
    return tensor

def main(model_path="./models/generator.pth"):
    device = get_device()
    print(f"Usando dispositivo: {device}")

    train_loader, val_loader, test_loader = get_data_loader("./CVUSA_subset", batch_size=1, shuffle=True, num_workers=4, split_ratios=(0.7, 0.15, 0.15))

    x_sat_correct, x_street, x_sat_wrong = next(iter(test_loader))  # Prendi il primo batch
    x_street = x_street.to(device)

    generator = Generator(input_channels=4, output_channels=3).to(device)

    try:
        state_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(state_dict)
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        sys.exit(1)

    generator.eval()

    # Genera l'immagine satellitare artificiale
    with torch.no_grad():
        x_sat_artificial, _ = generator(x_street)  # 👈 Prendi solo la prima uscita (immagine generata)

    print(f"Dimensioni immagini: x_sat_correct={x_sat_correct.shape}, x_street={x_street.shape}, x_sat_artificial={x_sat_artificial.shape}")

    # Correzione: rimuovi la dimensione batch selezionando direttamente l'elemento 0
    x_sat_correct = x_sat_correct[0]
    x_sat_artificial = x_sat_artificial[0]

    print(f"Dimensioni immagini (post rimozione batch): x_sat_correct={x_sat_correct.shape}, x_sat_artificial={x_sat_artificial.shape}")

    # Converti in NumPy per Matplotlib
    x_sat_correct = x_sat_correct.permute(1, 2, 0).cpu().numpy()
    x_sat_artificial = x_sat_artificial.permute(1, 2, 0).cpu().numpy()

    # Plot dei risultati
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(x_sat_correct[:, :, :3])  # Rimuovi il quarto canale se presente
    plt.title("Satellite Image original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(x_sat_artificial)  # Ora sarà un tensore corretto
    plt.title("Satellite Image artificial")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
