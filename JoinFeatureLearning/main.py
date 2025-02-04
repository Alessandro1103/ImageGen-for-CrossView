import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch import cuda
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from dataset import ImageDataset
from XFork.generator import Generator
from JFL import *


def get_device():
    """Returns CUDA if CUDA-supporting GPU available for use, else CPU."""
    return "cuda" if cuda.is_available() else "cpu"

def get_optimizer(model, lr, b1, b2):
    """Returns Adam optimizer."""
    return optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

def load_generator(model_path, device):
    generator = Generator(input_channels=4, output_channels=3).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval() 
    return generator

def generate_synthetic_image(generator, input_tensor, device):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        fake_satellite, _ = generator(input_tensor)
        
    return fake_satellite

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


def main(model_path):

    device = get_device()

    train_loader, val_loader, test_loader = get_data_loader("./../CVUSA_subset", batch_size=32, shuffle=True, num_workers=4)

    model = JointFeatureLearning(device).to(device)
    
    lr, b1, b2 = 0.0002, 0.5, 0.999

    optimizer = get_optimizer(model, lr, b1, b2)
    generator = load_generator(model_path, device)

    epochs = 100

    for epoch in range(epochs):

        total_loss = 0

        
        for x_sat_correct, x_street, x_sat_wrong in train_loader:
        
            x_sat_correct = x_sat_correct.to(device)
            x_sat_wrong = x_sat_wrong.to(device)
            x_street = x_street.to(device)
            

            x_synthetic = generate_synthetic_image(generator, x_street, device)


            optimizer.zero_grad()
            loss = model(x_street, x_sat_correct, x_sat_wrong, x_synthetic)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    print("Training finished")
        


if __name__=="__main__":
    MODEL_PATH = "./../XFork/models/generator.pth"
    main(MODEL_PATH)