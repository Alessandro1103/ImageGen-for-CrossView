import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch import cuda
from torch.utils.data import DataLoader

from generator import Generator
from discriminator import Discriminator

# Potremmo fare di meglio
def get_loss():
    """Returns L1 as loss."""
    return nn.L1Loss()

# 
# class GeneratorLoss(nn.Module):
#     def __init__(self, lambda_l1=100):
#         super(GeneratorLoss, self).__init__()
#         self.adversarial_loss = nn.BCELoss()  # Per la GAN Loss
#         self.l1_loss = nn.L1Loss()  # Per la ricostruzione
#         self.lambda_l1 = lambda_l1

#     def forward(self, discriminator_output, generated_image, target_image):
#         adversarial_loss = self.adversarial_loss(discriminator_output, torch.ones_like(discriminator_output))
#         l1_loss = self.l1_loss(generated_image, target_image)
#         return adversarial_loss + self.lambda_l1 * l1_loss

# def get_loss():
#     """Returns combined adversarial + L1 loss."""
#     return GeneratorLoss(lambda_l1=100)


# e nel train loop:

# loss_fn = get_loss()
# loss = loss_fn(discriminator(fake_imgs), fake_imgs, real_imgs)



def get_device():
    """Returns CUDA if CUDA-supporting GPU available for use, else CPU."""
    return "cuda" if cuda.is_available() else "cpu"

def get_optimizer(model, lr, b1, b2):
    """Returns Adam optimizer."""
    return Adam(model.parameters(), lr=lr, betas=(b1, b2))

def get_data_loader():
    """Returns DataLoader for the dataset."""
    pass


def main():
    device = get_device()
    data_loader = get_data_loader()

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
        for batch_idx, (imgs, _) in enumerate(tqdm.tqdm(data_loader)):
            batch_size = imgs.shape[0]
            real = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)
            imgs = imgs.to(device)
            
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
            real_loss = loss(discriminator(imgs), real)
            d_loss = (fake_loss + real_loss) / 2
            d_loss.backward()
            d_optim.step()
            mean_d_loss += d_loss.item()

        mean_g_loss /= len(data_loader)
        mean_d_loss /= len(data_loader)

        print(f"Epoch [{epoch+1}/{n_epochs}] | G Loss: {mean_g_loss:.4f} | D Loss: {mean_d_loss:.4f}")

        if mean_d_loss < curr_min_d_loss:
            curr_min_d_loss = mean_d_loss
            torch.save(discriminator.state_dict(), 'discriminator.pth')
        
        if mean_g_loss < curr_min_g_loss:
            curr_min_g_loss = mean_g_loss
            torch.save(generator.state_dict(), 'generator.pth')

if __name__ == "__main__":
    main()
