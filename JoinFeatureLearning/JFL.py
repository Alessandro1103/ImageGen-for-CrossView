import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from FeatureExtractor.VGG import VGG

# ============================
# JOINT FEATURE LEARNING LOSS
# ============================
class JointFeatureLearningLoss(nn.Module):
    def __init__(self):
        super(JointFeatureLearningLoss, self).__init__()

        self.alpha = 10  # Peso della loss triplet
        self.lambda1 = 10  # Peso della loss principale
        self.lambda2 = 1   # Peso della loss ausiliaria

    def forward(self, f_g, f_a_pos, f_a_neg, f_a_gen):
        """Calcola la triplet loss tra Street View e immagini Satellitari"""
        
        # Loss principale (Triplet loss)
        d_p = torch.norm(f_g - f_a_pos, p=2, dim=1)  # Distanza ground-satellite positivo
        d_n = torch.norm(f_g - f_a_neg, p=2, dim=1)  # Distanza ground-satellite negativo
        loss_triplet = torch.log(1 + self.alpha * torch.exp(d_p - d_n)).mean()

        # Loss ausiliaria (tra satellite reale e sintetico)
        d_gen = torch.norm(f_a_gen - f_a_pos, p=2, dim=1)
        loss_aux = torch.log(1 + self.alpha * torch.exp(d_gen)).mean()

        # Loss totale
        loss_tot = self.lambda1 * loss_triplet + self.lambda2 * loss_aux
        return loss_tot

# ============================
# JOINT FEATURE LEARNING NETWORK
# ============================
class JointFeatureLearningNetwork(nn.Module):
    def __init__(self, device):
        super(JointFeatureLearningNetwork, self).__init__()

        # Inizializza le tre reti VGG per estrarre le feature
        self.ground_vgg = VGG(input_channels=4).to(device)
        self.sat_vgg = VGG(input_channels=4).to(device)
        self.sat_gan_vgg = self.sat_vgg  # Condivide i pesi (weight sharing)


    def forward(self, x_ground, x_satellite, x_synthetic):
        """Estrae feature da Street View, Satellite reale e Satellite sintetico"""
        
        # Passa le immagini attraverso le VGG

        batch_size = x_ground.shape[0]

        assert x_ground.shape == (batch_size, 4, 224, 1232), f"Expected shape (batch_size, 4, 224, 1232), got {x_ground.shape}"
        assert x_satellite.shape == (batch_size, 4, 512, 512), f"Expected shape (batch_size, 4, 224, 1232), got {x_satellite.shape}"
        assert x_synthetic.shape == (batch_size, 4, 512, 512), f"Expected shape (batch_size, 4, 224, 1232), got {x_synthetic.shape}"

        f_g = self.ground_vgg(x_ground)
        f_a_pos = self.sat_vgg(x_satellite)
        f_a_gen = self.sat_gan_vgg(x_synthetic)

        return f_g, f_a_pos, f_a_gen

# ============================
# JOINT FEATURE LEARNING MODEL
# ============================
class JointFeatureLearning(nn.Module):
    def __init__(self, device):
        super(JointFeatureLearning, self).__init__()
        self.network = JointFeatureLearningNetwork(device).to(device)
        self.loss_fn = JointFeatureLearningLoss().to(device)

    def forward(self, x_ground, x_sat_correct, x_sat_wrong, x_synthetic):
        """Passa i dati attraverso la rete e calcola la loss"""
        
        # Estrai le feature dalle immagini
        f_g, f_a_pos, f_a_gen = self.network(x_ground, x_sat_correct, x_synthetic)
        f_a_neg, _, _ = self.network(x_ground, x_sat_wrong, x_synthetic)  # Feature dell'immagine sbagliata

        # Calcola la loss
        loss = self.loss_fn(f_g, f_a_pos, f_a_neg, f_a_gen)
        return loss
