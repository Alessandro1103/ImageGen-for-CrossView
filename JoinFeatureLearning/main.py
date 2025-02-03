import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from FeatureExtractor.VGG import VGG


class JointFeatureLearning(nn.Module):
    def __init__(self):
        super(JointFeatureLearning, self).__init__()

        self.vgg_sat = VGG()
        self.vgg_street = VGG()

        self.alpha = 10
        self.lambda1 = 10
        self.lambda2 = 1


    # x_sat è l'immagine satellitare dal dataset
    # x_sat_dummy è l'immagine satellitare generata dal generatore
    # x_street è l'immagine stradale dal dataset
    def forward(self, x_sat_correct, x_sat_wrong, x_sat_dummy, x_street):
        
        f_g = self.vgg_street(x_street)
        f_a_pos = self.vgg_sat(x_sat_wrong)
        f_a_neg = self.vgg_sat(x_sat_correct)
        f_a_gen = self.vgg_sat(x_sat_dummy) 

        d_p = torch.norm(f_g - f_a_pos, p=2, dim=1)
        d_n = torch.norm(f_g - f_a_neg, p=2, dim=1)
        d_gen = torch.norm(f_a_gen - f_a_pos, p=2, dim=1)
        
        loss = torch.log(1 + self.alpha * torch.exp(d_p - d_n)).mean()
        loss_aux = torch.log(1 + self.alpha * torch.exp(d_gen)).mean()

        loss_tot = self.lambda1 * loss + self.lambda2 * loss_aux

        return loss_tot
        