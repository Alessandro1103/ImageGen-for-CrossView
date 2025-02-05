import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from blocks import EncoderBlock 


class VGG(nn.Module):
    def __init__(self, input_channels=3):  # Aggiunto input_channels
        super(VGG, self).__init__()

        # Define the network using EncoderBlock to modularize the layers
        self.layer1 = EncoderBlock(input_channels, 64, use_bn=False)
        self.layer2 = EncoderBlock(64, 128, use_bn=False)
        self.layer3 = EncoderBlock(128, 256, use_bn=False)
        self.layer4 = EncoderBlock(256, 512, use_bn=False)
        self.layer5 = EncoderBlock(512, 512, use_bn=False, use_dropout=True)
        self.layer6 = EncoderBlock(512, 512, use_bn=False, use_dropout=True)
        self.layer7 = EncoderBlock(512, 512, use_bn=False, use_dropout=True)
        
        self.fc = None

        self.initialize_weights()

    def initialize_weights(self):
        """Inizializza i pesi con Xavier."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)

        out7 = torch.cat(
            [out5.reshape(batch_size, -1), out6.reshape(batch_size, -1), out7.reshape(batch_size, -1)], dim=1
        )

        if self.fc is None:
            self.fc = nn.Linear(out7.shape[1], 1000).to(x.device)

        out8 = self.fc(out7)

        return out8


if __name__ == "__main__":
    input_channels = 3
    batch_size = 3
    street = torch.randn(batch_size, input_channels, 224, 1232)

    model = VGG()
    output_of_street = model(street)

    assert output_of_street.shape == (batch_size, 1000), f"Unexpected shape for output_img: {output_of_street.shape}"

    print("Test passed for street images.")

    input_channels = 3
    batch_size = 3
    sat = torch.randn(batch_size, input_channels, 224, 1232)

    model = VGG()
    output_of_sat = model(sat)

    assert output_of_sat.shape == (batch_size, 1000), f"Unexpected shape for output_of_sat: {output_of_sat.shape}"

    print("Test passed for sat images")
