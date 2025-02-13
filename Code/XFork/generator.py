import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from blocks import EncoderBlock, DecoderBlock

class Generator(nn.Module):
    def __init__(self, input_channels = 4, output_channels = 3):
        super(Generator, self).__init__()

        # Encoder
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128, use_bn=True)                             
        self.encoder3 = EncoderBlock(128, 256, use_bn=True)                            
        self.encoder4 = EncoderBlock(256, 512, use_bn=True)                            
        self.encoder5 = EncoderBlock(512, 512, use_bn=True)                            
        self.encoder6 = EncoderBlock(512, 512, use_bn=True)                            
        self.encoder7 = EncoderBlock(512, 512, stride=(1,2), use_bn=True)                            
        self.encoder8 = EncoderBlock(512, 512, use_bn=True)      

        # Shared Decoder blocks
        self.decoder1 = DecoderBlock(512, 512, use_bn=True, use_dropout=True)
        self.decoder2 = DecoderBlock(512, 512, use_bn=True, use_dropout=True)
        self.decoder3 = DecoderBlock(512, 512, use_bn=True, use_dropout=True)
        self.decoder4 = DecoderBlock(512, 512, use_bn=True)
        self.decoder5 = DecoderBlock(512, 256, use_bn=True)
        self.decoder6 = DecoderBlock(256, 128, use_bn=True)

        self.decoder7_img = DecoderBlock(128, 64, use_bn=True)
        self.decoder7_seg = DecoderBlock(128, 64, use_bn=True)

        # Independent Decoder for street view image
        self.decoder8_img = nn.Sequential(
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # Independent Decoder for street view segmentation map
        self.decoder8_seg = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
   

    def forward(self, x):

        batch_size = x.shape[0]

        # Encoder forward
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        bottleneck = self.encoder8(e7)

        assert bottleneck.shape == (batch_size, 512, 1, 4), f"Unexpected shape for bottleneck: {bottleneck.shape}"

        # Controllo la dimensione del bottleneck
        bottleneck_reshaped = bottleneck.view(batch_size, 512, 2, 2)

        assert bottleneck_reshaped.shape == (batch_size, 512, 2, 2), f"Unexpected shape for bottleneck_reshaped: {bottleneck_reshaped.shape}"

        # Decoder forward (senza skip connections)
        d1 = self.decoder1(bottleneck_reshaped)
        d2 = self.decoder2(d1)
        d3 = self.decoder3(d2)
        d4 = self.decoder4(d3)
        d5 = self.decoder5(d4)
        d6 = self.decoder6(d5)

        d7_img = self.decoder7_img(d6)
        d7_seg = self.decoder7_seg(d6)

        streetview_img = self.decoder8_img(d7_img)
        streetview_seg = self.decoder8_seg(d7_seg)

        assert streetview_img.shape == (batch_size, 3, 512, 512), f"Unexpected shape for streetview_img: {streetview_img.shape}"
        assert streetview_seg.shape == (batch_size, 1, 512, 512), f"Unexpected shape for streetview_seg: {streetview_seg.shape}"

        return streetview_img, streetview_seg

    
if __name__ == "__main__":
    input_channels = 4
    output_channels = 3
    batch_size = 3
    input_tensor = torch.randn(batch_size, input_channels, 224, 1232)

    model = Generator(input_channels, output_channels)
    output_img, output_seg = model(input_tensor)

    assert output_img.shape == (batch_size, output_channels, 512, 512), f"Unexpected shape for output_img: {output_img.shape}"
    assert output_seg.shape == (batch_size, 1, 512, 512), f"Unexpected shape for output_seg: {output_seg.shape}"

    print("Test passed: Generator outputs correct shapes.")