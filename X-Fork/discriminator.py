from blocks import EncoderBlock
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        
        # Ultimo livello convoluzionale per ridurre la dimensionalità
        self.final_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.encoder1(x)
        print(f"e1 shape: {e1.shape}")
        e2 = self.encoder2(e1)
        print(f"e2 shape: {e2.shape}")
        e3 = self.encoder3(e2)
        print(f"e3 shape: {e3.shape}")
        e4 = self.encoder4(e3)
        print(f"e4 shape: {e4.shape}")
        e5 = self.encoder5(e4)
        print(f"e5 shape: {e5.shape}")
        e6 = self.encoder6(e5)
        print(f"e6 shape: {e6.shape}")
        e7 = self.encoder7(e6)
        print(f"e7 shape: {e7.shape}")
        
        out = self.final_conv(e7)
        out = self.sigmoid(out)
        
        print(f"out shape: {out.shape}")
        
        return out


def test():
    input_channels = 6  # 3 per immagine reale + 3 per immagine generata
    batch_size = 1 
    input_image = torch.randn(batch_size, input_channels, 512, 512)
    model = Discriminator(input_channels)
    output = model(input_image)

    assert output.shape == (batch_size, 1, 1, 1), f"Unexpected shape for discriminator output: {output.shape}"

    print("Test passed: Discriminator outputs correct shape.")

test()
