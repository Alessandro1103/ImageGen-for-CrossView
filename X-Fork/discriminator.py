from blocks import EncoderBlock
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels = 6):
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

        batch_size = x.shape[0]

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        
        out = self.final_conv(e7)
        out = self.sigmoid(out)
        
        assert out.shape == (batch_size, 1, 1, 1), f"Unexpected shape for discriminator output: {out.shape}"

        out = out.view(batch_size, -1)

        assert out.shape == (batch_size, 1), f"Unexpected shape for discriminator output: {out.shape}"

        return out


if __name__ == "__main__":
    input_channels = 6  # 3 per immagine reale + 3 per immagine generata
    batch_size = 1 
    input_image = torch.randn(batch_size, input_channels, 512, 512)
    model = Discriminator(input_channels)
    output = model(input_image)

    print("Test passed: Discriminator outputs correct shape.")