from blocks import EncoderBlock, DecoderBlock
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_channels = 4, output_channels = 3):
        super(Generator, self).__init__()

        # Encoder
        self.encoder1 = EncoderBlock(input_channels, 64, use_bn=False)
        self.encoder2 = EncoderBlock(64, 128)                             
        self.encoder3 = EncoderBlock(128, 256)                            
        self.encoder4 = EncoderBlock(256, 512)                            
        self.encoder5 = EncoderBlock(512, 512)                            
        self.encoder6 = EncoderBlock(512, 512)                            
        self.encoder7 = EncoderBlock(512, 512, stride=(1,2))                            
        self.encoder8 = EncoderBlock(512, 512, use_bn=False)      

        # Shared Decoder blocks
        self.decoder1 = DecoderBlock(512, 512, use_dropout=False)
        self.decoder2 = DecoderBlock(512, 512, use_dropout=False)
        self.decoder3 = DecoderBlock(512, 512, use_dropout=False)
        self.decoder4 = DecoderBlock(512, 512, use_dropout=False)
        self.decoder5 = DecoderBlock(512, 256, use_dropout=False)
        self.decoder6 = DecoderBlock(256, 128)
        self.decoder7 = DecoderBlock(128, 64, use_dropout=False)

        # Independent Decoder for street view image
        self.decoder8_img = nn.Sequential(
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.5),
            nn.Tanh()
        )

        # Independent Decoder for street view segmentation map
        self.decoder8_seg = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )
   

    def forward(self, x):
        # Encoder forward
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
        bottleneck = self.encoder8(e7)

        # Controllo la dimensione del bottleneck
        print(f"Bottleneck shape: {bottleneck.shape}")
        bottleneck_reshaped = bottleneck.view(1, 512, 2, 2)
        print(f"Bottleneck reshaped shape: {bottleneck_reshaped.shape}")

        # Decoder forward (senza skip connections)
        d1 = self.decoder1(bottleneck_reshaped)
        print(f"d1 shape: {d1.shape}")
        d2 = self.decoder2(d1)
        print(f"d2 shape: {d2.shape}")
        d3 = self.decoder3(d2)
        print(f"d3 shape: {d3.shape}")
        d4 = self.decoder4(d3)
        print(f"d4 shape: {d4.shape}")
        d5 = self.decoder5(d4)
        print(f"d5 shape: {d5.shape}")
        d6 = self.decoder6(d5)
        print(f"d6 shape: {d6.shape}")
        d7 = self.decoder7(d6)

        # Controllo dimensione prima dell'ultimo upsampling
        print(f"Final feature map before upsampling: {d7.shape}")

        # Independent decoders forward
        streetview_img = self.decoder8_img(d7)
        streetview_seg = self.decoder8_seg(d7)

        print(f"Output image shape: {streetview_img.shape}")
        print(f"Output segmentation shape: {streetview_seg.shape}")

        return streetview_img, streetview_seg

    
if __name__ == "__main__":
    input_channels = 4
    output_channels = 3
    batch_size = 1
    input_tensor = torch.randn(batch_size, input_channels, 224, 1232)

    model = Generator(input_channels, output_channels)
    output_img, output_seg = model(input_tensor)

    assert output_img.shape == (batch_size, output_channels, 512, 512), f"Unexpected shape for output_img: {output_img.shape}"
    assert output_seg.shape == (batch_size, 1, 512, 512), f"Unexpected shape for output_seg: {output_seg.shape}"

    print("Test passed: Generator outputs correct shapes.")

