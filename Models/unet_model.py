import torch.nn as nn


class UnetModel(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.layer1 = self.conv_block(n_channels, 64)
        self.layer2 = self.conv_block(64, 128)
        self.layer3 = self.conv_block(128, 256)
        self.layer4 = self.conv_block(256, 512)
        self.layer5 = self.conv_block(512, 1024)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.layer1_t = self.conv_block(1024, 512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.layer2_t = self.conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.layer3_t = self.conv_block(256, 128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.layer4_t = self.conv_block(128, 64)
        self.convout = nn.ConvTranspose2d(64, n_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(self.pool(layer1_out))
        layer3_out = self.layer3(self.pool(layer2_out))
        layer4_out = self.layer4(self.pool(layer3_out))
        layer5_out = self.layer5(self.pool(layer4_out))

        layer1_t = self.layer1_t(
            torch.cat([self.upconv1(layer5_out), layer4_out], dim=1)
        )
        layer2_t = self.layer2_t(torch.cat([self.upconv2(layer1_t), layer3_out], dim=1))
        layer3_t = self.layer3_t(torch.cat([self.upconv3(layer2_t), layer2_out], dim=1))
        layer4_t = self.layer4_t(torch.cat([self.upconv4(layer3_t), layer1_out], dim=1))

        out = self.convout(layer4_t)
        return out
