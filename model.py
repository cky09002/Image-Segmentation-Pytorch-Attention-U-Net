import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Upsampling path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2): # 1st in upsampling 2nd concatenate.
            x = self.ups[idx](x) #Upsample
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x) #Double conv

        return self.final_conv(x)


class UNetPlusPlus(nn.Module):
    """Full UNet++ (Nested U-Net)"""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNetPlusPlus, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder blocks (level 0)
        self.conv0_0 = DoubleConv(in_channels, features[0])
        self.conv1_0 = DoubleConv(features[0], features[1])
        self.conv2_0 = DoubleConv(features[1], features[2])
        self.conv3_0 = DoubleConv(features[2], features[3])
        self.conv4_0 = DoubleConv(features[3], features[3]*2)  # Bottleneck

        # Decoder nested blocks
        self.conv0_1 = DoubleConv(features[0] + features[1], features[0])
        self.conv1_1 = DoubleConv(features[1] + features[2], features[1])
        self.conv2_1 = DoubleConv(features[2] + features[3], features[2])
        self.conv3_1 = DoubleConv(features[3] + features[3]*2, features[3])

        self.conv0_2 = DoubleConv(features[0]*2 + features[1], features[0])
        self.conv1_2 = DoubleConv(features[1]*2 + features[2], features[1])
        self.conv2_2 = DoubleConv(features[2]*2 + features[3], features[2])

        self.conv0_3 = DoubleConv(features[0]*3 + features[1], features[0])
        self.conv1_3 = DoubleConv(features[1]*3 + features[2], features[1])

        self.conv0_4 = DoubleConv(features[0]*4 + features[1], features[0])

        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Decoder with safe upsampling
        def upsample_to(src, target):
            """Upsample src to the spatial size of target"""
            return F.interpolate(src, size=target.shape[2:], mode='bilinear', align_corners=True)

        x0_1 = self.conv0_1(torch.cat([x0_0, upsample_to(x1_0, x0_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, upsample_to(x2_0, x1_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, upsample_to(x3_0, x2_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, upsample_to(x4_0, x3_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, upsample_to(x1_1, x0_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, upsample_to(x2_1, x1_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, upsample_to(x3_1, x2_0)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, upsample_to(x1_2, x0_0)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, upsample_to(x2_2, x1_0)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, upsample_to(x1_3, x0_0)], 1))

        return self.final_conv(x0_4)

class AttentionBlock(nn.Module):
    """Attention Gate for U-Net skip connections"""
    def __init__(self, F_g, F_l, F_int):
        """
        F_g : channels of gating signal (decoder feature)
        F_l : channels of skip connection (encoder feature)
        F_int : intermediate channels (usually smaller)
        """
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g : decoder feature (gating signal)
        x : encoder feature (skip connection)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi  # element-wise attention weighting


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder with Attention
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.attention_blocks.append(
                AttentionBlock(F_g=feature, F_l=feature, F_int=feature // 2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Upsample if needed
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            # Apply attention gate
            skip_connection = self.attention_blocks[idx // 2](x, skip_connection)

            # Concatenate + conv
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 3, 256, 256))
    model = UNet(in_channels=3, out_channels=1)
    preds = model(x)
    print(f"UNet - Input: {x.shape}, Output: {preds.shape}, Params: {sum(p.numel() for p in model.parameters()):,}")

    model_pp = UNetPlusPlus(in_channels=3, out_channels=1)
    preds_pp = model_pp(x)
    print(f"UNet++ - Input: {x.shape}, Output: {preds_pp.shape}, Params: {sum(p.numel() for p in model_pp.parameters()):,}")


if __name__ == "__main__":
    test()
