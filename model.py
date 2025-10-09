import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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

        return torch.sigmoid(self.final_conv(x))


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

        return torch.sigmoid(self.final_conv(x0_4))

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

        return torch.sigmoid(self.final_conv(x))

class R2Block(nn.Module):
    def __init__(self, channels, t = 2):
        super(R2Block,self).__init__()
        self.t = t
        self.init_conv = nn.Sequential(nn.Conv2d(channels,channels*2,3,padding = 1, bias = False),nn.BatchNorm2d(channels*2),nn.ReLU(inplace = True))
    
        # Recurrent layer
        self.recurrent = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels*2, channels*2, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels*2),
                nn.ReLU(inplace=True)
            ) for _ in range(t)
        ])
        
    def forward(self,x):
        init_result = self.init_conv(x)
        out = init_result
        for layer in self.recurrent:
            out = layer(out) + out
        return out + init_result


class AttentionR2UNet(AttentionUNet):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], t = 2):
        super(AttentionR2UNet,self).__init__(in_channels, out_channels, features)
        self.bottleneck = R2Block(features[-1], t=t)

class PretrainedAttentionR2UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, backbone='resnet34', pretrained=True, t=2):
        super().__init__()
        
        # Load pretrained ResNet backbone
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            filters = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            filters = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError("Backbone not supported")

        # Handle different input channels
        if in_channels != 3:
            original_conv1 = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels, original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False
            )
            # Initialize with pretrained weights if possible
            if pretrained and in_channels == 1:
                # For grayscale, average RGB weights
                resnet.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
            elif pretrained and in_channels > 3:
                # Repeat weights for additional channels
                repeat_factor = in_channels // 3
                resnet.conv1.weight.data = original_conv1.weight.data.repeat(1, repeat_factor, 1, 1)

        # Encoder: Use ResNet layers
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Bottleneck with R2Block
        self.bottleneck = R2Block(filters[-1], t=t)

        # Calculate correct channel dimensions for decoder
        # After bottleneck, channels become filters[-1] * 2
        bottleneck_out = filters[-1] * 2
        
        # Decoder with corrected channel dimensions
        self.upconv4 = nn.ConvTranspose2d(bottleneck_out, filters[-2], kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=filters[-2], F_l=filters[-2], F_int=filters[-2]//2)
        self.dec4 = DoubleConv(filters[-2] + filters[-2], filters[-2])  # x3_att + d4

        self.upconv3 = nn.ConvTranspose2d(filters[-2], filters[-3], kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=filters[-3], F_l=filters[-3], F_int=filters[-3]//2)
        self.dec3 = DoubleConv(filters[-3] + filters[-3], filters[-3])  # x2_att + d3

        self.upconv2 = nn.ConvTranspose2d(filters[-3], filters[-4], kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=filters[-4], F_l=filters[-4], F_int=filters[-4]//2)
        self.dec2 = DoubleConv(filters[-4] + filters[-4], filters[-4])  # x1_att + d2

        self.upconv1 = nn.ConvTranspose2d(filters[-4], filters[0], kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=filters[0], F_l=filters[0], F_int=filters[0]//2)
        self.dec1 = DoubleConv(filters[0] + filters[0], filters[0])  # x0_att + d1

        # Final conv with interpolation to ensure exact output size
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        
        # Initialize weights for new layers
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Store input size for final output
        input_size = x.size()[2:]
        
        # Encoder
        x0 = self.initial(x)         # [B, 64, H/2, W/2]
        x1 = self.encoder1(self.maxpool(x0))  # [B, filters[1], H/4, W/4]
        x2 = self.encoder2(x1)       # [B, filters[2], H/8, W/8]
        x3 = self.encoder3(x2)       # [B, filters[3], H/16, W/16]
        x4 = self.encoder4(x3)       # [B, filters[4], H/32, W/32]

        # Bottleneck
        x_bottleneck = self.bottleneck(x4)  # [B, filters[4]*2, H/32, W/32]

        # Decoder stage 4
        d4 = self.upconv4(x_bottleneck)  # [B, filters[3], H/16, W/16]
        x3_att = self.att4(g=d4, x=x3)   # [B, filters[3], H/16, W/16]
        d4 = torch.cat([x3_att, d4], dim=1)  # [B, filters[3]*2, H/16, W/16]
        d4 = self.dec4(d4)  # [B, filters[3], H/16, W/16]

        # Decoder stage 3
        d3 = self.upconv3(d4)  # [B, filters[2], H/8, W/8]
        x2_att = self.att3(g=d3, x=x2)   # [B, filters[2], H/8, W/8]
        d3 = torch.cat([x2_att, d3], dim=1)  # [B, filters[2]*2, H/8, W/8]
        d3 = self.dec3(d3)  # [B, filters[2], H/8, W/8]

        # Decoder stage 2
        d2 = self.upconv2(d3)  # [B, filters[1], H/4, W/4]
        x1_att = self.att2(g=d2, x=x1)   # [B, filters[1], H/4, W/4]
        d2 = torch.cat([x1_att, d2], dim=1)  # [B, filters[1]*2, H/4, W/4]
        d2 = self.dec2(d2)  # [B, filters[1], H/4, W/4]

        # Decoder stage 1
        d1 = self.upconv1(d2)  # [B, filters[0], H/2, W/2]
        x0_att = self.att1(g=d1, x=x0)   # [B, filters[0], H/2, W/2]
        d1 = torch.cat([x0_att, d1], dim=1)  # [B, filters[0]*2, H/2, W/2]
        d1 = self.dec1(d1)  # [B, filters[0], H/2, W/2]

        # Final convolution and upsample to exact input size
        out = self.final_conv(d1)  # [B, out_channels, H/2, W/2]
        
        # Upsample to match input spatial dimensions
        if out.size()[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        return torch.sigmoid(out)

def test_improved_model():
    x = torch.randn((8, 3, 512, 512))  # batch size 8, 3-channel RGB, 512x512 images
    
    # Test Improved Pretrained Attention R2 UNet
    model = PretrainedAttentionR2UNet(in_channels=3, out_channels=1, pretrained=True)
    preds = model(x)
    print(f"Improved Pretrained Attention R2 UNet - Input: {x.shape}, Output: {preds.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with different backbones
    model_res50 = PretrainedAttentionR2UNet(in_channels=3, out_channels=1, backbone='resnet50', pretrained=True)
    preds_res50 = model_res50(x)
    print(f"ResNet50 Backbone - Input: {x.shape}, Output: {preds_res50.shape}")
    print(f"Params: {sum(p.numel() for p in model_res50.parameters()):,}")
    
    # Test with different input channels
    model_grayscale = PretrainedAttentionR2UNet(in_channels=1, out_channels=1, pretrained=True)
    x_grayscale = torch.randn((8, 1, 512, 512))
    preds_grayscale = model_grayscale(x_grayscale)
    print(f"Grayscale Input - Input: {x_grayscale.shape}, Output: {preds_grayscale.shape}")

if __name__ == "__main__":
    test_improved_model()
