import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from nn_utils import TemporalBatchMixin

class ResidualBlock(nn.Module):
    """Standard residual block with skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    

class ResNet5(TemporalBatchMixin,nn.Module): 
    """
    A lightweight ResNet with 5 layers (2 blocks).
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """ 

    def __init__(self, in_d, h_d, out_d, s1=1, s2=1, s3=1, avg_pool=False):
        super().__init__()
        self.avg_pool = avg_pool
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(h_d, h_d, stride=s1)
        self.layer2 = ResidualBlock(h_d, h_d * 2, stride=s2)
        self.layer3 = ResidualBlock(h_d * 2, out_d, stride=s3)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) if avg_pool else torch.nn.Identity()

    def _forward(self,x): 
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        if self.avg_pool:
            out = out.flatten(1)
        return out 
    

class SimplePredictor(nn.Module): 
    """Wrapper that concatenates states and actions channel-wise before prediction."""

    def __init__(self,predictor, context_length):
        super().__init__()
        self.predictor = predictor
        self.is_rnn = predictor.is_rnn
        self.context_length = context_length

    def forward(self, x, a):
        return self.predictor(torch.cat([x,a], dim=1))
    

class StateOnlyPredictor(SimplePredictor): 
    """Wrapper for a simple predictor which concatenates states and actions channel wise.""" 

    def forward(self, x, a):
        # action not used on purpose 
        prev_state = x[:, :, :-1] # [B, C, T-1, H, W]
        next_state = x[:, :, 1:]  # [B, C, T-1, H, W]
        combined_xa = torch.cat([prev_state, next_state], dim= 1)
        return self.predictor(combined_xa)
    

class ResUNet(TemporalBatchMixin, nn.Module): 
    """
    A small UNet with residual encoder blocks and transposed-conv upsampling. 
    Channels scale like h, 2h, 4h, 8h. Output keeps the input HxW.
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(self, in_d, h_d, out_d, is_rnn = False): 
        super().__init__()
        self.is_rnn = is_rnn
        # Stem 
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)

        # Encoder 
        self.enc1 = ResidualBlock(h_d, h_d, stride=1)           # H, W
        self.enc2 = ResidualBlock(h_d, 2 * h_d, stride=2)       # H/2, W/2
        self.enc3 = ResidualBlock(2 * h_d, 4 * h_d, stride=2)   # H/4, W/4 
        self.bott = ResidualBlock(4 * h_d, 8 * h_d, stride=2)   # H/8, W/8 

        # Decoder upsamples, then fuses skip with a residual block that reduces channels
        self.up3 = nn.ConvTranspose2d(8 * h_d, 4 * h_d, kernel_size=2, stride=2) 
        self.dec3 = ResidualBlock(8 * h_d, 4 * h_d, stride = 1)

        self.up2 = nn.ConvTranspose2d(4 * h_d, 2 * h_d, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(4 * h_d, 2 * h_d, stride=1)

        self.up1 = nn.ConvTranspose2d(2 * h_d, 1 * h_d, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(2 * h_d, 1 * h_d, stride=1)

        # Head 
        self.head = nn.Conv2d(h_d, out_d, kernel_size=1)

    @staticmethod
    def _match_size(x,ref):
        # Guards againt odd input sizes by resizing the upsample to the skip spatial dims
        if x.shape[-2:] != ref.shape[-2:]: 
            x = F.interpolate(
                x, size=ref.shape[-2:], mode="bilinear", align_corners=False
            )
        return x 
    
    def _forward(self,x):
        x0 = self.relu(self.bn1(self.conv1(x)))

        # Encoder with skips 
        s1 = self.enc1(x0)  # h
        s2 = self.enc2(s1)  # 2h
        s3 = self.enc3(s2)  # 4h
        b = self.bott(s3)   # 8h

        # Decoder stage 3 
        d3 = self.up3(b)
        d3 = self._match_size(d3, s3)
        d3 = torch.cat([d3,s3], dim=1) # 4h + 4h = 8h 
        d3 = self.dec3(d3)  # → 4h

        # Decoder stage 2
        d2 = self.up2(d3)
        d2 = self._match_size(d2,s2)
        d2 = torch.cat([d2,s2], dim=1)
        d2 = self.dec2(2)

        # Decoder stage 1
        d1 = self.up1(d2)
        d1 = self._match_size(d1,s1)
        d1 = torch.cat([d1,s1], dim=1)  # h + h = 2h 
        d1 = self.dec1(d1) # → h

        out = self.head(d1) # → out_d channels
        return out 



class Projector(nn.Module): 
    """MLP projector built from a spec string like '256-512-128'."""

    def __init__(self,mlp_spec): 
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]    # Store output dimension as attribute

    def forward(self, x):
        return self.net(x)