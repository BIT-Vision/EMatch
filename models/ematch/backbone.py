import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1,
                 ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)
    

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h
    
        
class RNNEncoder(nn.Module):
    def __init__(self, input_dim=15, output_dim=128, 
                 norm_layer=nn.InstanceNorm2d,
                 num_output_scales=1,
                 **kwargs,):
        super(RNNEncoder, self).__init__()
        self.feature_dims = [48, 64, 96, 128]
        self.num_branch = num_output_scales
        
        # Conv
        self.conv1 = nn.Conv2d(input_dim, self.feature_dims[0]//2, kernel_size=7, stride=2, padding=3, bias=False)  # 1/2
        self.norm1 = norm_layer(self.feature_dims[0]//2)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Lyaer0
        self.convGRU0 = ConvGRU(hidden_dim=self.feature_dims[0]//2, input_dim=self.feature_dims[0]//2) # 1/2

        # Layer1
        self.resblock1 = ResidualBlock(self.feature_dims[0], self.feature_dims[1]//2, norm_layer=norm_layer, stride=1, dilation=1) # 1/2
        self.convGRU1 = ConvGRU(hidden_dim=self.feature_dims[1]//2, input_dim=self.feature_dims[1]//2)
        
        # Layer2
        self.resblock2 = ResidualBlock(self.feature_dims[1], self.feature_dims[2]//2, norm_layer=norm_layer, stride=2, dilation=1) # 1/4
        self.convGRU2 = ConvGRU(hidden_dim=self.feature_dims[2]//2, input_dim=self.feature_dims[2]//2)
        
        # Layer3
        self.resblock3 = ResidualBlock(self.feature_dims[2], self.feature_dims[3]//2, norm_layer=norm_layer, stride=2, dilation=1) # 1/8
        self.convGRU3 = ConvGRU(hidden_dim=self.feature_dims[3]//2, input_dim=self.feature_dims[3]//2)

        # Conv
        self.conv2 = nn.Conv2d(self.feature_dims[3]//2, output_dim, 1, 1, 0)
        if self.num_branch >= 2:
            self.conv3 = nn.Conv2d(self.feature_dims[2]//2, output_dim, 1, 1, 0)
        if self.num_branch >= 3:
            self.conv4 = nn.Conv2d(self.feature_dims[1]//2, output_dim, 1, 1, 0)
            
        # Initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, nets=None, F_out=True):
        # Conv
        c0 = x 
        c0 = self.conv1(c0)
        c0 = self.norm1(c0)
        c0 = self.relu1(c0)

        # Layer0
        net0 = self.convGRU0(nets[0], c0) if nets is not None else self.convGRU0(c0, c0)

        # Layer1
        c1 = torch.concat([c0,net0], dim=1)
        c1 = self.resblock1(c1)
        net1 = self.convGRU1(nets[1], c1) if nets is not None else self.convGRU1(c1, c1)

        # Layer2
        c2 = torch.concat([c1,net1], dim=1)
        c2 = self.resblock2(c2)
        net2 = self.convGRU2(nets[2], c2) if nets is not None else self.convGRU2(c2, c2)

        # Layer3
        c3 = torch.concat([c2,net2], dim=1)
        c3 = self.resblock3(c3)
        net3 = self.convGRU3(nets[3], c3) if nets is not None else self.convGRU3(c3, c3)

        # Output
        nets = [net0, net1, net2, net3]
        if F_out:
            if self.num_branch > 1:
                if self.num_branch == 2:
                    y1 = self.conv2(net3)
                    y2 = self.conv3(net2)
                    out = [y2,y1] # resolution from hight to low
                elif self.num_branch == 3:
                    y1 = self.conv2(net3)
                    y2 = self.conv3(net2)
                    y3 = self.conv4(net1)
                    out = [y3,y2,y1] # resolution from hight to low
                else:
                    raise NotImplementedError
            else:
                y1 = self.conv2(net3)
                out = [y1]
            
            return out, nets
        else:
            return None, nets
