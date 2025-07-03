import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, middle_dim=256,
                 out_dim=2,
                 ):
        super(FlowHead, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, middle_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(middle_dim, out_dim, 3, padding=1)

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))

        return out

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_channels=324,
                 flow_channels=2,
                 out_channels=128
                 ):
        super(BasicMotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(corr_channels, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(flow_channels, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, out_channels - flow_channels, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)
    
class BasicUpdateBlock(nn.Module):
    def __init__(self, corr_channels=81,
                 hidden_dim=128,
                 context_dim=128,
                 flow_dim=2,
                 ):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(corr_channels=corr_channels, flow_channels=flow_dim, out_channels=128)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=context_dim + 128)

        self.flow_head = FlowHead(hidden_dim, middle_dim=hidden_dim*2, out_dim=flow_dim,)
        # self.mask = nn.Sequential(
        #     nn.Conv2d(hidden_dim, 256, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, downsample_factor ** 2 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)

        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # mask = self.mask(net)

        return net, delta_flow

class LiteMotionEncoder(nn.Module):
    def __init__(self, corr_channels=324,
                 flow_channels=2,
                 out_channels=64
                 ):
        super(LiteMotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(corr_channels, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(128, 96, 3, padding=1)
        self.convf1 = nn.Conv2d(flow_channels, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(32 + 96, out_channels - flow_channels, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class LiteUpdateBlock(nn.Module):
    def __init__(self, corr_channels=81,
                 hidden_dim=64,
                 context_dim=64,
                 flow_dim=2,
                 ):
        super(LiteUpdateBlock, self).__init__()
        self.encoder = LiteMotionEncoder(corr_channels=corr_channels, flow_channels=flow_dim, out_channels=64)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=context_dim + 64)
        self.flow_head = FlowHead(input_dim=hidden_dim, middle_dim=hidden_dim*2, out_dim=flow_dim)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)

        delta_flow = self.flow_head(net)

        return net, delta_flow
