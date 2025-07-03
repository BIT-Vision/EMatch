import torch.nn as nn
import torch.nn.functional as F
import math

# FPN类
class FPN(nn.Module):
    def __init__(self, feature_dims=None):
        super(FPN, self).__init__()
        self.inplanes = 64
        self.feature_dims = feature_dims if feature_dims is not None else [24, 32, 48, 64]
        # self.feature_dims = [24, 48, 96, 192]
        # 横向连接，保证通道数目相同
        self.latlayer3 = nn.Conv2d(self.feature_dims[3], self.inplanes, 1, 1, 0) 
        self.latlayer2 = nn.Conv2d(self.feature_dims[2], self.inplanes, 1, 1, 0) 
        self.latlayer1 = nn.Conv2d(self.feature_dims[1], self.inplanes, 1, 1, 0)
        self.latlayer0 = nn.Conv2d(self.feature_dims[0], self.inplanes, 1, 1, 0)
        # 代表3*3的卷积融合，目的是消除上采样过程带来的重叠效应，以生成最终的特征图
        self.smooth3 = nn.Conv2d(self.inplanes, self.feature_dims[3], 3, 1, 1)
        self.smooth2 = nn.Conv2d(self.inplanes, self.feature_dims[2], 3, 1, 1)
        self.smooth1 = nn.Conv2d(self.inplanes, self.feature_dims[1], 3, 1, 1)
        self.smooth0 = nn.Conv2d(self.inplanes, self.feature_dims[0], 3, 1, 1)
        
    def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, nets):
        # 自下而上
        net0, net1, net2, net3 = nets
        # 自上而下
        p3 = self.latlayer3(net3)
        p2 = self._upsample_add(p3, self.latlayer2(net2))
        p1 = self._upsample_add(p2, self.latlayer1(net1))
        p0 = self._upsample_add(p1, self.latlayer0(net0))
        ###卷积融合，平滑处理
        net3 = self.smooth3(p3)
        net2 = self.smooth2(p2)
        net1 = self.smooth1(p1)
        net0 = self.smooth0(p0)

        return [net0,net1,net2,net3]
