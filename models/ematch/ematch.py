import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import RNNEncoder
from .FPN import FPN
from .transformer import FeatureTransformer
from .matching import (global_correlation_softmax, local_correlation_softmax,
                       global_correlation_softmax_stereo, local_correlation_softmax_stereo,)
from .geometry import flow_warp
from .reg_refine import BasicUpdateBlock
from .utils import feature_add_position, upsample_flow_with_mask, CorrBlock, coords_grid


class EventMatch(nn.Module):
    def __init__(self, cfgs):
        super(EventMatch, self).__init__()
        self.input_dim = cfgs['input_dim']
        self.feature_channels = cfgs['feature_channels']
        self.split = cfgs['split']
        self.interval = int(self.input_dim/self.split)

        self.num_scales = cfgs['num_scales']
        self.upsample_factor = cfgs['upsample_factor']
        self.attn_type = cfgs['attn_type']
        self.attn_splits_list = cfgs['attn_splits_list']
        self.corr_radius_list = cfgs['corr_radius_list']
        self.prop_radius_list = cfgs['prop_radius_list']
        assert len(self.attn_splits_list) == len(self.corr_radius_list) == len(self.prop_radius_list) == self.num_scales

        self.num_head = cfgs['num_head']
        self.ffn_dim_expansion = cfgs['ffn_dim_expansion']
        self.num_transformer_layers = cfgs['num_transformer_layers']
        
        # TRN
        self.backbone = RNNEncoder(input_dim=self.interval, output_dim=self.feature_channels, num_output_scales=self.num_scales)
        self.fpn = FPN()

        # SCA
        self.transformer = FeatureTransformer(num_layers=self.num_transformer_layers,
                                              d_model=self.feature_channels,
                                              nhead=self.num_head,
                                              ffn_dim_expansion=self.ffn_dim_expansion,)

        # Refinement
        self.num_refine = cfgs['num_refine']
        if 'flow' in cfgs['tasks']:
            # optional regression refinement
            self.refine_proj_flow = nn.Conv2d(self.feature_channels, 256, 1)
            self.refine_flow = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                            hidden_dim=128,
                                            context_dim=128,
                                            flow_dim=2)
        if 'disparity' in cfgs['tasks']:
            # optional regression refinement
            self.refine_proj_disparity = nn.Conv2d(self.feature_channels, 256, 1)
            self.refine_disparity = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                            hidden_dim=128,
                                            context_dim=128,
                                            flow_dim=1)
        
        self.upsampler = nn.Sequential(nn.Conv2d(2 + self.feature_channels, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, self.upsample_factor ** 2 * 9, 1, 1, 0))

    def extract_feature(self, event_0, event_1):
        event_concat = torch.cat((event_0, event_1), dim=0)

        # RNN Encoding
        events = torch.chunk(event_concat, self.split, dim=1)
        nets = None
        for i,slice in enumerate(events):
            features, nets = self.backbone(slice, nets, F_out=i==self.split-1)
            if i != self.split - 1:
                nets = self.fpn(nets)

        # reverse: resolution from low to high
        features = features[::-1]
        feature_0, feature_1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature_0.append(chunks[0])
            feature_1.append(chunks[1])
            
        return feature_0, feature_1
    
    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8, mask=None):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor
        else:
            # concat = torch.cat((flow, feature), dim=1)
            # mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=upsample_factor)

        return up_flow
    
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
            
    def forward(self, event_0, event_1, task='flow'):
        # 0. initial
        results_dict = {}
        flow_preds = []
        flow = None

        ########################################################## 1. Extraction ##########################################################
        # list of features, resolution low to high
        feature_0_list, feature_1_list = self.extract_feature(event_0, event_1)  # list of features

        # multi scales
        for scale_idx in range(self.num_scales):
            # 1) update state
            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            attn_splits = self.attn_splits_list[scale_idx]
            corr_radius = self.corr_radius_list[scale_idx]
            prop_radius = self.prop_radius_list[scale_idx]

            feature_0, feature_1 = feature_0_list[scale_idx], feature_1_list[scale_idx]
            feature_0_init, feature_1_init = feature_0, feature_1

            ########################################################## 2. Transformers ########################################################
            # 2) transformer
            # 2.1) warp features
            if flow is not None:
                flow = flow.detach()
                if task == 'disparity':
                    # construct flow vector for disparity
                    # flow here is actually disparity
                    zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                    # NOTE: reverse disp, disparity is positive
                    displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                    feature_1 = flow_warp(feature_1, displace)  # [B, C, H, W]
                elif task == 'flow':
                    feature_1 = flow_warp(feature_1, flow)  # [B, C, H, W]

            # 2.2) encoding
            # add position to features
            feature_0, feature_1 = feature_add_position(feature_0, feature_1, attn_splits, self.feature_channels)
            
            # Transformer
            feature_0, feature_1 = self.transformer(feature_0, feature_1,
                                                    attn_type=self.attn_type,
                                                    attn_num_splits=attn_splits,
                                                    )

            ########################################################## 3. Matching ############################################################
            # 4) matching
            if corr_radius == -1:  # global matching
                if task == 'flow':
                    flow_pred = global_correlation_softmax(feature_0, feature_1)[0]
                elif task == 'disparity':
                    flow_pred = global_correlation_softmax_stereo(feature_0, feature_1)[0]
            else:  # local matching
                if task == 'flow':
                    flow_pred = local_correlation_softmax(feature_0, feature_1, corr_radius)[0]
                elif task == 'disparity':
                    flow_pred = local_correlation_softmax_stereo(feature_0, feature_1, corr_radius)[0]
            
            # 4) flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred
            flow = flow.clamp(min=0)  if task == 'disparity' else flow
            # upsample to the original resolution for supervison at training time only
            # bilinear upsampling at training time except the last one

            if self.training:
                if scale_idx < self.num_scales - 1:
                    flow_up = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
                    flow_preds.append(flow_up)
                else:
                    if task == 'flow':
                        concat = torch.cat((flow, feature_0), dim=1)
                        mask = self.upsampler(concat)
                        flow_up = self.upsample_flow(flow, feature_0, upsample_factor=upsample_factor, mask=mask)
                        flow_preds.append(flow_up)
                    else:
                        flow_pad = torch.cat((-flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                        concat = torch.cat((flow_pad, feature_0), dim=1)
                        mask = self.upsampler(concat)
                        flow_up_pad = self.upsample_flow(flow_pad, feature_0, upsample_factor=upsample_factor, mask=mask)
                        flow_up = -flow_up_pad[:, :1]  # [B, 1, H, W]
                        flow_preds.append(flow_up)
            else:
                if scale_idx == self.num_scales - 1:
                    if task == 'flow':
                        concat = torch.cat((flow, feature_0), dim=1)
                        mask = self.upsampler(concat)
                    else:
                        flow_pad = torch.cat((-flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                        concat = torch.cat((flow_pad, feature_0), dim=1)
                        mask = self.upsampler(concat)
            
            ######################################################### 5. Refinement ############################################################
            corr_fn = CorrBlock(feature_0, feature_1, num_levels=1, radius=4)
            coords0, coords1 = self.initialize_flow(feature_0)
            coords1 = coords1 + flow
            if task == 'flow':
                flow = flow.detach()
                proj = self.refine_proj_flow(feature_0)
                net, inp = torch.chunk(proj, chunks=2, dim=1)
                net = torch.tanh(net)
                inp = torch.relu(inp)
            elif task == 'disparity':
                flow = flow.detach()
                proj = self.refine_proj_disparity(feature_0)
                net, inp = torch.chunk(proj, chunks=2, dim=1)
                net = torch.tanh(net)
                inp = torch.relu(inp)
            # 6) Refinement            
            for refine_idx in range(self.num_refine):
                if task == 'flow':
                    flow = flow.detach()
                    # 6.1) cost volume
                    correlation = corr_fn(coords1) # index correlation volume
                    # 6.2) refine
                    net, residual_flow = self.refine_flow(net, inp, correlation, flow.clone())
                    # 6.3) residual flow
                    flow = flow + residual_flow
                    coords1 = coords1 + residual_flow
                elif task == 'disparity':
                    flow = flow.detach()
                    # 6.1) cost volume
                    zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                    correlation = corr_fn(coords1) # index correlation volume
                    # 6.2) refine
                    net, residual_flow = self.refine_disparity(net, inp, correlation, flow.clone())
                    # 6.3) residual flow
                    flow = flow + residual_flow
                    coords1 = coords1 + residual_flow + (flow.clamp(min=0) - flow)
                    flow = flow.clamp(min=0)  # positive

                if scale_idx < self.num_scales - 1 or refine_idx < self.num_refine - 1:
                    if self.training:
                        flow_up = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
                        flow_preds.append(flow_up)
                else:
                    if task == 'flow':
                        flow_up = self.upsample_flow(flow, None, upsample_factor=upsample_factor, mask=mask)
                        flow_preds.append(flow_up)
                    else:
                        flow_pad = torch.cat((-flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                        flow_up_pad = self.upsample_flow(flow_pad, None, upsample_factor=upsample_factor, mask=mask)
                        flow_up = -flow_up_pad[:, :1]  # [B, 1, H, W]
                        flow_preds.append(flow_up)
                
                
        if task == 'flow':
            results_dict.update({'flow_preds': flow_preds})
        elif task == 'disparity':
            for i in range(len(flow_preds)):
                flow_preds[i] = flow_preds[i].squeeze(1)  # [B, H, W]
            results_dict.update({'disparity_preds': flow_preds})

        return results_dict
   