# utils/model_arch.py

import torch
import torch.nn as nn
from models.network_swinir import SwinIR  
from torchvision.ops import DeformConv2d

# MFSR_SwinIR class definition f
class MFSR_SwinIR(nn.Module):
    """
    A Multi-Frame Super-Resolution model using a SwinIR backbone.
    This architecture performs frame alignment and fusion before reconstruction.
    """
    def __init__(self, swinir_backbone, num_frames=5):
        super(MFSR_SwinIR, self).__init__()
        self.backbone = swinir_backbone
        embed_dim = self.backbone.embed_dim 

        # Layers for Feature Alignment (using Deformable Convolution)
        self.conv_first = nn.Conv2d(1, embed_dim, 3, 1, 1)
        self.offset_conv1 = nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1) # Takes reference and neighbor features
        self.offset_conv2 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.offset_conv3 = nn.Conv2d(embed_dim, 18, 3, 1, 1) # Output is 18 for DeformConv2d (2 * kernel_size^2)
        self.dcn = DeformConv2d(embed_dim, embed_dim, 3, padding=1)
        
        # Layer for Feature Fusion
        self.fusion_conv = nn.Conv2d(embed_dim * num_frames, embed_dim, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
    def forward(self, x):
        # Input x has shape [B, N, C, H, W] where N is num_frames
        b, n, c, h, w = x.size()
        ref_idx = n // 2 # Use the middle frame as the reference
        
        # Extract initial features for all frames at once
        features = self.conv_first(x.view(b * n, c, h, w)).view(b, n, -1, h, w)
        
        ref_features = features[:, ref_idx, :, :, :]
        aligned_features = []
        
        # Align each neighbor frame to the reference frame
        for i in range(n):
            if i == ref_idx:
                aligned_features.append(ref_features)
            else:
                neighbor_features = features[:, i, :, :, :]
                offsets = self.lrelu(self.offset_conv1(torch.cat([ref_features, neighbor_features], dim=1)))
                offsets = self.lrelu(self.offset_conv2(offsets))
                offsets = self.lrelu(self.offset_conv3(offsets))
                aligned_features.append(self.dcn(neighbor_features, offsets))
                
        # Fuse the aligned features into a single feature map
        fused_features = self.fusion_conv(torch.cat(aligned_features, dim=1))
        
        # Pass the fused features through the main body of the SwinIR backbone
        deep_features = self.backbone.conv_after_body(self.backbone.forward_features(fused_features)) + fused_features
        
        features_before_upsampling = self.backbone.conv_before_upsample(deep_features)
        out = self.backbone.upsample(features_before_upsampling)
        out = self.backbone.conv_last(out)
        
        return out