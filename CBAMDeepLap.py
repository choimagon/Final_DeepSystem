import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from network.backbone import xception
#애포치 25마무리
# ----------------------------- Attention Modules -----------------------------
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, C, H * W).permute(1, 0, 2, 3)  # (3, B, C, HW)
        q, k, v = qkv[0], qkv[1], qkv[2]                               # (B, C, HW)
        attn = torch.bmm(q.transpose(1, 2), k) / (C ** 0.5)           # (B, HW, HW)
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2)      # (B, C, HW)
        return self.proj(out.reshape(B, C, H, W))
    
class CBAM(nn.Module):
    def __init__(self, dim, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel Attention
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False)
        )
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_pool_max = nn.AdaptiveMaxPool2d(1)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # --- Channel Attention ---
        avg_pool = self.channel_pool(x).view(B, C)
        max_pool = self.channel_pool_max(x).view(B, C)
        channel_attn = self.mlp(avg_pool) + self.mlp(max_pool)
        channel_attn = self.sigmoid_channel(channel_attn).view(B, C, 1, 1)
        x = x * channel_attn.expand_as(x)

        # --- Spatial Attention ---
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.conv_spatial(torch.cat([avg_out, max_out], dim=1))
        spatial_attn = self.sigmoid_spatial(spatial_attn)
        x = x * spatial_attn.expand_as(x)

        return x  # shape 유지: [B, C, H, W]
class CBAMCrossAttention(nn.Module):
    def __init__(self, dim, reduction=16, kernel_size=7):
        super(CBAMCrossAttention, self).__init__()
        # Channel attention
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, q_feat, kv_feat):
        B, C, H, W = q_feat.shape

        # --- Channel Attention ---
        avg_pool = F.adaptive_avg_pool2d(kv_feat, 1).view(B, C)
        max_pool = F.adaptive_max_pool2d(kv_feat, 1).view(B, C)
        channel_attn = self.mlp(avg_pool) + self.mlp(max_pool)
        channel_attn = self.sigmoid_channel(channel_attn).view(B, C, 1, 1)
        out = q_feat * channel_attn.expand_as(q_feat)

        # --- Spatial Attention ---
        avg_out = torch.mean(kv_feat, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(kv_feat, dim=1, keepdim=True)
        spatial_attn = self.conv_spatial(torch.cat([avg_out, max_out], dim=1))
        spatial_attn = self.sigmoid_spatial(spatial_attn)
        out = out * spatial_attn.expand_as(out)

        return out  # [B, C, H, W]

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.kv_proj = nn.Conv2d(dim, dim * 2, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q_feat, kv_feat):
        B, C, H, W = q_feat.shape
        HW = H * W

        q = self.q_proj(q_feat).reshape(B, C, HW).permute(0, 2, 1)  # (B, HW, C)
        kv = self.kv_proj(kv_feat).reshape(B, 2, C, HW)
        k = kv[:, 0]  # (B, C, HW)
        v = kv[:, 1]  # (B, C, HW)

        attn = torch.bmm(q, k) / (C ** 0.5)  # (B, HW, HW)
        attn = attn.softmax(dim=-1)

        out = torch.bmm(attn, v.permute(0, 2, 1))  # (B, HW, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return self.out_proj(out)


# ----------------------------- ASPP Modules -----------------------------
# class ASPP(nn.Module):
#     def __init__(self, in_channels, atrous_rates):
#         super(ASPP, self).__init__()
#         out_channels = 256
#         self.convs = nn.ModuleList()

#         self.convs.append(nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.SiLU(inplace=True)))

#         for rate in atrous_rates:
#             self.convs.append(nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
#                 nn.BatchNorm2d(out_channels),
#                 nn.SiLU(inplace=True)))

#         self.convs.append(nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.SiLU(inplace=True)))

#         self.project = nn.Sequential(
#             nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1))

#     def forward(self, x):
#         res = []
#         for i, conv in enumerate(self.convs):
#             out = conv(x)
#             if i == len(self.convs) - 1:
#                 out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
#             res.append(out)
#         return self.project(torch.cat(res, dim=1))
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

# ----------------------------- Decoder Head -----------------------------
class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(in_channels, aspp_dilate)
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, features, input_shape=None):
        x = self.aspp(features['out'])  # [B, 256, H, W]
        low_level = self.project(features['low_level'])  # [B, 48, H/4, W/4]
        x = F.interpolate(x, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        x = self.decoder(torch.cat((x, low_level), dim=1))
        if input_shape is not None:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

# ----------------------------- Main Model -----------------------------
class DualDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21, output_stride=16, pretrained_backbone=True):
        super().__init__()
        if output_stride == 8:
            replace_stride_with_dilation = [False, False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation = [False, False, False, True]
            aspp_dilate = [6, 12, 18]

        # --- Backbone ---
        backbone1 = xception.xception(
            pretrained='imagenet' if pretrained_backbone else False,
            replace_stride_with_dilation=replace_stride_with_dilation)
        backbone2 = xception.xception(
            pretrained='imagenet' if pretrained_backbone else False,
            replace_stride_with_dilation=replace_stride_with_dilation)
        
        # --- 변경된 return_layers ---
        return_layers = {'conv3': 'out', 'block1': 'low_level'}
        self.backbone1 = IntermediateLayerGetter(backbone1, return_layers=return_layers)
        self.backbone2 = IntermediateLayerGetter(backbone2, return_layers=return_layers)
        # --- 변경된 channel reduce ---
        self.reduce1 = nn.Sequential(
            nn.Conv2d(1536, 1536, kernel_size=3, padding=1, groups=1536, bias=False),  # Depthwise 3x3
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            nn.Conv2d(1536, 512, kernel_size=1, bias=False),                # Pointwise
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.reduce2 = nn.Sequential(
            nn.Conv2d(1536, 1536, kernel_size=3, padding=1, groups=1536, bias=False),  # Depthwise 3x3
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            nn.Conv2d(1536, 512, kernel_size=1, bias=False),                # Pointwise
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )


        # --- attention 모듈 dim 일괄 512로 변경 ---
        self.self_attn1 = CBAM(512)
        self.self_attn2 = CBAM(512)
        self.cross_attn = CBAMCrossAttention(512)

        # --- low-level은 기존 유지 (128) ---
        self.low_self_attn1 = CBAM(128)
        self.low_self_attn2 = CBAM(128)
        self.low_cross_attn = CBAMCrossAttention(128)

        # --- classifier도 바꿔야 함 ---
        self.classifier = DeepLabHeadV3Plus(
            in_channels=512,            # ASPP 입력
            low_level_channels=128,     # 그대로 또는 64로 축소 가능
            num_classes=num_classes,
            aspp_dilate=aspp_dilate)
    def forward(self, x_orig, x_dog):
        input_shape = x_orig.shape[2:]  # (H, W)

        # --- Feature Extraction ---
        feat1 = self.backbone1(x_orig)
        feat2 = self.backbone2(x_dog)

        # --- High-Level Attention Fusion ---
        f1 = self.self_attn1(self.reduce1(feat1['out']))
        f2 = self.self_attn2(self.reduce2(feat2['out']))
        fused = self.cross_attn(f1, f2)

        # --- Low-Level Attention Fusion ---
        l1 = self.low_self_attn1(feat1['low_level'])
        l2 = self.low_self_attn2(feat2['low_level'])
        low_level = self.low_cross_attn(l1, l2)  # shape: [B, 128, H/4, W/4]

        return self.classifier({'out': fused, 'low_level': low_level}, input_shape=input_shape)
