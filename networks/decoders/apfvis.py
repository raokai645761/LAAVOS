import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.layers.basic import ConvGN

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F



def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    if not pad:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        pad = pad
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, filter_in, filter_out):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_in, filter_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_out, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filter_out, filter_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_out, momentum=0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


# class Upsample(nn.Module):
#     def __init__(self, in_channels, out_channels, scale_factor=2):
#         super(Upsample, self).__init__()

#         self.upsample = nn.Sequential(
#             BasicConv(in_channels, out_channels, 1),
#             nn.Upsample(scale_factor=scale_factor, mode='bilinear')
#         )

#         # carafe
#         # from mmcv.ops import CARAFEPack
#         # self.upsample = nn.Sequential(
#         #     BasicConv(in_channels, out_channels, 1),
#         #     CARAFEPack(out_channels, scale_factor=scale_factor)
#         # )

#     def forward(self, x):
#         x = self.upsample(x)

#         return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.conv = BasicConv(in_channels, out_channels, 1)
        self.scale_factor = scale_factor

    def forward(self, x, x1):
        x = self.conv(x)
        x = F.interpolate(x, size=x1.size()[-2:], mode='bilinear', align_corners=False)

        return x
    
# class Downsample_x2(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Downsample_x2, self).__init__()

#         self.downsample = nn.Sequential(
#             BasicConv(in_channels, out_channels, 2, 2, 0)
#         )

#     def forward(self, x, ):
#         x = self.downsample(x)

#         return x

class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()

        self.conv = BasicConv(in_channels, out_channels, 1)

    def forward(self, x, x1):
        x = self.conv(x)
        x = F.interpolate(x, size=x1.size()[-2:], mode='bilinear', align_corners=False)

        return x
    
# class Downsample_x4(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Downsample_x4, self).__init__()

#         self.downsample = nn.Sequential(
#             BasicConv(in_channels, out_channels, 4, 4, 0)
#         )

#     def forward(self, x, ):
#         x = self.downsample(x)

#         return x
    
class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x4, self).__init__()

        self.conv = BasicConv(in_channels, out_channels, 1)

    def forward(self, x, x1):
        x = self.conv(x)
        x = F.interpolate(x, size=x1.size()[-2:], mode='bilinear', align_corners=False)
        return x
    

# class Downsample_x8(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Downsample_x8, self).__init__()

#         self.downsample = nn.Sequential(
#             BasicConv(in_channels, out_channels, 8, 8, 0)
#         )

#     def forward(self, x, ):
#         x = self.downsample(x)

#         return x

    
class Downsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x8, self).__init__()

        self.conv = BasicConv(in_channels, out_channels, 1)

    def forward(self, x, x1):
        x = self.conv(x)
        x = F.interpolate(x, size=x1.size()[-2:], mode='bilinear', align_corners=False)

        return x
    
class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
#         print('111',level_1_weight_v.shape)
#         print('222',level_2_weight_v.shape)
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return out


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2, input3):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]

        out = self.conv(fused_out_reduced)

        return out


class BlockBody(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        super(BlockBody, self).__init__()

        self.blocks_scalezero1 = nn.Sequential(
            BasicConv(channels[0], channels[0], 1),
        )
        self.blocks_scaleone1 = nn.Sequential(
            BasicConv(channels[1], channels[1], 1),
        )
        self.blocks_scaletwo1 = nn.Sequential(
            BasicConv(channels[2], channels[2], 1),
        )

        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])

        self.blocks_scalezero2 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone2 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )

        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])


        self.blocks_scalezero4 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone4 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo4 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )

    def forward(self, x):
        x0, x1, x2 = x
#         print('111', x0.shape)
#         print('222', x1.shape)
        x0 = self.blocks_scalezero1(x0)
        x1 = self.blocks_scaleone1(x1)
        x2 = self.blocks_scaletwo1(x2)
#         print('111', x0.shape)
#         print('222', x1.shape)
        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1, x0))
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0, x1), x1)

        x0 = self.blocks_scalezero2(scalezero)
        x1 = self.blocks_scaleone2(scaleone)

        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1, x0), self.upsample_scaletwo2_4(x2, x0))
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0, x1), x1, self.upsample_scaletwo2_2(x2, x1))
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0, x2), self.downsample_scaleone2_2(x1, x2), x2)

        scalezero = self.blocks_scalezero4(scalezero)
        scaleone = self.blocks_scaleone4(scaleone)
        scaletwo = self.blocks_scaletwo4(scaletwo)

        return scalezero, scaleone, scaletwo

class AFPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256):
        super(AFPN, self).__init__()

        self.fp16_enabled = False

        self.conv0 = BasicConv(in_channels[0], in_channels[0] // 8, 1)
        self.conv1 = BasicConv(in_channels[1], in_channels[1] // 8, 1)
        self.conv2 = BasicConv(in_channels[2], in_channels[2] // 8, 1)

        self.body = nn.Sequential(
            BlockBody([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8])
        )

        self.conv00 = BasicConv(in_channels[0] // 8, out_channels, 1)
        self.conv11 = BasicConv(in_channels[1] // 8, 512, 1)
        self.conv22 = BasicConv(in_channels[2] // 8, 1024, 1)


        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x0, x1, x2):

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        out0, out1, out2 = self.body([x0, x1, x2])

        out0 = self.conv00(out0)
        out1 = self.conv11(out1)
        out2 = self.conv22(out2)

        return out0, out1, out2

class FPNSegmentationHead(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 decode_intermediate_input=True,
                 hidden_dim=256,
                 shortcut_dims=[24, 32, 96, 1280],
                 align_corners=True):
        super().__init__()
        # "MODEL_ENCODER_DIM": [
        #     256,
        #     512,
        #     1024,
        #     1024
        # ]
        self.align_corners = align_corners

        self.decode_intermediate_input = decode_intermediate_input

        self.conv_in = ConvGN(in_dim, hidden_dim, 1)    # 512  >>> 256

        self.conv_16x = ConvGN(hidden_dim, hidden_dim, 3)  # 256 >>> 256
        self.conv_8x = ConvGN(hidden_dim, hidden_dim // 2, 3)   # 256 >>> 128
        self.conv_4x = ConvGN(hidden_dim // 2, hidden_dim // 2, 3)   # 128 >>> 128

        self.adapter_16x = nn.Conv2d(shortcut_dims[-2], hidden_dim, 1)   #  1024 >>> 256
        self.adapter_8x = nn.Conv2d(shortcut_dims[-3], hidden_dim, 1)    #  512  >>> 256
        self.adapter_4x = nn.Conv2d(shortcut_dims[-4], hidden_dim // 2, 1) # 256 >>> 128

        self.conv_out = nn.Conv2d(hidden_dim // 2, out_dim, 1)   # 128  >>> 11

        self.afpn = AFPN()

        self._init_weight()

    def forward(self, inputs, shortcuts):

        if self.decode_intermediate_input:
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs[-1]

#         print('11111', shortcuts[-2].shape)
#         print('22222', shortcuts[-3].shape)
#         print('33333', shortcuts[-4].shape)
#         print('vv', shortcuts[-1].shape)
        x1 = shortcuts[-4]
        x2 = shortcuts[-3]
        x3 = shortcuts[-2]
        

        out1, out2, out3 = self.afpn(x1, x2, x3)

#         print('v66v', out1.shape)
#         print('v77v', out2.shape)
#         print('v88v', out3.shape)
        x = F.relu_(self.conv_in(x))
        x = F.relu_(self.conv_16x(self.adapter_16x(out3) + x))

        x = F.interpolate(x,
                          size=shortcuts[-3].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        x = F.relu_(self.conv_8x(self.adapter_8x(out2) + x))

        x = F.interpolate(x,
                          size=shortcuts[-4].size()[-2:],
                          mode="bilinear",
                          align_corners=self.align_corners)
        x = F.relu_(self.conv_4x(self.adapter_4x(out1) + x))

        x = self.conv_out(x)

        return x

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
