import random
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from src.modules.ffc import MyConv2d
from src.modules.legacy import EqualLinear
from src.utils.diffaug import rand_cutout
from src.modules.cnn_utils import CBR
from src.modules.eesp import EESPBlock,DownSampler
from src.modules.mobilevit import MobileViT
# from src.utils.complexity import flop_counter, print_network_params


class ToStyle(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer = 'bn', activation='leaky'):
        super().__init__()
        self.conv = nn.Sequential(
            MyConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,stride=2,padding=1, norm_layer=norm_layer,activation=activation),
            MyConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,stride=2,padding=1, norm_layer=norm_layer, activation=activation),
            MyConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2,padding=1, norm_layer=norm_layer, activation=activation),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = EqualLinear(in_channels,out_channels,activation='fused_lrelu')

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))

        return x

class Encoder(nn.Module):

    def __init__(self, channels = [], input_nc=4, input_size = 256, latent_nc = 512, down_block_num = 4,
                 to_style = False, use_mobile_vit = True,
                 eesp_block_num = 3, style_merge_type = "add",
                 eesp_block_args = {}):
        '''
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()
        self.to_style = to_style
        self.use_mobile_vit = use_mobile_vit
        self.style_merge_type = style_merge_type

        self.down_block_num = down_block_num
        prev_nc_idx = 0
        for i in range(down_block_num):
            end_nc_idx = min(len(channels) - 1, prev_nc_idx + 1)
            if i == 0:
                down_block = CBR(input_nc, channels[prev_nc_idx], 5, 2)
                end_nc_idx = prev_nc_idx
            else:
                down_block = DownSampler(channels[prev_nc_idx], channels[end_nc_idx])

            setattr(self, f"layer{i}_0", down_block)
            res_blocks = []
            for j in range(eesp_block_num):
                res_blocks.append(EESPBlock(channels[end_nc_idx],channels[end_nc_idx], **eesp_block_args))

            prev_nc_idx = end_nc_idx
            setattr(self, f"layer{i}_1", nn.Sequential(*res_blocks))

        self.img_size = input_size // 2 ** down_block_num

        if self.use_mobile_vit:
            self.transformer = MobileViT(in_channels=channels[-1], image_size=(self.img_size, self.img_size))

        if to_style:
            self.to_style_block = ToStyle(channels[-1], latent_nc)

        if self.style_merge_type != None:
            self.to_square = EqualLinear(latent_nc, self.img_size * self.img_size, activation="fused_lrelu")
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input, ws):
        feats = []
        out = input
        for i in range(self.down_block_num):
            temp = getattr(self, f"layer{i}_0")(out)
            out = getattr(self, f"layer{i}_1")(temp)
            feats.append(out)

        if self.style_merge_type != None:
            style_m = self.to_square(ws).view(-1, self.img_size, self.img_size).unsqueeze(1)
            style_m = F.interpolate(style_m, size=out.size()[-2:], mode='bilinear', align_corners=False)

            if self.style_merge_type == "add":
                out = out + style_m
            elif self.style_merge_type == "dropout":
                mul_map = torch.ones_like(out)
                mul_map = F.dropout(mul_map, training=True)
                out = out * mul_map + style_m * (1 - mul_map)
            elif self.style_merge_type == "rand_mask":
                mask = rand_cutout(out, random.random())
                out = out * mask + style_m * (1 - mask)

        if self.use_mobile_vit:
            out = self.transformer(out)

        feats[-1] = out
        feats = feats[::-1]


        if self.to_style:
            gs = self.to_style_block(feats[0])
            return ws, gs, feats

        # ws = ws.unsqueeze(1).repeat(1, 11, 1)
        return ws, None, feats


if __name__ == '__main__':
    channels = {
        4: 256,
        8: 512,
        16: 512,
        32: 512,
        64: 256,
        128: 128,
        256: 64,
        512: 64,
    }
    input = torch.Tensor(1, 4, 256, 256)
    ws = torch.randn(1,512)
    style_square = torch.randn(1,512,16,16)
    # en_channels = [v for k,v in channels.items() if k < input.size(-1)][::-1]
    en_channels = [128,256,512,512,512]
    model = Encoder(channels = en_channels, style_merge_type="rand_mask", down_block_num=4)
    # print_network_params(model,"model")
    # flop_counter(model,(input,ws))
    # out = model(input,ws)
    # print()
