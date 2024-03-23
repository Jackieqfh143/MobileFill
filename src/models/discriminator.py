import numpy as np
import torch.nn as nn
from src.modules.attention import SEAttention

class SeperableConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1, groups=1,bias = True,padding_mode='reflect'):
        super(SeperableConv, self).__init__()
        self.depthConv = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups = in_channels,bias=bias, padding_mode=padding_mode)
        self.pointConv = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,groups=1,bias=bias, padding_mode=padding_mode)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self,x):
        x = self.depthConv(x)
        x = self.pointConv(x)

        return x

class MultidilatedConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size = 3, dilation_num=3, comb_mode='sum', equal_dim=True,
                 padding=1, min_dilation=1, use_depthwise=False, **kwargs):
        super().__init__()
        convs = []
        self.equal_dim = equal_dim
        assert comb_mode in ('cat_out', 'sum', 'cat_in', 'cat_both'), comb_mode
        if comb_mode in ('cat_out', 'cat_both'):
            self.cat_out = True
            if equal_dim:
                assert out_dim % dilation_num == 0
                out_dims = [out_dim // dilation_num] * dilation_num
                self.index = sum([[i + j * (out_dims[0]) for j in range(dilation_num)] for i in range(out_dims[0])], [])
            else:
                out_dims = [out_dim // 2 ** (i + 1) for i in range(dilation_num - 1)]
                out_dims.append(out_dim - sum(out_dims))
                index = []
                starts = [0] + out_dims[:-1]
                lengths = [out_dims[i] // out_dims[-1] for i in range(dilation_num)]
                for i in range(out_dims[-1]):
                    for j in range(dilation_num):
                        index += list(range(starts[j], starts[j] + lengths[j]))
                        starts[j] += lengths[j]
                self.index = index
                assert(len(index) == out_dim)
            self.out_dims = out_dims
        else:
            self.cat_out = False
            self.out_dims = [out_dim] * dilation_num

        if comb_mode in ('cat_in', 'cat_both'):
            if equal_dim:
                assert in_dim % dilation_num == 0
                in_dims = [in_dim // dilation_num] * dilation_num
            else:
                in_dims = [in_dim // 2 ** (i + 1) for i in range(dilation_num - 1)]
                in_dims.append(in_dim - sum(in_dims))
            self.in_dims = in_dims
            self.cat_in = True
        else:
            self.cat_in = False
            self.in_dims = [in_dim] * dilation_num

        conv_type = SeperableConv if use_depthwise else nn.Conv2d
        dilation = min_dilation
        for i in range(dilation_num):
            if isinstance(padding, int):
                cur_padding = padding * dilation
            else:
                cur_padding = padding[i]
            convs.append(conv_type(
                self.in_dims[i], self.out_dims[i], kernel_size, padding=cur_padding, dilation=dilation, **kwargs
            ))
            dilation *= 2
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        outs = []
        if self.cat_in:
            if self.equal_dim:
                x = x.chunk(len(self.convs), dim=1)
            else:
                new_x = []
                start = 0
                for dim in self.in_dims:
                    new_x.append(x[:, start:start+dim])
                    start += dim
                x = new_x
        for i, conv in enumerate(self.convs):
            if self.cat_in:
                input = x[i]
            else:
                input = x
            outs.append(conv(input))
        if self.cat_out:
            out = torch.cat(outs, dim=1)[:, self.index]
        else:
            out = sum(outs)
        return out

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,with_attn=False,nc_reduce=8):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//nc_reduce , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//nc_reduce , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,height,width  = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1) # B X (*W*H) X (C)
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,height,width)

        out = self.gamma*out + x

        if self.with_attn:
            return out,attention
        else:
            return out

class AttentionModule(nn.Module):
    def __init__(self,in_nc,use_spatial_att=True,reduction=8):
        super(AttentionModule, self).__init__()
        self.channel_attention = SEAttention(channel=in_nc,reduction=reduction)
        if use_spatial_att:
            self.spatial_attention = Self_Attn(in_dim=in_nc)
        else:
            self.spatial_attention = nn.Identity()

    def forward(self,x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x

#input is the concatenation of x and mask
class MultidilatedNLayerDiscriminatorWithAtt(nn.Module):
    def __init__(self, input_nc = 4, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, size = 256):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                MultidilatedConv(nf_prev, nf, kernel_size=kw, stride=2, dilation_num=4,padding=[2, 3, 6, 12]),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf

        # cur_model = []
        # cur_model += [
        #     AttentionModule(in_nc=nf_prev)
        # ]
        #
        # sequence.append(cur_model)

        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        cur_model = []
        cur_model += [
            AttentionModule(in_nc=nf)
        ]

        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 3):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, images_in, masks_in):
        x = torch.cat([images_in, masks_in], dim=1)
        act = self.get_all_activations(x)
        return act[-1], act[:-1]

from src.modules.legacy import *

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class StyleGAN_Discriminator(nn.Module):
    def __init__(self, size = 256, input_nc=4, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], activate=False):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(input_nc, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )
        self.activate = activate

    def forward(self, images_in, masks_in):
        x = torch.cat([images_in, masks_in], dim=1)
        out = self.convs(x)
        out = self.minibatch_discrimination(out, self.stddev_group, self.stddev_feat)
        out = self.final_conv(out)
        out = out.view(out.size(0), -1)
        out = self.final_linear(out)
        if self.activate:
            out = out.sigmoid()
        # return {"out": out}

        return out,None

    @staticmethod
    def minibatch_discrimination(x, stddev_group, stddev_feat):
        out = x
        batch, channel, height, width = out.shape
        group = min(batch, stddev_group)
        stddev = out.view(group, -1, stddev_feat, channel // stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        return out


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,size = 256):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x, mask):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]

if __name__ == '__main__':
    from complexity import *



