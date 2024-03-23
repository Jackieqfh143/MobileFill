import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        # create conv
        self.weight = nn.Parameter(
            torch.randn(channels_out, channels_in, kernel_size, kernel_size)
        )
        # create modulation network
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        # create demodulation parameters
        self.demodulate = demodulate
        if self.demodulate:
            self.register_buffer("style_inv", torch.randn(1, 1, channels_in, 1, 1))
        # some service staff
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight, padding=self.padding)
        if self.demodulate:
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style):
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        modulation = self.scale * style
        return modulation

    def get_demodulation(self, style):
        w = self.weight.unsqueeze(0)
        norm = torch.rsqrt((self.scale * self.style_inv * w).pow(2).sum([2, 3, 4]) + 1e-8)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)

class ModulatedDWConv2d(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        # create conv
        #(128, 1, 3, 3)
        self.weight_dw = nn.Parameter(
            torch.randn(channels_in, 1, kernel_size, kernel_size)
        )

        #(512, 128, 1, 1)
        self.weight_permute = nn.Parameter(
            torch.randn(channels_out, channels_in, 1, 1)
        )
        # create modulation network
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        # create demodulation parameters
        self.demodulate = demodulate
        if self.demodulate:
            #(1, 1, 128, 1, 1)
            self.register_buffer("style_inv", torch.randn(1, 1, channels_in, 1, 1))
        # some service staff
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight_dw, padding=self.padding, groups=x.size(1))
        x = F.conv2d(x, self.weight_permute)
        if self.demodulate:
            #(1, 512, 1, 1)
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style):
        #(2,128,1,1)
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        modulation = self.scale * style
        return modulation

    def get_demodulation(self, style):
        #(1, 512, 128, 3, 3)
        w = (self.weight_dw.transpose(0, 1) * self.weight_permute).unsqueeze(0)
        norm = torch.rsqrt((self.scale * self.style_inv * w).pow(2).sum([2, 3, 4]) + 1e-8)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)

class ModulatedDWConv2d_v2(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        # create conv
        self.weight_dw = nn.Parameter(
            torch.randn(channels_in, 1, kernel_size, kernel_size)
        )
        self.weight_permute = nn.Parameter(
            torch.randn(channels_out, channels_in, 1, 1)
        )
        # create modulation network
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        # create demodulation parameters
        self.demodulate = demodulate
        # if self.demodulate:
        #     self.register_buffer("style_inv", torch.randn(1, 1, channels_in, 1, 1))
        # some service staff
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        batch, in_channel, height, width = x.shape
        modulation, style = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight_dw, padding=self.padding, groups=x.size(1))
        x = F.conv2d(x, self.weight_permute)

        if self.demodulate:
            demodulation = self.get_demodulation(style, batch, in_channel)
            x = demodulation * x

        return x

    def get_modulation(self, style):
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        modulation = self.scale * style
        return modulation, style

    def get_demodulation(self, style, batch, channel):
        style = style.view(batch, 1, channel, 1, 1)
        w = (self.weight_dw.transpose(0, 1) * self.weight_permute).unsqueeze(0)
        norm = torch.rsqrt((style * w).pow(2).sum([2, 3, 4]) + 1e-8)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)



