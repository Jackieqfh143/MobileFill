from .styled_conv2d import *
from .multichannel_image import *
from .modulated_conv2d import *
from .idwt_upsample import *
from .ffc import *
from src.modules.attention import AttentionLayer
from src.modules.eesp import EESPBlock
import importlib



class MobileSynthesisBlock(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size=3,
            conv_module=ModulatedDWConv2d_v2,
            att_type = "eca",
            att_args = {},
            use_eesp_block = True,
            eesp_block_num = 3,
            eesp_block_args = {},
    ):
        super().__init__()
        self.use_eesp_block = use_eesp_block
        self.up = IDWTUpsaplme(channels_in, style_dim)
        self.conv1 = StyledConv2d(
            channels_in // 4,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.conv2 = StyledConv2d(
            channels_out,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )

        if self.use_eesp_block:
            eesp_res_blocks = []
            for i in range(eesp_block_num):
                eesp_res_blocks.append(EESPBlock(channels_out,channels_out, **eesp_block_args))
            self.eesp_res_block = nn.Sequential(*eesp_res_blocks)

            if att_type != None:
                self.att = AttentionLayer(kind=att_type, **att_args)

        self.to_img = MultichannelIamge(
            channels_in=channels_out,
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )

    def forward(self, hidden, style, en_feat,noise=[None, None]):
        hidden = self.up(hidden, style if style.ndim == 2 else style[:, 0, :])
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=noise[0])
        hidden = hidden + en_feat
        if self.use_eesp_block:
            hidden = self.eesp_res_block(hidden)

            if hasattr(self, "att"):
                hidden = self.att(hidden)

        hidden = self.conv2(hidden, style if style.ndim == 2 else style[:, 1, :], noise=noise[1])

        img = self.to_img(hidden, style if style.ndim == 2 else style[:, 2, :])
        return hidden, img

    def wsize(self):
        return 3

