from src.utils.noise_manager import NoiseManager
from src.modules.mobile_synthesis_block import MobileSynthesisBlock
from src.modules.idwt_upsample import DWTInverse
from src.modules.ffc import actLayer
from torchscan import summary
from src.modules.legacy import *
import importlib


def find_conv_using_name(package_name, model_name):
    modellib = importlib.import_module(package_name)
    model = None
    target_model_name = model_name
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls
    if model !=None:
        return model
    else:
        raise Exception(f'No convolution module named {model_name}')


class MobileSynthesisNetwork(nn.Module):
    def __init__(
            self,
            channels = [512, 512, 256, 128],
            style_dim = 512,
            kernel_size = 3,
            device = 'cuda',
            out_act = 'tanh',
            conv_type = 'ModulatedDWConv2d_v2',
            trace_model = False,
            block_args = {},
    ):
        super().__init__()
        self.style_dim = style_dim
        self.device = device

        self.layers = nn.ModuleList()
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock(
                    channels_in,
                    channels_out,
                    style_dim,
                    kernel_size,
                    conv_module = find_conv_using_name("src.modules.modulated_conv2d",
                                                       model_name=conv_type),
                    **block_args,
                )
            )
            channels_in = channels_out

        self.trace_model = trace_model
        self.idwt = DWTInverse(mode="zero", wave="db1",trace_model=self.trace_model)
        self.out_act = actLayer(kind=out_act)

    def forward(self,style, en_feats,noise=None):
        out = {"noise": [], "freq": [], "img": None,
               "en_feats": en_feats}
        noise = NoiseManager(noise, self.device, self.trace_model)

        hidden = en_feats[0]

        for i, m in enumerate(self.layers):
            start_style_idx = m.wsize()*i + 1
            end_style_idx = m.wsize()*i + m.wsize() + 1
            out["noise"].append(noise(2 * hidden.size(-1), 2))
            hidden, freq = m(hidden,style if len(style.shape)==2 else style[:, start_style_idx:end_style_idx, :],
                             en_feats[i+1],
                             noise=out["noise"][-1])

            out["freq"].append(freq)

        out["img"] = self.dwt_to_img(out["freq"][-1])

        out["img"] = self.out_act(out["img"])

        return out

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))

    def wsize(self):
        return len(self.layers) * self.layers[0].wsize() + 2


if __name__ == '__main__':
    from src.utils.complexity import *
    ws = torch.randn(1,512).cuda()
    en_feats = [
        torch.randn(1,512,16,16).cuda(),
        torch.randn(1,512,32,32).cuda(),
        torch.randn(1,256,64,64).cuda(),
        torch.randn(1,128,128,128).cuda()
    ]
    model = MobileSynthesisNetwork(style_dim=512).cuda()
    print_network_params(model,"model")
    flop_counter(model, [ws, en_feats])
    out = model(ws, en_feats)
