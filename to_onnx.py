import traceback
import torch
import torch.nn as nn
import torchvision
from src.models.mobileFill import MobileFill


def apply_trace_model_mode(mode=False):
    def _apply_trace_model_mode(m):
        if hasattr(m, 'trace_model'):
            m.trace_model = mode
    return _apply_trace_model_mode


class Wrapper(nn.Module):
    def __init__(
            self,
            synthesis_network,
            style_tmp,
            en_feats_tmp,
    ):
        super().__init__()
        self.m = synthesis_network
        self.noise = self.m(style_tmp,en_feats_tmp)["noise"]

    def forward(self, style,en_feats):
        return self.m(style,en_feats,noise=None)["img"]



if __name__ == '__main__':
    import yaml
    from src.utils.complexity import *
    from collections import OrderedDict
    model_path = './checkpoints/places/G-step=720000_lr=0.0001_ema_loss=0.4854.pth'
    model_config_path = "configs/mobilefill.yaml"
    with open(model_config_path, "r") as f:
        opts = yaml.load(f, yaml.FullLoader)

    device = "cpu"
    opts["device"] = device
    input_size = opts["target_size"]
    x = torch.randn(1, 4, input_size, input_size).to(device)
    ws = torch.randn(1, 512).to(device)

    model = MobileFill(**opts)

    net_state_dict = model.state_dict()
    state_dict = torch.load(model_path, map_location='cpu')["ema_G"]
    new_state_dict = {k: v for k, v in state_dict.items() if k in net_state_dict}
    model.load_state_dict(OrderedDict(new_state_dict), strict=False)
    model.eval().requires_grad_(False).to(device)
    ws_, gs, en_feats = model.encoder(x, ws)
    # out = model(x)
    # print_network_params(model, "MobileFill")
    # print_network_params(model.encoder, "MobileFill.encoder")
    # print_network_params(model.generator, "MobileFill.generator")
    # print_network_params(model.mapping_net, "MobileFill.mapping_net")
    #
    # flops = 0.0
    # flops += flop_counter(model.encoder, (x, ws))
    # flops += flop_counter(model.mapping_net, ws)
    # ws_ = ws.unsqueeze(1).repeat(1, model.latent_num, 1)
    # if opts["encoder"]["to_style"]:
    #     gs = gs.unsqueeze(1).repeat(1, model.latent_num, 1)
    #     ws = torch.cat((ws, gs), dim=-1)
    # flops += flop_counter(model.generator, (ws, en_feats))
    # print(f"Total FLOPs: {flops:.5f} G")
    model.encoder.apply(apply_trace_model_mode(True))
    torch.onnx.export(
        model.encoder,
        (x,ws),
        "./encoder.onnx",
        input_names=['input','in_ws'],
        output_names=["out_ws", "gs", "en_feats"],
        verbose=True,
        dynamic_axes={
            "input":[0],
            "in_ws":[0],
            "out_ws":[0],
            "gs": [0],
            "en_feats": [0],
        }
    )

    noise = torch.randn(1, 512).to(device)

    model.mapping_net.apply(apply_trace_model_mode(True))
    torch.onnx.export(
        model.mapping_net,
        (noise, ),
        "./mapping.onnx",
        input_names=['noise'],
        output_names=['style'],
        verbose=True,
        dynamic_axes={
            'noise':[0],
            'style':[0]
        }
    )

    model.generator.apply(apply_trace_model_mode(True))
    

    en_feats_name = [f"en_feats{i}" for i in range(len(en_feats))]

    torch.onnx.export(
        Wrapper(model.generator, ws_, en_feats),
        (ws_, en_feats,),
        "./generator.onnx",
        input_names=['style', *en_feats_name],
        output_names=['out'],
        verbose=True,
        dynamic_axes={
            "style": [0],
            'out': [0]
        }
    )











