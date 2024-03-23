#-*- coding: utf-8 -*-
from src.models.generator import MobileSynthesisNetwork
from src.models.mapping_network import MappingNetwork
from src.models.encoder import Encoder
import torch.nn as nn
import torch

class MobileFill(nn.Module):
    def __init__(self, device= 'cuda',target_size=256, mlp_layers=8, latent_nc=512, **kwargs):
        super(MobileFill, self).__init__()
        self.target_size = target_size
        self.device = device
        self.encoder = Encoder(**kwargs["encoder"]).to(device)
        self.mapping_net = MappingNetwork(style_dim=latent_nc,n_layers=mlp_layers).to(device)
        self.generator = MobileSynthesisNetwork(**kwargs["generator"], device = device).to(device)
        self.latent_nc = latent_nc
        self.latent_num = self.generator.wsize()

    def make_style(self,batch_size):
        z = torch.randn(batch_size, self.latent_nc).to(self.device)
        ws = self.mapping_net(z)
        return ws

    def forward(self,x):
        ws = self.make_style(batch_size=x.size(0))
        ws, gs, en_feats = self.encoder(x,ws)
        ws = ws.unsqueeze(1).repeat(1, self.latent_num, 1)
        if gs != None:
            gs = gs.unsqueeze(1).repeat(1, self.latent_num, 1)
            ws = torch.cat((ws,gs), dim = -1)

        gen_out = self.generator(ws, en_feats)
        return gen_out,ws

if __name__ == '__main__':
    import yaml
    from src.utils.complexity import get_flops

    model_config_path = "../../configs/ablations/wo_style.yaml"
    with open(model_config_path, "r") as f:
        opts = yaml.load(f, yaml.FullLoader)

    device = opts["device"]
    input_size = opts["target_size"]
    x = torch.randn(1,4,input_size,input_size).to(device)
    ws = torch.randn(1,512).to(device)

    model = MobileFill(**opts)

    get_flops(model, [x])
    # ws,gs,en_feats = model.encoder(x,ws)
    # out = model(x)
    # print_network_params(model,"MobileFill")
    # print_network_params(model.encoder,"MobileFill.encoder")
    # print_network_params(model.generator,"MobileFill.generator")
    # print_network_params(model.mapping_net, "MobileFill.mapping_net")
    #
    # flops = 0.0
    # flops += flop_counter(model.encoder,(x,ws))
    # flops += flop_counter(model.mapping_net, ws)
    # ws = ws.unsqueeze(1).repeat(1, model.latent_num, 1)
    # if opts["encoder"]["to_style"]:
    #     gs = gs.unsqueeze(1).repeat(1, model.latent_num, 1)
    #     ws = torch.cat((ws, gs), dim = -1)
    # flops += flop_counter(model.generator,(ws, en_feats))
    # print(f"Total FLOPs: {flops:.5f} G")


