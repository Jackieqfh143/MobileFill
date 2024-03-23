import torch.nn as nn
from src.modules.ffc import SpectralTransform
from src.modules.attention import AttentionLayer


class GlobalUnit(nn.Module):
    def __init__(self, g_unit_type = "attention", **kwargs):
        super(GlobalUnit, self).__init__()
        if g_unit_type == "attention":
            self.g_unit = AttentionLayer(**kwargs)
        else:
            self.g_unit = SpectralTransform(**kwargs)

    def forward(self, x):
        return self.g_unit(x)




