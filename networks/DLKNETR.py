import torch
from torch import nn

from networks.DLKNETR_blocks import DLKNETR_model


class DLKNETR(nn.Module):
    def __init__(self,
                 input_channels,
                 n_stages,
                 features_per_stage,
                 conv_op,
                 num_classes,
                 deep_supervision
                 ):

        super().__init__()

        self.decoder = DLKNETR_model(
            in_channels=input_channels,
            out_channels=num_classes,
            deep_supervision=deep_supervision
        )

    def forward(self, x):
        return self.decoder(x)
