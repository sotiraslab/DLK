import torch
import torch.nn as nn

from networks.DNet_blocks import Encoder, Decoder

class DNet(nn.Module):

    def __init__(
        self,
        input_channels,
        n_stages,
        features_per_stage,
        conv_op,
        num_classes,
        deep_supervision,
        depths=[2, 2, 2, 2, 2],
        feat_size=[48, 96, 192, 384, 768],
        drop_path_rate=0            
    ) -> None:
        super().__init__()     

        self.in_channels = input_channels
        self.out_channels = num_classes
        self.deep_supervision=deep_supervision

        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size

        self.encoder = Encoder(
            in_chans=self.in_channels,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate
        )

        self.decoder = Decoder(
            out_channels=self.out_channels,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            deep_supervision=self.deep_supervision
        )

    def forward(self, x):
        skips, out = self.encoder(x)
        return self.decoder(skips, out)