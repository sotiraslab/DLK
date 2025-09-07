from typing import Sequence, Tuple, Union
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.layers import DropPath, trunc_normal_

class DLKNETR_model(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            deep_supervision,
            depths=[2, 2, 2, 2, 2],
            feat_size=[48, 96, 192, 384, 768],
            spatial_dims = 3,
            norm_name = "instance",
            drop_path_rate=0 
        ):
        super(DLKNETR_model, self).__init__()

        self.deep_supervision = deep_supervision

        self.DLKEncoder = DLKEncoder(
            in_chans = in_channels,
            depths = depths,
            dims = feat_size,
            drop_path_rate = drop_path_rate
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[4],
            out_channels=feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feat_size[0], out_channels=out_channels
        )

        if self.deep_supervision:
            self.out_4 = nn.Conv3d(feat_size[4], out_channels, 1)
            self.out_3 = nn.Conv3d(feat_size[3], out_channels, 1)
            self.out_2 = nn.Conv3d(feat_size[2], out_channels, 1)
            self.out_1 = nn.Conv3d(feat_size[1], out_channels, 1)
            self.out_0 = nn.Conv3d(feat_size[0], out_channels, 1)

    def forward(self, x):
        hidden_states_out = self.DLKEncoder(x)
        enc0 = self.encoder1(x)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        enc4 = self.encoder5(hidden_states_out[3])

        dec3 = self.decoder5(hidden_states_out[4], enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        if self.deep_supervision:
            return tuple([self.out(out), self.out_0(dec0), self.out_1(dec1), self.out_2(dec2), self.out_3(dec3)])
        else:
            return self.out(out)

class UnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.
        """
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = out[:,:, :skip.size()[2], :skip.size()[3], :skip.size()[4]]
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out

class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        drop = 0.
        self.fc1 = nn.Conv3d(dim, dim * 4, 1)
        self.dwconv = nn.Conv3d(dim * 4, dim * 4, 3, 1, 1, bias=True, groups=dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(dim * 4, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DLK(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.channel_proj = nn.Conv3d(dim, dim//2, kernel_size=1, bias=False)
        self.att_conv1 = nn.Conv3d(dim//2, dim//2, kernel_size=5, stride=1, padding=2, groups=dim//2)
        self.att_conv2 = nn.Conv3d(dim//2, dim//2, kernel_size=7, stride=1, padding=9, groups=dim//2, dilation=3)

        self.spatial_se = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        proj_x = self.channel_proj(x)

        out1 = self.att_conv1(proj_x)
        out2 = self.att_conv2(out1)

        out = torch.cat([out1, out2], dim=1)
        
        avg_att = torch.mean(out, dim=1, keepdim=True)
        max_att,_ = torch.max(out, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        out = out * att[:,0,:,:,:].unsqueeze(1) + out * att[:,1,:,:,:].unsqueeze(1)

        att = self.conv_atten(self.avg_pool(out))
        out = att * out

        out = out + x
        return out

class DLKModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_1 = nn.Conv3d(dim, dim, 1)
        self.act = nn.GELU()
        self.spatial_gating_unit = DLK(dim)
        self.proj_2 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x

class DLKBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.norm_layer = nn.LayerNorm(dim, eps=1e-6)
        self.attn = DLKModule(dim)
        self.mlp = Mlp(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        layer_scale_init_value = 1e-2   
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()
        x = channel_to_last(x)
        x = self.norm_layer(x)
        x = channel_to_first(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x)

        shortcut = x.clone()
        x = channel_to_last(x)
        x = self.norm_layer(x)
        x = channel_to_first(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x)
        
        return x

class DLKEncoder(nn.Module):
    def __init__(
            self,
            in_chans,
            depths,
            dims,
            drop_path_rate: float = 0.0
    ) -> None:
        super(DLKEncoder, self).__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3)
        self.downsample_layers.append(stem)
        for i in range(4):
            downsample_layer = nn.Conv3d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1)
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[DLKBlock(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm_layers = nn.ModuleList()
        for i in range(5):
            norm_layer = nn.LayerNorm(dims[i], eps=1e-6)
            self.norm_layers.append(norm_layer)

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = channel_to_last(x)
            x = self.norm_layers[i](x)
            x = channel_to_first(x)
            x = self.stages[i](x)

            x_out = channel_to_last(x)
            x_out = self.norm_layers[i](x_out)
            x_out = channel_to_first(x_out)
            outs.append(x_out)

        x = self.downsample_layers[-1](x)
        x_out = channel_to_last(x)
        x_out = self.norm_layers[-1](x_out)
        x_out = channel_to_first(x_out)
        outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class Decoder(nn.Module):
    def __init__(self, deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision

    def forward(self):
        None

def channel_to_last(x):
    """
    Args:
        x: (B, C, H, W, D)

    Returns:
        x: (B, H, W, D, C)
    """
    return x.permute(0, 2, 3, 4, 1)


def channel_to_first(x):
    """
    Args:
        x: (B, H, W, D, C)

    Returns:
        x: (B, C, H, W, D)
    """
    return x.permute(0, 4, 1, 2, 3)


if __name__ == '__main__':
    data = torch.randn((1, 1, 96, 96, 96), dtype=torch.float32)
    model = DLKNETR_model(
        in_channels=1,
        out_channels=16,
        deep_supervision=True
    )
    result = model(data)
    import thop
    flops, params = thop.profile(model, inputs=(data,))
    print("flops = ", flops / 1024.0 / 1024.0 / 1024.0, 'G')
    print("params = ", params / 1024.0 / 1024.0, 'M')
