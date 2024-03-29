import torch
import torch.nn as nn

from timm.models.layers import DropPath

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
        self.att_conv1 = nn.Conv3d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.att_conv2 = nn.Conv3d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)

        self.spatial_se = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):   
        att1 = self.att_conv1(x)
        att2 = self.att_conv2(att1)

        att = torch.cat([att1, att2], dim=1)
        avg_att = torch.mean(att, dim=1, keepdim=True)
        max_att,_ = torch.max(att, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        output = att1 * att[:,0,:,:,:].unsqueeze(1) + att2 * att[:,1,:,:,:].unsqueeze(1)
        output = output + x
        return output

class DLkModule(nn.Module):
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
        self.attn = DLkModule(dim)
        self.mlp = Mlp(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        layer_scale_init_value = 1e-6         
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

class Encoder(nn.Module):
    def __init__(self, in_chans, depths, dims, drop_path_rate):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2)
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
        for i in range(4):
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
            outs.append(x)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_chans, depths, dims, drop_path_rate):
        super().__init__()
        
        self.upsample_layers = nn.ModuleList()
        for i in range(3):
            upsample_layer = nn.ConvTranspose3d(dims[-i-1], dims[-i-2], kernel_size=2, stride=2)
            self.upsample_layers.append(upsample_layer)

        stem = nn.ConvTranspose3d(dims[0], dims[0], kernel_size=2, stride=2)
        self.upsample_layers.append(stem)

        self.steps = nn.ModuleList()
        for i in range(4):
            step = DFF(dims[-i-1])
            self.steps.append(step)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[DLKBlock(dim=dims[-i-1], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm_layers = nn.ModuleList()
        for i in range(3):
            norm_layer = nn.LayerNorm(dims[-i-2], eps=1e-6)
            self.norm_layers.append(norm_layer)

        norm_layer = nn.LayerNorm(dims[0], eps=1e-6)
        self.norm_layers.append(norm_layer)

    def forward(self, x, skips):
        for i in range(4):
            x = self.steps[i](x, skips[-i-1])
            x = self.stages[i](x)
            x = self.upsample_layers[i](x)
            x = channel_to_last(x)
            x = self.norm_layers[i](x)
            x = channel_to_first(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        
        self.stage = nn.Sequential(
            DLKBlock(dim=out_chans),
            DLKBlock(dim=out_chans)
        )

        self.downsample_layer = nn.Conv3d(in_chans, out_chans, kernel_size=2, stride=2)
        self.upsample_layer = nn.ConvTranspose3d(out_chans, in_chans, kernel_size=2, stride=2)
        self.norm_layer1 = nn.LayerNorm(out_chans, eps=1e-6)
        self.norm_layer2 = nn.LayerNorm(in_chans, eps=1e-6)

    def forward(self, x):
        x = self.downsample_layer(x)
        x = channel_to_last(x)
        x = self.norm_layer1(x)
        x = channel_to_first(x)

        x = self.stage(x)

        x = self.upsample_layer(x)
        x = channel_to_last(x)
        x = self.norm_layer2(x)
        x = channel_to_first(x)
        return x

class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv3d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv3d(dim * 2, dim, kernel_size=1, bias=False)

        self.conv1 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)
        
        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        output = output * att
        return output


class Convblock(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(input_dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        return output


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
