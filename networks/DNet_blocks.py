import torch
import torch.nn as nn

from timm.models.layers import DropPath


class Encoder(nn.Module):
    def __init__(self, in_chans, depths, dims, drop_path_rate):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3)
        self.downsample_layers.append(stem)
        for i in range(4):
            downsample_layer = nn.Conv3d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1)
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(5):
            stage = nn.Sequential(
                *[DLKBlock(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm_layers = nn.ModuleList()
        for i in range(5):
            norm_layer = nn.LayerNorm(dims[i], eps=1e-6)
            self.norm_layers.append(norm_layer)

        self.saliency = SaliencyMLPblock(in_chans, dims[0])

    def forward(self, x):
        outs = []
        out = self.saliency(x)
        for i in range(5):
            x = self.downsample_layers[i](x)
            x = channel_to_last(x)
            x = self.norm_layers[i](x)
            x = channel_to_first(x)
            x = self.stages[i](x)
            outs.append(x)

        return outs, out


class Decoder(nn.Module):
    def __init__(self, out_channels, depths, dims, drop_path_rate, deep_supervision):
        super().__init__()

        self.deep_supervision=deep_supervision

        self.upsample_layers = nn.ModuleList()
        for i in range(4):
            upsample_layer = nn.ConvTranspose3d(dims[-i-1], dims[-i-2], kernel_size=2, stride=2)
            self.upsample_layers.append(upsample_layer)

        self.norm_layers = nn.ModuleList()
        for i in range(4):
            norm_layer = nn.LayerNorm(dims[-i-2], eps=1e-6)
            self.norm_layers.append(norm_layer)

        self.steps = nn.ModuleList()
        for i in range(4):
            step = DFF(dims[-i-2])
            self.steps.append(step)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[DLKBlock(dim=dims[-i-2], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.seg_layers = nn.ModuleList()
        for i in range(4):
            seg_layer = nn.Conv3d(dims[-i-2], out_channels, 1, 1, 0, bias=True)
            self.seg_layers.append(seg_layer)

        self.stem = nn.ConvTranspose3d(dims[0], dims[0], kernel_size=2, stride=2)
        self.final_fusion = DFF(dims[0])
        self.final_layer = Convblock(dims[0], dims[0])
        self.output_layer = nn.Conv3d(dims[0], out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, skips, out):
        seg_outputs = []
        input = skips[-1]
        for i in range(4):
            x = self.upsample_layers[i](input)
            x = channel_to_last(x)
            x = self.norm_layers[i](x)
            x = channel_to_first(x)
            x = self.steps[i](x, skips[-i-2])
            x = self.stages[i](x)
            seg_outputs.append(self.seg_layers[i](x))
            input = x

        x = self.stem(x)
        x = self.final_fusion(x, out)
        x = self.final_layer(x)
        x = self.output_layer(x)

        if self.deep_supervision:
            seg_outputs.append(x)
            seg_outputs = seg_outputs[::-1]
            return seg_outputs
        else:
            return x


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

class ChannelMLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()

        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv3d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SaliencyMLPblock(nn.Module):
    def __init__(self, input_dim, dim, mlp_ratio=4):
        super().__init__()

        self.channel_proj = nn.Conv3d(input_dim, dim, kernel_size=3, stride=1, padding=1)
        self.mlp = ChannelMLP(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.norm_layer = nn.BatchNorm3d(dim)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.channel_proj(x)
        x = self.norm_layer(x)
        x = self.act(x)

        shortcut = x.clone()
        x = self.norm_layer(x)
        x = self.mlp(x)
        x = shortcut + x
        return x

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

class SaliencyDLKBlock(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels
                 ):

        super().__init__()

        self.conv1 = (nn.Sequential(
                NewDLK(input_channels, output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.BatchNorm3d(output_channels)
            ))

        self.conv2 = (nn.Sequential(
                NewDLK(output_channels, output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.BatchNorm3d(output_channels)
            ))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class NewDLK(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()

        self.channel_proj = nn.Conv3d(input_dim, dim//2, kernel_size=1, bias=False)
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

        return out

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