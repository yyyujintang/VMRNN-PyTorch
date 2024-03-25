import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.swin_transformer import PatchEmbed
from vmamba import VSSBlock, SS2D  # 确保从正确的模块导入 VSSBlock 和 SS2D
from typing import Optional, Callable
from functools import partial


class VSB(VSSBlock):
    def __init__(
        self,
        hidden_dim: int = 0,
        input_resolution: tuple = (224, 224), 
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            input_resolution=input_resolution,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            **kwargs
        )
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.input_resolution = input_resolution

    def forward(self, x, hx=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.ln_1(x)

        if hx is not None:
            hx = self.ln_1(hx)
            x = torch.cat((x, hx), dim=-1)
            x = self.linear(x)
        x = x.view(B, H, W, C) 

        x = self.drop_path(self.self_attention(x))
 
        x = x.view(B, H * W, C)
        x = shortcut + x

        return x


class PatchInflated(nn.Module):
    r""" Tensor to Patch Inflating

    Args:
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        input_resolution (tuple[int]): Input resulotion.
    """

    def __init__(self, in_chans, embed_dim, input_resolution, patch_size, stride=2, padding=1, output_padding=1):
        super(PatchInflated, self).__init__()

        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution
        self.patch_size = patch_size

        self.Conv = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(3, 3),
                                       stride=stride, padding=padding, output_padding=output_padding)
        
        modules = []
        for _ in range(patch_size // 4):
            modules.append(nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3),
                                              stride=stride, padding=padding, output_padding=output_padding))
            modules.append(nn.GroupNorm(16, embed_dim))
            modules.append(nn.LeakyReLU(0.2, inplace=True))

        self.Conv_ = nn.Sequential(*modules)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        if self.patch_size > 2:
            x = self.Conv_(x)
        x = self.Conv(x)

        return x


class VMRNNCell(nn.Module):
    def __init__(self, hidden_dim, input_resolution, depth,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, d_state=16, **kwargs):
        """
        Args:
        hidden_dim: Dimension of the hidden layer.
        input_resolution: Tuple of the input resolution.
        depth: Depth of the cell.
        drop, attn_drop, drop_path: Parameters for VSB.
        norm_layer: Normalization layer.
        d_state: State dimension for SS2D in VSB.
        """
        super(VMRNNCell, self).__init__()


        self.VSBs = nn.ModuleList(
            VSB(hidden_dim=hidden_dim, input_resolution=input_resolution,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, attn_drop_rate=attn_drop,
                d_state=d_state, **kwargs)
            for i in range(depth))

    def forward(self, xt, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C).to(xt.device)
            cx = torch.zeros(B, L, C).to(xt.device)
        else:
            # print(hidden_states)
            hx, cx = hidden_states
        
        outputs = []
        for index, layer in enumerate(self.VSBs):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)
            else:
                x = layer(outputs[-1], None)  # Assuming VSB does not use hx for layers after the first
                outputs.append(x)
                
        o_t = outputs[-1]
        Ft = torch.sigmoid(o_t)

        cell = torch.tanh(o_t)

        Ct = Ft * (cx + cell)
        Ht = Ft * torch.tanh(Ct)

        return Ht, (Ht, Ct)


class MSTConvert(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, 
                 norm_layer=nn.LayerNorm, d_state=16, flag=2):
        super(MSTConvert, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                      in_chans=in_chans, embed_dim=embed_dim, 
                                      norm_layer=norm_layer)
        patches_resolution = self.patch_embed.grid_size

        self.patch_inflated = PatchInflated(in_chans=in_chans, embed_dim=embed_dim,
                                            input_resolution=patches_resolution, patch_size=patch_size)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VMRNNCell(hidden_dim=embed_dim, 
                                input_resolution=(patches_resolution[0], patches_resolution[1]), 
                                depth=depths[i_layer], drop=drop_rate, 
                                attn_drop=attn_drop_rate, drop_path=drop_path_rate, 
                                norm_layer=norm_layer, d_state=d_state, flag=flag)
            self.layers.append(layer)

    def forward(self, x, h=None):
        x = self.patch_embed(x)
        hidden_states = []

        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, h[index])
            hidden_states.append(hidden_state)

        x = torch.sigmoid(self.patch_inflated(x))

        return hidden_states, x


class VMRNN(nn.Module):

    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths,
                  drop_rate, attn_drop_rate, drop_path_rate):
        super(VMRNN, self).__init__()

        self.ST = MSTConvert(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                            embed_dim=embed_dim, depths=depths,drop_rate=drop_rate,
                            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate)

    def forward(self, input, states):
        states_next, output = self.ST(input, states)

        return output, states_next
