import torch
import torch.nn as nn
from einops import rearrange
from timm.models.swin_transformer import PatchEmbed, PatchMerging
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
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



# cite the 'PatchExpand' code from Swin-Unet, Thanks!
# https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/
# swin_transformer_unet_skip_expand_decoder_sys.py, line 333-355
class PatchExpanding(nn.Module):
    r""" Patch Expanding Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x




class PatchInflated(nn.Module):
    r""" Tensor to Patch Inflating

    Args:
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        input_resolution (tuple[int]): Input resulotion.
    """

    def __init__(self, in_chans, embed_dim, input_resolution, stride=2, padding=1, output_padding=1):
        super(PatchInflated, self).__init__()

        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution

        self.Conv = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(3, 3),
                                       stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.Conv(x)

        return x


class VMRNNCell(nn.Module):
    def __init__(self, hidden_dim, input_resolution,depth,
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





class DownSample(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_downsample, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, d_state=16, flag=1):
        super(DownSample, self).__init__()

        self.num_layers = len(depths_downsample)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        patches_resolution = self.patch_embed.grid_size


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_downsample))]

        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i_layer in range(self.num_layers):
            downsample = PatchMerging(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                        patches_resolution[1] // (2 ** i_layer)),
                                      dim=int(embed_dim * 2 ** i_layer))

            layer = VMRNNCell(hidden_dim=int(embed_dim * 2 ** i_layer),
                                  input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                    patches_resolution[1] // (2 ** i_layer)),
                                  depth=depths_downsample[i_layer],
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[sum(depths_downsample[:i_layer]):sum(depths_downsample[:i_layer + 1])],
                                  norm_layer=norm_layer, d_state=d_state, flag=flag)

            self.layers.append(layer)
            self.downsample.append(downsample)

    def forward(self, x, y):

        x = self.patch_embed(x)

        hidden_states_down = []

        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.downsample[index](x)
            hidden_states_down.append(hidden_state)

        return hidden_states_down, x


class UpSample(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_upsample, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, d_state=16, flag=0):
        super(UpSample, self).__init__()

        self.img_size = img_size
        self.num_layers = len(depths_upsample)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        patches_resolution = self.patch_embed.grid_size
        self.Unembed = PatchInflated(in_chans=in_chans, embed_dim=embed_dim, input_resolution=patches_resolution)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_upsample))]

        self.layers = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i_layer in range(self.num_layers):
            resolution1 = (patches_resolution[0] // (2 ** (self.num_layers - i_layer)))
            resolution2 = (patches_resolution[1] // (2 ** (self.num_layers - i_layer)))

            dimension = int(embed_dim * 2 ** (self.num_layers - i_layer))
            upsample = PatchExpanding(input_resolution=(resolution1, resolution2), dim=dimension)

            layer = VMRNNCell(hidden_dim=dimension, input_resolution=(resolution1, resolution2),
                                  depth=depths_upsample[(self.num_layers - 1 - i_layer)],drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[sum(depths_upsample[:(self.num_layers - 1 - i_layer)]):
                                                                          sum(depths_upsample[:(self.num_layers - 1 - i_layer) + 1])],
                                  norm_layer=norm_layer, d_state=d_state, flag=flag)

            self.layers.append(layer)
            self.upsample.append(upsample)

    def forward(self, x, y):
        hidden_states_up = []

        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.upsample[index](x)
            hidden_states_up.append(hidden_state)

        x = torch.sigmoid(self.Unembed(x))

        return hidden_states_up, x


class VMRNN(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_downsample, depths_upsample, **kwargs):
        super(VMRNN, self).__init__()

        self.Downsample = DownSample(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                     embed_dim=embed_dim, depths_downsample=depths_downsample,
                                     **kwargs)

        self.Upsample = UpSample(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                 embed_dim=embed_dim, depths_upsample=depths_upsample,
                                 **kwargs)

    def forward(self, input, states_down, states_up,**kwargs):
        states_down, x = self.Downsample(input, states_down)
        states_up, output = self.Upsample(x, states_up)

        return output, states_down, states_up
