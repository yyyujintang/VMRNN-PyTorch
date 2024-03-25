import os
import math
import torch
import torch.nn as nn


def GroupNorm32(channels):
    return nn.GroupNorm(32, channels)


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        # Create sinusoidal position embeddings (same as those from the transformer)
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class ClassEmbedding(nn.Module):
    def __init__(self, n_classes, n_channels):
        """
        * `n_classes` is the number of classes in the condition
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_classes = n_classes
        self.lin1 = nn.Linear(n_classes, n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(n_channels, n_channels)
    
    def forward(self, c, drop_mask):
        """
        * `c` is the labels
        * `drop_mask`: mask out the condition if drop_mask == 1
        """
        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        # mask out context if context_mask == 1
        drop_mask = drop_mask[:, None]
        drop_mask = drop_mask.repeat(1, self.n_classes)
        drop_mask = (-1 * (1 - drop_mask))  # need to flip 0 <-> 1
        c = c * drop_mask

        # Transform with the MLP
        emb = self.act(self.lin1(c))
        emb = self.lin2(emb)
        return emb


class ClassEmbeddingTable(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, n_classes, n_channels):
        super().__init__()
        self.n_classes = n_classes
        self.embedding_table = nn.Embedding(n_classes + 1, n_channels)

    def forward(self, c, drop_mask):
        assert (c < self.n_classes).all()
        c = torch.where(drop_mask==0, c, self.n_classes)
        assert ((c == self.n_classes) == (drop_mask)).all()

        embeddings = self.embedding_table(c)
        return embeddings


class AttentionBlock(nn.Module):
    def __init__(self, n_channels, d_k):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        n_heads = n_channels // d_k

        self.norm = GroupNorm32(n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)

        self.scale = 1 / math.sqrt(math.sqrt(d_k))
        self.n_heads = n_heads
        self.d_k = d_k
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            print(f"{self.n_heads} heads, {self.d_k} channels per head")

    def forward(self, x):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        """
        batch_size, n_channels, height, width = x.shape
        # Normalize and rearrange to `[batch_size, seq, n_channels]`
        h = self.norm(x).view(batch_size, n_channels, -1).permute(0, 2, 1)

        # {q, k, v} all have a shape of `[batch_size, seq, n_heads, d_k]`
        qkv = self.projection(h).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn = torch.einsum('bihd,bjhd->bijh', q * self.scale, k * self.scale) # More stable with f16 than dividing afterwards
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)

        # Reshape to `[batch_size, seq, n_heads * d_k]` and transform to `[batch_size, seq, n_channels]`
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res + x


class Upsample(nn.Module):
    def __init__(self, n_channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            return self.conv(x)
        else:
            return x


class Downsample(nn.Module):
    def __init__(self, n_channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        if self.use_conv:
            return self.conv(x)
        else:
            return self.pool(x)

