# Copyright (c) CAIRI AI Lab. All rights reserved

from .convlstm import ConvLSTM
from .crevnet import CrevNet
from .e3dlstm import E3DLSTM
from .mau import MAU
from .mim import MIM
from .phydnet import PhyDNet
from .prednet import PredNet
from .predrnn import PredRNN
from .predrnnpp import PredRNNpp
from .predrnnv2 import PredRNNv2
from .simvp import SimVP
from .tau import TAU
from .dmvfn import DMVFN
from .swinlstm import SwinLSTM_D, SwinLSTM_B
from .block import GroupNorm32, TimeEmbedding, AttentionBlock, Upsample, Downsample
from .VMRNN import VMRNN_D, VMRNN_B
method_maps = {
    'convlstm': ConvLSTM,
    'crevnet': CrevNet,
    'e3dlstm': E3DLSTM,
    'mau': MAU,
    'mim': MIM,
    'phydnet': PhyDNet,
    'prednet': PredNet,
    'predrnn': PredRNN,
    'predrnnpp': PredRNNpp,
    'predrnnv2': PredRNNv2,
    'simvp': SimVP,
    'tau': TAU,
    'dmvfn': DMVFN,
    'swinlstm_d': SwinLSTM_D,
    'swinlstm_b': SwinLSTM_B,
    'swinlstm': SwinLSTM_B,
    'vmrnn_b': VMRNN_B,
    'vmrnn_d': VMRNN_D,
    'vmrnn': VMRNN_B
}

__all__ = [
    'method_maps', 'ConvLSTM', 'CrevNet', 'E3DLSTM', 'MAU', 'MIM',
    'PredRNN', 'PredRNNpp', 'PredRNNv2', 'PhyDNet', 'PredNet', 'SimVP', 
    'TAU', 'SwinLSTM_D', 'SwinLSTM_B', 'VMRNN_D','VMRNN_B'
]