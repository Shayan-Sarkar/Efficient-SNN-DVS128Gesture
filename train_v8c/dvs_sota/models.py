from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import snntorch as snn
    from snntorch import surrogate, utils as snn_utils
    _SNN_OK = True
except ImportError:
    _SNN_OK = False


class _ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.sc: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.sc = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.sc(x), inplace=True)


class _TemporalAttention(nn.Module):
    def __init__(self, d: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.pos  = nn.Parameter(torch.zeros(1, 64, d))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        T = x.size(1)
        x = x + self.pos[:, :T]
        out, _ = self.attn(x, x, x)
        return self.norm(x + out).mean(1)


class ResNet2DT(nn.Module):

    def __init__(self, num_classes: int = 11, dropout: float = 0.4,
                 attn_heads: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            _ResBlock2D(32,  64,  stride=1),
            _ResBlock2D(64,  64,  stride=1),
        )
        self.stage2 = nn.Sequential(
            _ResBlock2D(64,  128, stride=2),
            _ResBlock2D(128, 128, stride=1),
        )
        self.stage3 = nn.Sequential(
            _ResBlock2D(128, 256, stride=2),
            _ResBlock2D(256, 256, stride=1),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.temporal_attn = _TemporalAttention(256, heads=attn_heads,
                                                dropout=0.1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        x = x.reshape(B, T, -1)
        x = self.temporal_attn(x)
        return self.fc(self.drop(x))


ResNet3D = ResNet2DT


class EfficientResNet2DT(nn.Module):

    def __init__(self, num_classes: int = 11, dropout: float = 0.4,
                 attn_heads: int = 4):
        super().__init__()
        d = 128
        self.stem = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            _ResBlock2D(16,  32,  stride=1),
            _ResBlock2D(32,  32,  stride=1),
        )
        self.stage2 = nn.Sequential(
            _ResBlock2D(32,  64,  stride=2),
            _ResBlock2D(64,  64,  stride=1),
        )
        self.stage3 = nn.Sequential(
            _ResBlock2D(64,  128, stride=2),
            _ResBlock2D(128, 128, stride=1),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.temporal_attn = _TemporalAttention(d, heads=attn_heads, dropout=0.1)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(d, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        x = x.reshape(B, T, -1)
        x = self.temporal_attn(x)
        return self.fc(self.drop(x))


def _lif(beta: float = 0.9) -> 'snn.Leaky':
    return snn.Leaky(
        beta=beta,
        spike_grad=surrogate.fast_sigmoid(slope=25),
        learn_beta=True,
        init_hidden=True,
    )


class _SpikeResBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, beta: float = 0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,
                               stride=stride, padding=1, bias=False)
        self.in1   = nn.InstanceNorm2d(out_ch, affine=True)
        self.lif1  = _lif(beta)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.in2   = nn.InstanceNorm2d(out_ch, affine=True)
        self.lif2  = _lif(beta)

        if stride != 1 or in_ch != out_ch:
            self.sc: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
            )
        else:
            self.sc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1  = self.in1(self.conv1(x))
        spk1 = self.lif1(z1)
        z2  = self.in2(self.conv2(spk1))
        return self.lif2(z2 + self.sc(x))


class _SEBlock(nn.Module):

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(-2, -1))
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class _SpikeResBlockSE(nn.Module):

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, beta: float = 0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,
                               stride=stride, padding=1, bias=False)
        self.in1   = nn.InstanceNorm2d(out_ch, affine=True)
        self.lif1  = _lif(beta)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.in2   = nn.InstanceNorm2d(out_ch, affine=True)
        self.lif2  = _lif(beta)

        self.se    = _SEBlock(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.sc: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
            )
        else:
            self.sc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1   = self.in1(self.conv1(x))
        spk1 = self.lif1(z1)
        z2   = self.in2(self.conv2(spk1))
        spk2 = self.lif2(z2 + self.sc(x))
        return self.se(spk2)


class EfficientSpikingResNetV2(nn.Module):

    def __init__(self, num_classes: int = 11, T: int = 20, beta: float = 0.9,
                 head_dropout: float = 0.0, temporal_attn: bool = False):
        super().__init__()
        if not _SNN_OK:
            raise ImportError('snntorch is required for EfficientSpikingResNetV2')
        self.T = T

        self.stem_conv = nn.Conv2d(2, 16, 3, padding=1, bias=False)
        self.stem_in   = nn.InstanceNorm2d(16, affine=True)
        self.stem_lif  = _lif(beta)

        self.blk1 = _SpikeResBlockSE(16,  16,  stride=1, beta=beta)
        self.blk2 = _SpikeResBlockSE(16,  32,  stride=2, beta=beta)
        self.blk3 = _SpikeResBlockSE(32,  64,  stride=2, beta=beta)
        self.blk4 = _SpikeResBlockSE(64,  128, stride=2, beta=beta)

        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(head_dropout) if head_dropout > 0.0 else nn.Identity()
        self.fc   = nn.Linear(128, num_classes)

        if temporal_attn:
            self.temporal_scorer = nn.Linear(128, 1, bias=False)

    def forward(self, x: torch.Tensor,
                return_feat: bool = False):
        B, C, T, H, W = x.shape
        snn_utils.reset(self)
        feat_list: list = []

        for t in range(T):
            xt   = x[:, :, t]
            spk  = self.stem_lif(self.stem_in(self.stem_conv(xt)))
            spk  = self.blk1(spk)
            spk  = self.blk2(spk)
            spk  = self.blk3(spk)
            spk  = self.blk4(spk)
            feat_list.append(self.gap(spk).flatten(1))

        feats = torch.stack(feat_list, dim=1)

        if hasattr(self, 'temporal_scorer'):
            scores  = self.temporal_scorer(feats).squeeze(-1)
            weights = F.softmax(scores, dim=1).unsqueeze(-1)
            agg     = (feats * weights).sum(1)
        else:
            agg = feats.mean(1)

        logits = self.fc(self.drop(agg))

        if return_feat:
            return logits, agg
        return logits
