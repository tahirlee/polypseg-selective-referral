"""
CFDA-Net: Cross-domain Feature Decomposition and Aggregation Network (Section 3.2.1 of the paper).

Paper -> code
-------------
Eq. 1       encoder E1-E5             MobileNetV2 stages (ImageNet-pretrained if pretrained=True)
Eq. 2       MixStyle after each stage self.enc_mix
Eqs. 3-11   DFA at E5                 class DFA
Eqs. 12-20  MSA at E4 and E3          class MSA
Eqs. 21-23  EAA at E2 and E1          class EAA
Eqs. 24-25  decoder with SN           class DecoderBlock, class SwitchableNorm2d
Eq. 26      1x1 conv + sigmoid        self.final_conv
AdaIN       depthwise-separable convs of MSA and the decoder   class AdaptiveInstanceNorm2d
DropBlock   end of DFA and decoder blocks 5-2                  class DropBlock2d

Settings
--------
Encoder         MobileNetV2, E1-E5 = 16/24/32/96/320 channels
DFA             320 -> 256 reduction, 18-channel offsets, depthwise deformable 3x3 conv,
                spatial and channel attention (reduction 16), 1x1 projection, DropBlock last
MSA             channel shuffle, three branches with dilation rates 1, 3, 6, GroupNorm,
                learnable fusion weights (softmax-normalised, initialised equal)
EAA             x - AvgPool3x3(x) -> 1x1 conv -> SN -> sigmoid; Y = W*x + x; PReLU; dropout 0.5
MixStyle        p = 0.5 in DFA, 0.3 in the encoder stages and MSA, Beta(0.1, 0.1)
DropBlock       drop probability 0.1; block size 7 in DFA, 5 in the decoder
AdaIN           instance normalisation with learnable per-channel scale and shift

Ablation
--------
use_mixstyle, use_adain, use_sn, use_dropblock, use_dfa, use_msa and use_eaa switch one component off
(AdaIN -> GroupNorm, SN -> BatchNorm, DFA / MSA / EAA -> identity). ablation_config() builds the
additive and leave-one-out variants of the ablation study; `python model/cfda_net.py` prints the
parameters and GMACs of every variant.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models
from torchvision.ops import DeformConv2d


# ============================================================================
# Helpers
# ============================================================================

def _gn_groups(ch: int, max_groups: int = 4) -> int:
    """Largest g <= max_groups that divides ch (so GroupNorm is always valid)."""
    for g in range(max_groups, 0, -1):
        if ch % g == 0:
            return g
    return 1


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    b, c, h, w = x.size()
    if groups <= 1 or c % groups != 0:
        return x
    x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


# ============================================================================
# Normalisation / regularisation modules
# ============================================================================

class AdaptiveInstanceNorm2d(nn.Module):
    """Instance normalisation with learnable per-channel scale/shift (the "AdaIN"
    of the paper). Statistics are computed per sample at train AND test time,
    so there are no running statistics and no train/test mismatch."""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        b, c = x.shape[:2]
        xr = x.reshape(b, c, -1)
        mean = xr.mean(2).view(b, c, 1, 1)
        var = xr.var(2).view(b, c, 1, 1)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SwitchableNorm2d(nn.Module):
    """Switchable Normalisation (Eq. 25): softmax-weighted mix of BN, IN and LN
    statistics. The BN branch uses running statistics at inference, so the output
    of one image no longer depends on the other images in the batch."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        in_mean = x.mean(dim=(2, 3), keepdim=True)
        in_var = x.var(dim=(2, 3), keepdim=True)
        ln_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        ln_var = x.var(dim=(1, 2, 3), keepdim=True)
        if self.training:
            bn_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            bn_var = x.var(dim=(0, 2, 3), keepdim=True)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * bn_mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * bn_var)
        else:
            bn_mean, bn_var = self.running_mean, self.running_var
        mw = F.softmax(self.mean_weight, dim=0)
        vw = F.softmax(self.var_weight, dim=0)
        mean = mw[0] * bn_mean + mw[1] * in_mean + mw[2] * ln_mean
        var = vw[0] * bn_var + vw[1] * in_var + vw[2] * ln_var
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


def make_norm(kind: str, ch: int) -> nn.Module:
    if kind == 'adain':
        return AdaptiveInstanceNorm2d(ch)
    if kind == 'sn':
        return SwitchableNorm2d(ch)
    if kind == 'bn':
        return nn.BatchNorm2d(ch)
    if kind == 'gn':
        return nn.GroupNorm(_gn_groups(ch), ch)
    raise ValueError(kind)


class MixStyleBlock(nn.Module):
    """MixStyle (Zhou et al.): mixes channel-wise feature statistics between
    samples of a batch with probability p. Active only in training (Eq. 2)."""

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps
        self._beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if (not self.training) or torch.rand(1).item() > self.p or x.size(0) < 2:
            return x
        b, c = x.shape[:2]
        xr = x.reshape(b, c, -1)
        mu = xr.mean(2).view(b, c, 1, 1)
        sig = (xr.var(2).view(b, c, 1, 1) + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()                      # as in the official MixStyle
        x_normed = (x - mu) / sig
        lmda = self._beta.sample((b, 1, 1, 1)).to(x.device, x.dtype)
        perm = torch.randperm(b, device=x.device)
        mu_mix = lmda * mu + (1 - lmda) * mu[perm]
        sig_mix = lmda * sig + (1 - lmda) * sig[perm]
        return x_normed * sig_mix + mu_mix


class DropBlock2d(nn.Module):
    """DropBlock: structured (contiguous) feature dropout, training only."""

    def __init__(self, drop_prob=0.1, block_size=7):
        super().__init__()
        self.drop_prob, self.block_size = drop_prob, block_size

    def forward(self, x):
        if (not self.training) or self.drop_prob == 0:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = torch.bernoulli(torch.full_like(x, gamma))
        block_mask = 1 - F.max_pool2d(mask, kernel_size=self.block_size, stride=1,
                                      padding=self.block_size // 2)
        if block_mask.shape[-2:] != x.shape[-2:]:               # even block sizes
            block_mask = block_mask[..., :x.shape[-2], :x.shape[-1]]
        return x * block_mask * (block_mask.numel() / block_mask.sum().clamp(min=1.0))


def _mix(use, p):
    return MixStyleBlock(p=p) if use else nn.Identity()


def _drop(use, block_size, prob=0.1):
    return DropBlock2d(prob, block_size) if use else nn.Identity()


# ============================================================================
# Building blocks
# ============================================================================

class BasicConv2d(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DWCon(nn.Module):
    """Depthwise-separable conv + normalisation + ReLU."""

    def __init__(self, cin, cout, kernel_size=3, padding=1, dilation=1, norm='adain'):
        super().__init__()
        self.depthwise = nn.Conv2d(cin, cin, kernel_size, padding=padding,
                                   dilation=dilation, groups=cin)
        self.pointwise = nn.Conv2d(cin, cout, 1)
        self.norm = make_norm(norm, cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.pointwise(self.depthwise(x))))


class SpatialAttention(nn.Module):
    """Channel-max -> 3x3 conv -> sigmoid (one attention map per branch)."""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.conv(x.max(dim=1, keepdim=True)[0]))


# ============================================================================
# DFA: Deformable Feature Aggregation  (Eqs. 3-11)  -- applied at E5
# ============================================================================

class DFA(nn.Module):
    def __init__(self, in_ch=320, mid_ch=256, out_ch=320, reduction=16,
                 use_mixstyle=True, mix_p=0.5, use_dropblock=True,
                 depthwise_deform=True):
        super().__init__()
        r = max(mid_ch // reduction, 4)
        self.reduce = nn.Conv2d(in_ch, mid_ch, 1)                              # Eq. 3
        self.mixstyle = _mix(use_mixstyle, mix_p)                              # Eq. 4
        self.offset = nn.Conv2d(mid_ch, 2 * 3 * 3, 3, padding=1)               # Eq. 5 (18 ch)
        nn.init.zeros_(self.offset.weight)                                     # start as a regular conv
        nn.init.zeros_(self.offset.bias)
        self.deform = DeformConv2d(mid_ch, mid_ch, 3, padding=1,               # Eq. 6
                                   groups=mid_ch if depthwise_deform else 1, bias=False)
        self.spatial_att = nn.Sequential(                                      # Eq. 7
            nn.Conv2d(mid_ch, r, 1), nn.BatchNorm2d(r), nn.ReLU(inplace=True),
            nn.Conv2d(r, 1, 1), nn.Sigmoid())
        self.channel_att = nn.Sequential(                                      # Eqs. 8-9
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(mid_ch, r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(r, mid_ch, 1), nn.Sigmoid())
        self.project = nn.Conv2d(mid_ch, out_ch, 1)                            # Eq. 11
        self.dropblock = _drop(use_dropblock, 7)

    def forward(self, e5):
        f = self.mixstyle(self.reduce(e5))
        f_def = self.deform(f, self.offset(f))
        f_att = f_def * self.spatial_att(f_def) * self.channel_att(f_def)      # Eq. 10
        return self.dropblock(self.project(f_att))


# ============================================================================
# MSA: Multiscale Spatial Attention  (Eqs. 12-20)  -- applied at E4, E3
# ============================================================================

class MSA(nn.Module):
    def __init__(self, ch, dilations=(1, 3, 6), use_mixstyle=True, mix_p=0.3,
                 use_adain=True, shuffle_groups=4, apply_to='shuffled'):
        super().__init__()
        assert apply_to in ('shuffled', 'concat')
        self.apply_to = apply_to
        n = len(dilations)
        base, rem = divmod(ch, n)
        self.sizes = [base + (1 if i < rem else 0) for i in range(n)]          # E3 (32 ch): 11/11/10
        self.mixstyle = _mix(use_mixstyle, mix_p)                              # Eq. 12
        self.shuffle_groups = shuffle_groups
        norm = 'adain' if use_adain else 'gn'
        self.gns = nn.ModuleList([nn.GroupNorm(_gn_groups(c, 2), c) for c in self.sizes])
        self.dw = nn.ModuleList([DWCon(c, c, 3, padding=d, dilation=d, norm=norm)
                                 for c, d in zip(self.sizes, dilations)])      # Eqs. 15-17
        self.sa = nn.ModuleList([SpatialAttention() for _ in dilations])
        self.alpha = nn.Parameter(torch.ones(n))                               # Eq. 18, softmax-normalised

    def forward(self, x):
        f_s = channel_shuffle(self.mixstyle(x), self.shuffle_groups)           # Eq. 13
        branches = torch.split(f_s, self.sizes, dim=1)                         # Eq. 14
        a = F.softmax(self.alpha, dim=0)
        feats = [self.dw[i](self.gns[i](b)) for i, b in enumerate(branches)]   # Eqs. 15-17
        s = sum(a[i] * self.sa[i](feats[i]) for i in range(len(feats)))        # Eq. 18
        base = f_s if self.apply_to == 'shuffled' else torch.cat(feats, dim=1)
        return base * s + x                                                    # Eqs. 19-20


# ============================================================================
# EAA: Edge-Aware Attention  (Eqs. 21-23)  -- applied at E2, E1
# ============================================================================

class EAA(nn.Module):
    def __init__(self, ch, use_sn=True, dropout=0.5):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.conv = nn.Conv2d(ch, ch, 1)
        self.norm = make_norm('sn' if use_sn else 'bn', ch)
        self.act = nn.PReLU(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        edge = x - self.avg_pool(x)                                            # Eq. 21
        w = torch.sigmoid(self.norm(self.conv(edge)))                          # Eq. 22
        return self.drop(self.act(w * x + x))                                  # Eq. 23


# ============================================================================
# Decoder  (Eqs. 24-26)
# ============================================================================

class DecoderBlock(nn.Module):
    """DWConv(+AdaIN) -> DropBlock -> bilinear x2 -> 1x1 conv -> SN -> ReLU."""

    def __init__(self, cin, cout, use_adain=True, use_sn=True, use_dropblock=True):
        super().__init__()
        self.dw = DWCon(cin, cout, 3, padding=1, norm='adain' if use_adain else 'gn')
        self.dropblock = _drop(use_dropblock, 5)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(cout, cout, 1)
        self.norm = make_norm('sn' if use_sn else 'bn', cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(self.dropblock(self.dw(x)))
        return self.relu(self.norm(self.conv(x)))


def _cat(d, skip):
    if d.shape[-2:] != skip.shape[-2:]:                                        # odd input sizes
        d = F.interpolate(d, size=skip.shape[-2:], mode='bilinear', align_corners=False)
    return torch.cat([d, skip], dim=1)


# ============================================================================
# CFDA-Net
# ============================================================================

class CFDANet(nn.Module):
    def __init__(self, num_channels=3, num_classes=1, pretrained=True,
                 use_uncertainty=False,                       # unused; kept so that saved configs load
                 use_mixstyle=True, use_adain=True, use_sn=True, use_dropblock=True,
                 use_dfa=True, use_msa=True, use_eaa=True,
                 mixstyle_stages=(1, 2, 3, 4, 5), enc_mix_p=0.3,
                 depthwise_deform=True, msa_apply_to='shuffled',
                 dropblock_in_decoder=True, **unused):
        super().__init__()
        self.use_uncertainty = use_uncertainty

        # ---- Encoder: five MobileNetV2 stages (Eq. 1) --------------------
        try:
            w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            mb = models.mobilenet_v2(weights=w)
        except AttributeError:                                # old torchvision
            mb = models.mobilenet_v2(pretrained=pretrained)
        f = mb.features
        self.stem = f[0]
        self.stage1 = f[1]                                                      # 16 ch, 1/2
        self.stage2 = nn.Sequential(f[2], f[3])                                 # 24 ch, 1/4
        self.stage3 = nn.Sequential(f[4], f[5], f[6])                           # 32 ch, 1/8
        self.stage4 = nn.Sequential(f[7], f[8], f[9], f[10], f[11], f[12], f[13])  # 96 ch, 1/16
        self.stage5 = nn.Sequential(f[14], f[15], f[16], f[17])                 # 320 ch, 1/32
        self.enc_mix = nn.ModuleList(                                           # Eq. 2
            [_mix(use_mixstyle and (i in mixstyle_stages), enc_mix_p) for i in range(1, 6)])

        # ---- Feature refinement modules ---------------------------------
        self.dfa = DFA(320, 256, 320, use_mixstyle=use_mixstyle, use_dropblock=use_dropblock,
                       depthwise_deform=depthwise_deform) if use_dfa else nn.Identity()
        ms = dict(use_mixstyle=use_mixstyle, use_adain=use_adain, apply_to=msa_apply_to)
        self.msa4 = MSA(96, **ms) if use_msa else nn.Identity()
        self.msa3 = MSA(32, **ms) if use_msa else nn.Identity()
        self.eaa2 = EAA(24, use_sn) if use_eaa else nn.Identity()
        self.eaa1 = EAA(16, use_sn) if use_eaa else nn.Identity()

        # ---- Lightweight decoder ----------------------------------------
        kw = dict(use_adain=use_adain, use_sn=use_sn)
        dec_db = use_dropblock and dropblock_in_decoder
        self.dec5 = DecoderBlock(320, 64, use_dropblock=dec_db, **kw)           # 1/32 -> 1/16
        self.dec4 = DecoderBlock(64 + 96, 48, use_dropblock=dec_db, **kw)       # -> 1/8
        self.dec3 = DecoderBlock(48 + 32, 24, use_dropblock=dec_db, **kw)       # -> 1/4
        self.dec2 = DecoderBlock(24 + 24, 16, use_dropblock=dec_db, **kw)       # -> 1/2
        self.dec1 = DecoderBlock(16 + 16, 16, use_dropblock=False, **kw)          # -> 1/1
        self.final_conv = nn.Conv2d(16, num_classes, 1)                         # Eq. 26

    def forward(self, x):
        size = x.shape[-2:]
        e1 = self.enc_mix[0](self.stage1(self.stem(x)))
        e2 = self.enc_mix[1](self.stage2(e1))
        e3 = self.enc_mix[2](self.stage3(e2))
        e4 = self.enc_mix[3](self.stage4(e3))
        e5 = self.enc_mix[4](self.stage5(e4))

        d = self.dec5(self.dfa(e5))
        d = self.dec4(_cat(d, self.msa4(e4)))
        d = self.dec3(_cat(d, self.msa3(e3)))
        d = self.dec2(_cat(d, self.eaa2(e2)))
        d = self.dec1(_cat(d, self.eaa1(e1)))

        out = self.final_conv(d)
        if out.shape[-2:] != size:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=False)
        return torch.sigmoid(out)


# Alternative name of the class
CFFANet_OOD = CFDANet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Ablation helpers
# ============================================================================

COMPONENTS = ('mixstyle', 'adain', 'sn', 'dropblock', 'dfa', 'msa', 'eaa')


def ablation_config(mode='full', component=None):
    """Keyword arguments for CFDANet for each ablation variant.

    mode='full'      all seven components on (the proposed CFDA-Net)
    mode='baseline'  all seven components off (MobileNetV2 encoder + lightweight decoder,
                     GroupNorm / BatchNorm instead of AdaIN / SN)
    mode='add'       baseline + ONE component            (additive study)
    mode='drop'      full model - ONE component          (leave-one-out study)

    Example:
        CFDANet(**ablation_config('drop', 'msa'))
        CFDANet(**ablation_config('add', 'dfa'))
    """
    assert mode in ('full', 'baseline', 'add', 'drop')
    if mode == 'full':
        flags = {c: True for c in COMPONENTS}
    elif mode == 'baseline':
        flags = {c: False for c in COMPONENTS}
    else:
        assert component in COMPONENTS, 'component must be one of %s' % (COMPONENTS,)
        flags = {c: (mode == 'drop') for c in COMPONENTS}
        flags[component] = (mode == 'add')
    return {'use_' + c: v for c, v in flags.items()}


def ablation_variants():
    """Ordered {name: kwargs} for every row of the ablation table."""
    v = {'Baseline': ablation_config('baseline')}
    for c in COMPONENTS:
        v['+ ' + c] = ablation_config('add', c)
    for c in COMPONENTS:
        v['Full - ' + c] = ablation_config('drop', c)
    v['Full CFDA-Net'] = ablation_config('full')
    return v


def profile(model, size=256):
    """(parameters in M, GMACs) at a size x size input. GMACs are counted with torch's FlopCounterMode,
    which may not count the deformable-convolution operator."""
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.randn(1, 3, size, size))
    return count_parameters(model) / 1e6, fc.get_total_flops() / 2e9


if __name__ == '__main__':
    net = CFDANet(pretrained=False).eval()
    with torch.no_grad():
        y = net(torch.randn(2, 3, 256, 256))
    print('output', tuple(y.shape), 'params %.3fM' % (count_parameters(net) / 1e6))
    print('\n%-22s %10s %10s' % ('variant', 'params(M)', 'GMACs@256'))
    for name, kw in ablation_variants().items():
        p, g = profile(CFDANet(pretrained=False, **kw))
        print('%-22s %10.3f %10.3f' % (name, p, g))