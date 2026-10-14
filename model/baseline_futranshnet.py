"""
Fu-TransHNet (Luo et al., Pattern Recognition 170 (2026) 112116), retrained under the CFDA-Net protocol.

Uses the authors' model code in ./lib (TransFuse.py, DeiT.py, vision_transformer.py, HarDMSEG.py, hardnet_68.py)
without changing it. This wrapper only:
  * keeps all three views of the paper: CNN branch (map_x), Transformer branch (map_1) and fusion module (map_2);
  * weights the views adaptively, w_k = softmax(-loss_k / lambda) with lambda = 1 (Eq. 20 of the paper), and
    returns the weighted sum of the three predictions as the final output (Eq. 22);
  * runs at any input size divisible by 32 (the original code is fixed to 352 x 352): the DeiT position embedding
    is interpolated from the 14 x 14 ImageNet grid to (size/16) x (size/16), as the original code does for 22 x 22;
  * initialises the encoders with ImageNet weights only (DeiT-S and HarDNet68). The polyp-trained HarD-MSEG
    checkpoint of the original code is NOT used, because it was trained on Kvasir and CVC-ClinicDB images, and
    CVC-ClinicDB is one of our unseen test sets.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project folder (contains lib/)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import lib.hardnet_68 as _hardnet_mod  # noqa: E402
import lib.HarDMSEG as _hardmseg_mod   # noqa: E402


def _hardnet_factory(weight_path):
    """Replaces lib.hardnet_68.hardnet, which loads from a fixed path on the authors' computer."""
    def hardnet(arch=68, pretrained=True, **kwargs):
        model = _hardnet_mod.HarDNet(arch=arch)
        if weight_path:
            sd = torch.load(weight_path, map_location='cpu')
            sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
            msg = model.load_state_dict(sd, strict=False)
            print(f'HarDNet{arch} ImageNet weights loaded from {weight_path} '
                  f'({len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected keys)')
        return model
    return hardnet


class FuTransHNet(nn.Module):
    N_VIEWS = 3   # CNN branch, Transformer branch, fusion module

    def __init__(self, img_size=256, deit_ckpt=None, hardnet_ckpt=None, lam=1.0):
        super().__init__()
        if img_size % 32:
            raise ValueError('img_size must be divisible by 32')
        _hardmseg_mod.hardnet = _hardnet_factory(hardnet_ckpt)
        from lib.TransFuse import TransFuse_S
        self.net = TransFuse_S(pretrained=False)          # pretrained=False: avoid the hard-coded paths
        self.grid = img_size // 16
        self.lam = lam
        self._init_deit(deit_ckpt)
        # running view losses -> view weights used for the final (inference) output, saved with the model
        self.register_buffer('view_loss_ema', torch.ones(self.N_VIEWS))
        self.register_buffer('view_w', torch.full((self.N_VIEWS,), 1.0 / self.N_VIEWS))

    # ------------------------------------------------------------------ encoders
    def _init_deit(self, ckpt_path):
        t = self.net.transformer
        if ckpt_path:
            ck = torch.load(ckpt_path, map_location='cpu')
            sd = dict(ck.get('model', ck))
            pe = sd.pop('pos_embed')[:, 1:, :]            # drop the class-token position -> [1, 196, 384]
            msg = t.load_state_dict(sd, strict=False)
            print(f'DeiT-S ImageNet weights loaded from {ckpt_path} '
                  f'({len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected keys; '
                  f'the authors use the first 8 of 12 blocks)')
        else:
            pe = t.pos_embed.detach()                     # [1, g*g, 384]; replaced by the checkpoint at test time
        g0 = int(round(math.sqrt(pe.shape[1])))
        pe = pe.transpose(1, 2).reshape(1, pe.shape[2], g0, g0)
        pe = F.interpolate(pe, size=(self.grid, self.grid), mode='bilinear', align_corners=True)
        t.pos_embed = nn.Parameter(pe.flatten(2).transpose(1, 2).contiguous())

    # ------------------------------------------------------------------ the three views (original forward)
    def _views(self, imgs):
        n = self.net
        x_b = n.transformer(imgs).transpose(1, 2)
        g = int(round(math.sqrt(x_b.shape[2])))
        x_b = n.drop(x_b.reshape(x_b.shape[0], -1, g, g))
        x_b_1 = n.drop(n.up1(x_b))
        x_b_2 = n.drop(n.up2(x_b_1))

        x_u = n.net(imgs)
        x_u = F.interpolate(x_u, scale_factor=0.125)
        x_u_2 = n.drop(n.up3(x_u))
        x_u_1 = n.drop(F.interpolate(n.up4(x_u_2), scale_factor=0.25))
        x_u = n.drop(F.interpolate(n.up5(x_u_1), scale_factor=0.25))

        x_c = n.fusion1(x_u, x_b)
        x_c_1_1 = n.fusion2(x_u_1, x_b_1)
        x_c_2_1 = n.fusion3(x_u_2, x_b_2)
        f_all = n.CFM(x_c, x_c_1_1, x_c_2_1)

        map_x = F.interpolate(n.final_x(x_u_2), scale_factor=4, mode='bilinear')   # CNN view
        map_1 = F.interpolate(n.final_1(x_b_2), scale_factor=4, mode='bilinear')   # Transformer view
        map_2 = F.interpolate(n.final_2(f_all), scale_factor=4, mode='bilinear')   # fusion view
        return [map_x, map_1, map_2]

    def forward(self, x):
        probs = [torch.sigmoid(m) for m in self._views(x)]
        if self.training:
            return probs
        return sum(self.view_w[k] * probs[k] for k in range(self.N_VIEWS))   # Eq. (22)

    # ------------------------------------------------------------------ multi-view loss (Eqs. 11, 20, 21)
    def view_loss(self, probs, target, criterion):
        losses = torch.stack([criterion(p, target) for p in probs])
        with torch.no_grad():
            w = torch.softmax(-losses / self.lam, dim=0)                 # Eq. (20), Theta fixed
            self.view_loss_ema.mul_(0.99).add_(0.01 * losses)
            self.view_w.copy_(torch.softmax(-self.view_loss_ema / self.lam, dim=0))
        return (w * losses).sum()                                          # Eq. (21), w fixed
