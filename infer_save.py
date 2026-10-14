"""
Step 1 of the calibration and failure-detection analysis (Sections 4.4.2-4.4.3 and 4.7 of the paper):
run a trained CFDA-Net once and save everything that analyze_calibration.py needs.

For the validation images and every test dataset it saves
  <out_dir>/<dataset>/logits/<name>.npy   logit map at the network resolution (256 x 256, float16)
  <out_dir>/<dataset>/gt/<name>.png       ground truth resized to the network resolution (nearest neighbour)
  <out_dir>/<dataset>/cases.csv           one row per image:
        segmentation metrics at the original resolution (same post-processing as test_paper.py),
        confidence descriptors (Section 3.2.2), entropy scores, image-level indicators,
        MC-dropout scores (K = 20 passes; dropout and DropBlock active), PolypGen center.

Example (one training run):
  python infer_save.py --checkpoint ./trained_models/CFDA_seed6645/best_model.pth ^
      --out_dir ./calibration_outputs/CFDA_seed6645
"""
import argparse
import csv
import os
import re
import shutil
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from calib_utils import (prob_to_logit, postprocess, seg_metrics, confidence_features, binary_entropy)
from model.cfda_net import CFDANet

def torch_load(path, device):
    """Load our own checkpoints on any PyTorch version (2.6+ defaults to weights_only=True)."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:            # PyTorch < 1.13 has no weights_only argument
        return torch.load(path, map_location=device)


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--data_root', type=str, default='./dataset/test/')
parser.add_argument('--datasets', nargs='*', default=['Kvasir-SEG', 'CVC-ClinicDB', 'BKAI-IGH', 'PolypGen'],
                    help='test datasets; give --datasets with no names to process only the validation set')
parser.add_argument('--image_dir', type=str, default='images', help='sub-folder with images')
parser.add_argument('--gt_dir', type=str, default='masks_binary', help='sub-folder with masks')
parser.add_argument('--val_image_root', type=str, default='./dataset/validation/Kvasir-SEG/images/',
                    help='folder with the 88 validation images (used to fit the temperature)')
parser.add_argument('--val_gt_root', type=str, default='./dataset/validation/Kvasir-SEG/masks_binary/')
parser.add_argument('--val_list', type=str, default='',
                    help='optional text file with the validation file names (if they are inside a larger folder)')
parser.add_argument('--out_dir', type=str, default='./calibration_outputs/')
parser.add_argument('--testsize', type=int, default=256)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--min_size', type=int, default=300,
                    help='post-processing: remove components smaller than this (pixels)')
parser.add_argument('--mc_passes', type=int, default=20, help='MC-dropout forward passes (0 = skip)')
parser.add_argument('--seed', type=int, default=0)
opt = parser.parse_args()

IMG_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')
CENTER_RE = re.compile(r'(?:^|[\\/_\-])(?:data_)?C([1-6])(?:[\\/_\-]|$)', re.IGNORECASE)


def pairs_in(img_root, gt_root, names=None):
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_root) if f.lower().endswith(IMG_EXT)}
    gts = {os.path.splitext(f)[0]: f for f in os.listdir(gt_root) if f.lower().endswith(IMG_EXT)}
    stems = sorted(set(imgs) & set(gts))
    if names is not None:
        wanted = {os.path.splitext(n.strip())[0] for n in names if n.strip()}
        stems = [s for s in stems if s in wanted]
        if len(stems) != len(wanted):
            print(f'WARNING: {len(wanted) - len(stems)} names in --val_list were not found.')
    return [(os.path.join(img_root, imgs[s]), os.path.join(gt_root, gts[s]), s) for s in stems]


def enable_mc_dropout(model):
    """Dropout / DropBlock active, everything else (BN, SN, MixStyle) in eval mode."""
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)) or 'dropblock' in type(m).__name__.lower():
            m.train()
            n += 1
    return n


def main():
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CFDANet(num_channels=3, num_classes=1, pretrained=False)
    ckpt = torch_load(opt.checkpoint, device)
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'{opt.checkpoint}: {len(missing)} missing / {len(unexpected)} unexpected weights. '
                           f'The checkpoint does not match model/cfda_net.py.')
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((opt.testsize, opt.testsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    jobs = []
    if opt.val_image_root:
        names = open(opt.val_list).readlines() if opt.val_list else None
        jobs.append(('Validation', pairs_in(opt.val_image_root, opt.val_gt_root or opt.val_image_root, names)))
    else:
        print('WARNING: no validation set given; temperature scaling cannot be fitted.')
    for ds in opt.datasets:
        root = os.path.join(opt.data_root, ds)
        jobs.append((ds, pairs_in(os.path.join(root, opt.image_dir), os.path.join(root, opt.gt_dir))))

    for ds, pairs in jobs:
        print(f'\n=== {ds}: {len(pairs)} images ===')
        out = os.path.join(opt.out_dir, ds)
        if os.path.isdir(out):
            shutil.rmtree(out)          # never mix outputs of different runs
        os.makedirs(os.path.join(out, 'logits'), exist_ok=True)
        os.makedirs(os.path.join(out, 'gt'), exist_ok=True)
        rows = []
        t0 = time.time()
        for k, (img_path, gt_path, stem) in enumerate(pairs, 1):
            img = Image.open(img_path).convert('RGB')
            gt = np.array(Image.open(gt_path).convert('L'), dtype=np.float32)
            gt = gt / (gt.max() + 1e-8)
            x = tf(img).unsqueeze(0).to(device)

            with torch.no_grad():
                model.eval()
                prob_net = model(x)                                        # (1,1,H,W) at network resolution, sigmoid output
                prob_full = F.interpolate(prob_net, size=gt.shape, mode='bilinear', align_corners=False)
            p_net = prob_net.squeeze().cpu().numpy()
            pf = prob_full.squeeze().cpu().numpy()

            # save network-resolution logits and GT for pixel-level calibration
            np.save(os.path.join(out, 'logits', stem + '.npy'), prob_to_logit(p_net).astype(np.float16))
            gt_net = cv2.resize((gt > 0.5).astype(np.uint8), (opt.testsize, opt.testsize),
                               interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(out, 'gt', stem + '.png'), gt_net * 255)

            # case-level metrics and descriptors at the original resolution (Section 4.3)
            pp = postprocess(pf, opt.threshold, opt.min_size)
            row = dict(dataset=ds, name=stem, height=gt.shape[0], width=gt.shape[1])
            m = CENTER_RE.search(img_path)
            row['center'] = f'C{m.group(1)}' if (m and ds.lower().startswith('polypgen')) else ''
            row.update(seg_metrics(gt, pp))
            row['dsc_raw'] = seg_metrics(gt, (pf > opt.threshold).astype(np.uint8))['dsc']
            row['gt_area'] = int((gt > 0.5).sum())
            f = confidence_features(pf, pp, opt.threshold)
            row.update(f)

            # MC dropout
            if opt.mc_passes > 0:
                n_drop = enable_mc_dropout(model)
                with torch.no_grad():
                    samples = torch.cat([model(x) for _ in range(opt.mc_passes)], 0).squeeze(1)
                model.eval()
                s = samples.cpu().numpy()                                 # (K, H, W)
                mean = s.mean(0)
                var = s.var(0)
                mi = binary_entropy(mean) - binary_entropy(s).mean(0)     # mutual information (BALD)
                fg = mean > opt.threshold
                row['mc_var_image'] = float(var.mean())
                row['mc_mi_image'] = float(mi.mean())
                row['mc_var_fg'] = float(var[fg].mean()) if fg.any() else float(var.max())
                row['mc_mi_fg'] = float(mi[fg].mean()) if fg.any() else float(mi.max())
                row['mc_dropout_layers'] = n_drop
            rows.append(row)
            if k % 50 == 0 or k == len(pairs):
                print(f'  {k}/{len(pairs)}  ({time.time() - t0:.0f}s)')

        with open(os.path.join(out, 'cases.csv'), 'w', newline='') as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        d = np.array([r['dsc'] for r in rows])
        print(f'  mDSC = {d.mean():.4f}  (should equal the result of test_paper.py for this model)')


if __name__ == '__main__':
    main()
