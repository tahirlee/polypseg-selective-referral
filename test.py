"""
Evaluation of trained CFDA-Net models (Sections 4.4-4.7 of the paper).

For every model (one per training seed) and every test dataset it computes:
  * segmentation metrics per image, averaged over the dataset (mIoU, mDSC, recall, precision, F2),
    and the dataset-level (pooled-pixel) precision, recall and F2
  * the 95% bootstrap confidence interval of mDSC
  * the confidence descriptors and the triage decision (Algorithm 1, Eq. 34) for each image
  * triage results: referral rate, mDSC of approved vs. referred cases (Mann-Whitney U test),
    sensitivity, specificity and false-safe rate for failures (DSC < 0.5 and < 0.7)
  * the triage thresholds, derived on each model's own validation images (Eq. 33)
  * PolypGen results per acquisition center (when the center folders data_C1 ... data_C6 are evaluated)
  * inference speed (batch 1, FP32, warm-up, model forward pass only)
Across models it reports mean +/- SD.

Settings used in the paper (all are the defaults): input 256 x 256, binarization threshold 0.5,
post-processing with morphological opening/closing and removal of components < 300 pixels,
thresholds derived on the 88 validation images.

Outputs (in --out_dir):
  <model_tag>/per_image_<dataset>.csv     every image: metrics, confidence descriptors, flag
  <model_tag>/metrics.json                 all numbers of that model
  summary_per_model.csv                    one row per model x dataset
  summary_mean_sd.csv / summary.md         mean +/- SD over models

Example (the three runs of the paper):
  python test_paper.py --models ./trained_models/CFDA_seed6645/best_model.pth ^
      ./trained_models/CFDA_seed1234/best_model.pth ./trained_models/CFDA_seed2024/best_model.pth
"""
import argparse
import json
import os
import re
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from scipy import stats

from model.cfda_net import CFDANet

def torch_load(path, device):
    """Load our own checkpoints on any PyTorch version (2.6+ defaults to weights_only=True)."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:            # PyTorch < 1.13 has no weights_only argument
        return torch.load(path, map_location=device)


parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+', required=True, help='one or more checkpoint files')
parser.add_argument('--data_root', type=str, default='./dataset/test/')
parser.add_argument('--datasets', nargs='+', default=['Kvasir-SEG', 'CVC-ClinicDB', 'BKAI-IGH', 'PolypGen'])
parser.add_argument('--image_dir', type=str, default='images')
parser.add_argument('--gt_dir', type=str, default='masks_binary')
parser.add_argument('--val_image_root', type=str, default='./dataset/validation/Kvasir-SEG/images/',
                    help='the 88 validation images used to derive the triage thresholds')
parser.add_argument('--val_gt_root', type=str, default='./dataset/validation/Kvasir-SEG/masks_binary/')
parser.add_argument('--out_dir', type=str, default='./results/')
parser.add_argument('--testsize', type=int, default=256)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--min_size', type=int, default=300, help='post-processing: remove components < min_size px')
parser.add_argument('--thresholds', type=str, default='validation', choices=['validation', 'fixed'],
                    help="'validation' (paper): mean category shares on the validation images (Eq. 33); "
                         "'fixed': use --vh/--h/--u, e.g. for the threshold sensitivity analysis")
parser.add_argument('--vh', type=float, default=None, help='fixed theta_VH (only with --thresholds fixed)')
parser.add_argument('--h', type=float, default=None, help='fixed theta_H (only with --thresholds fixed)')
parser.add_argument('--u', type=float, default=None, help='fixed theta_U (only with --thresholds fixed)')
parser.add_argument('--n_boot', type=int, default=1000)
parser.add_argument('--fps', type=int, default=1, help='1 = measure inference speed')
parser.add_argument('--save_masks', type=int, default=0, help='1 = save predicted masks as PNG')
parser.add_argument('--arch', type=str, default='cfda', choices=['cfda', 'futranshnet'],
                    help="'cfda' = CFDA-Net; 'futranshnet' = Fu-TransHNet baseline (Luo et al., 2026)")
opt = parser.parse_args()

IMG_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')
CENTER_RE = re.compile(r'(?:^|[\\/_\-])(?:data_)?C([1-6])(?:[\\/_\-]|$)', re.IGNORECASE)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TF = transforms.Compose([transforms.Resize((opt.testsize, opt.testsize)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def pairs_in(img_root, gt_root, names=None):
    """Pairs image and mask by file name (not by sort order)."""
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(img_root) if f.lower().endswith(IMG_EXT)}
    gts = {os.path.splitext(f)[0]: f for f in os.listdir(gt_root) if f.lower().endswith(IMG_EXT)}
    stems = sorted(set(imgs) & set(gts))
    if names is not None:
        wanted = {os.path.splitext(n.strip())[0] for n in names if n.strip()}
        stems = [s for s in stems if s in wanted]
    return [(os.path.join(img_root, imgs[s]), os.path.join(gt_root, gts[s]), s) for s in stems]


# ----------------------------------------------------------------------------
# Post-processing, metrics, confidence descriptors (Sections 3.2.2 and 4.4)
# ----------------------------------------------------------------------------
def postprocess(prob):
    b = (prob > opt.threshold).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = np.zeros_like(b)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= opt.min_size:
            out[lab == i] = 1
    return out


def metrics(gt, pred):
    g = (gt > 0.5).astype(np.float64)
    p = (pred > 0.5).astype(np.float64)
    tp, fp, fn = (g * p).sum(), ((1 - g) * p).sum(), (g * (1 - p)).sum()
    prec = tp / (tp + fp + 1e-5)
    rec = tp / (tp + fn + 1e-5)
    return dict(dsc=2 * tp / (g.sum() + p.sum() + 1e-5), iou=tp / (g.sum() + p.sum() - tp + 1e-5),
                precision=prec, recall=rec, f2=5 * prec * rec / (4 * prec + rec + 1e-5), tp=tp, fp=fp, fn=fn)


def confidence(prob):
    fg = prob > opt.threshold
    n = int(fg.sum())
    if n == 0:
        return dict(n_pred_pixels=0, mean_conf=0.0, std_conf=0.0, uncertain_pct=0.0, high_pct=0.0,
                    very_high_pct=0.0)
    c = prob[fg]
    return dict(n_pred_pixels=n, mean_conf=float(c.mean()), std_conf=float(c.std()),
                uncertain_pct=float(((c >= 0.5) & (c < 0.7)).mean() * 100),
                high_pct=float(((c >= 0.7) & (c < 0.9)).mean() * 100),
                very_high_pct=float((c >= 0.9).mean() * 100))


def flag(c, thr):
    """Low-confidence rule of Algorithm 1 (Eq. 34): refer if all three shares are less confident than
    the validation profile. Cases without a detected polyp are handled by the no-detection safeguard,
    which is reported separately (columns *_with_safeguard), as in Section 4.7.1."""
    if c['n_pred_pixels'] == 0:
        return 0
    return int(c['very_high_pct'] < thr['vh'] and c['high_pct'] > thr['h'] and c['uncertain_pct'] > thr['u'])


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_model(path):
    if opt.arch == 'futranshnet':
        from model.baseline_futranshnet import FuTransHNet
        model = FuTransHNet(img_size=opt.testsize)
    else:
        model = CFDANet(num_channels=3, num_classes=1, pretrained=False)
    ck = torch_load(path, DEVICE)
    sd = ck.get('model_state_dict', ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'{path}: {len(missing)} missing / {len(unexpected)} unexpected weights. '
                           f'The checkpoint does not match --arch {opt.arch}.')
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model, img_path, out_shape):
    x = TF(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE)
    p = F.interpolate(model(x), size=out_shape, mode='bilinear', align_corners=False)
    return p.squeeze().cpu().numpy()


@torch.no_grad()
def measure_fps(model, n_warmup=20, n_runs=200):
    """Batch size 1, FP32, testsize x testsize input, model forward only (no I/O, no post-processing)."""
    x = torch.randn(1, 3, opt.testsize, opt.testsize, device=DEVICE)
    for _ in range(n_warmup):
        model(x)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_runs * 1000
    return dict(ms_per_image=ms, fps=1000 / ms, batch_size=1, precision='FP32', warmup=n_warmup, runs=n_runs,
                device=torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu')


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def run_set(model, pairs, dataset, thr, save_dir=None):
    rows = []
    t0 = time.time()
    for k, (img_path, gt_path, stem) in enumerate(pairs, 1):
        if k % 100 == 0 or k == len(pairs):
            print(f'    {dataset}: {k}/{len(pairs)} images ({time.time() - t0:.0f}s)', flush=True)
        gt = np.asarray(Image.open(gt_path).convert('L'), np.float32)
        gt = gt / (gt.max() + 1e-8)
        prob = predict(model, img_path, gt.shape)
        pred = postprocess(prob)
        r = dict(dataset=dataset, name=stem)
        m = CENTER_RE.search(img_path)
        r['center'] = f'C{m.group(1)}' if (m and dataset.lower().startswith('polypgen')) else ''
        r.update(metrics(gt, pred))
        r.update(confidence(prob))
        r['empty_mask'] = int(pred.sum() == 0)
        r['flag'] = flag(r, thr) if thr else 0
        rows.append(r)
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, stem + '.png'), pred * 255)
    return pd.DataFrame(rows)


def boot_ci(x):
    rng = np.random.default_rng(0)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(opt.n_boot)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(df):
    s = dict(n=len(df), mIoU=df.iou.mean(), mDSC=df.dsc.mean(), recall=df.recall.mean(),
             precision=df.precision.mean(), F2=df.f2.mean())
    s['mDSC_CI95_low'], s['mDSC_CI95_high'] = boot_ci(df.dsc.values)
    tp, fp, fn = df.tp.sum(), df.fp.sum(), df.fn.sum()          # dataset-level (pooled pixels)
    p, r = tp / (tp + fp), tp / (tp + fn)
    s.update(pooled_precision=p, pooled_recall=r, pooled_F2=5 * p * r / (4 * p + r))
    fl = df.flag.values.astype(bool)
    s.update(n_referred=int(fl.sum()), referral_rate=fl.mean(),
             mDSC_approved=df.dsc[~fl].mean() if (~fl).any() else np.nan,
             mDSC_referred=df.dsc[fl].mean() if fl.any() else np.nan,
             mIoU_approved=df.iou[~fl].mean() if (~fl).any() else np.nan,
             mIoU_referred=df.iou[fl].mean() if fl.any() else np.nan,
             empty_masks=int(df.empty_mask.sum()), empty_masks_approved=int((df.empty_mask.astype(bool) & ~fl).sum()))
    fs = fl | df.empty_mask.values.astype(bool)                   # with the no-detection safeguard
    s.update(referral_rate_with_safeguard=fs.mean(),
             false_safe_rate_with_safeguard=((~fs) & (df.dsc.values < 0.5)).sum() / (~fs).sum() if (~fs).any() else np.nan)
    if fl.any() and (~fl).any():
        s['approved_vs_referred_p'] = stats.mannwhitneyu(df.dsc[~fl], df.dsc[fl]).pvalue
    for t in (0.5, 0.7):
        fail = df.dsc.values < t
        k = str(t).replace('.', '')
        s[f'failures_dsc<{t}'] = int(fail.sum())
        s[f'sensitivity_dsc<{t}'] = (fl & fail).sum() / fail.sum() if fail.any() else np.nan
        s[f'specificity_dsc<{t}'] = (~fl & ~fail).sum() / (~fail).sum() if (~fail).any() else np.nan
        s[f'false_safe_rate_dsc<{t}'] = (~fl & fail).sum() / (~fl).sum() if (~fl).any() else np.nan
        s[f'failure_rate_no_triage_dsc<{t}'] = fail.mean()
    return s


def thresholds_for(model, ck_path):
    """Fixed thresholds, or the mean category shares on the validation images (Eq. 33)."""
    if opt.thresholds == 'fixed':
        if None in (opt.vh, opt.h, opt.u):
            raise SystemExit('ERROR: --thresholds fixed needs --vh, --h and --u.')
        return dict(vh=opt.vh, h=opt.h, u=opt.u, source='fixed'), None
    if not opt.val_image_root or not os.path.isdir(opt.val_image_root):
        raise SystemExit(f'ERROR: validation folder not found: {opt.val_image_root!r}. Set --val_image_root.')
    pairs = pairs_in(opt.val_image_root, opt.val_gt_root or opt.val_image_root)
    if not pairs:
        raise SystemExit('ERROR: no validation images found.')
    v = run_set(model, pairs, 'Validation', thr=None)
    nz = v[v.n_pred_pixels > 0]
    thr = dict(vh=float(nz.very_high_pct.mean()), h=float(nz.high_pct.mean()), u=float(nz.uncertain_pct.mean()),
               source=f'mean category shares on {len(v)} validation images')
    v['flag'] = [flag(r, thr) for r in v.to_dict('records')]
    thr['validation_referral_rate'] = float(v.flag.mean())
    return thr, v


def tag_of(path):
    d = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return d or os.path.splitext(os.path.basename(path))[0]


def main():
    os.makedirs(opt.out_dir, exist_ok=True)
    rows = []
    for ck in opt.models:
        tag = tag_of(ck)
        mdir = os.path.join(opt.out_dir, tag)
        os.makedirs(mdir, exist_ok=True)
        print(f'\n################ Model: {tag} ({ck}) ################')
        model = load_model(ck)
        thr, val_df = thresholds_for(model, ck)
        print(f"Triage thresholds ({thr['source']}): very-high < {thr['vh']:.2f}, high > {thr['h']:.2f}, "
              f"uncertain > {thr['u']:.2f}" + (f" | validation referral rate {thr['validation_referral_rate']:.1%}"
                                                 if 'validation_referral_rate' in thr else ''))
        if val_df is not None:
            val_df.to_csv(os.path.join(mdir, 'per_image_Validation.csv'), index=False)
        info = dict(checkpoint=os.path.abspath(ck), thresholds=thr, min_size=opt.min_size, datasets={})
        if opt.fps:
            info['speed'] = measure_fps(model)
            print(f"Speed: {info['speed']['ms_per_image']:.2f} ms/image ({info['speed']['fps']:.1f} FPS) "
                  f"on {info['speed']['device']}")
        for ds in opt.datasets:
            root = os.path.join(opt.data_root, ds)
            pairs = pairs_in(os.path.join(root, opt.image_dir), os.path.join(root, opt.gt_dir))
            if not pairs:
                print(f'  {ds}: no images found, skipped')
                continue
            sdir = None
            if opt.save_masks:
                sdir = os.path.join(mdir, 'masks', ds)
                os.makedirs(sdir, exist_ok=True)
            t0 = time.time()
            df = run_set(model, pairs, ds, thr, sdir)
            df.to_csv(os.path.join(mdir, f'per_image_{ds}.csv'), index=False)
            s = summarize(df)
            if (df.center.astype(str).str.len() > 0).any():
                s['per_center'] = {c: dict(n=len(g), mDSC=g.dsc.mean(), mIoU=g.iou.mean(), referral_rate=g.flag.mean(),
                                           mDSC_approved=g.dsc[g.flag == 0].mean() if (g.flag == 0).any() else np.nan,
                                           false_safe_rate=((g.flag == 0) & (g.dsc < 0.5)).sum() / max((g.flag == 0).sum(), 1))
                                   for c, g in df[df.center.astype(str).str.len() > 0].groupby('center')}
            info['datasets'][ds] = s
            print(f"  {ds:<14} n={s['n']:<5} mDSC {s['mDSC']:.4f} [{s['mDSC_CI95_low']:.3f}-{s['mDSC_CI95_high']:.3f}] "
                  f"mIoU {s['mIoU']:.4f} | referred {s['n_referred']} ({s['referral_rate']:.1%}) | "
                  f"mDSC approved {s['mDSC_approved']:.3f} vs referred {s['mDSC_referred']:.3f} | "
                  f"false-safe {s['false_safe_rate_dsc<0.5']:.1%}  ({time.time() - t0:.0f}s)")
            flat = {k: v for k, v in s.items() if not isinstance(v, dict)}
            rows.append(dict(model=tag, dataset=ds, thr_vh=thr['vh'], thr_h=thr['h'], thr_u=thr['u'], **flat))
        with open(os.path.join(mdir, 'metrics.json'), 'w') as f:
            json.dump(info, f, indent=2, default=float)

    per_model = pd.DataFrame(rows)
    per_model.to_csv(os.path.join(opt.out_dir, 'summary_per_model.csv'), index=False)

    # mean +/- SD across models (seeds)
    num = per_model.drop(columns=['model']).groupby('dataset', sort=False)
    mean, sd = num.mean(numeric_only=True), num.std(numeric_only=True, ddof=1)
    key = ['mIoU', 'mDSC', 'recall', 'precision', 'F2', 'pooled_F2', 'referral_rate', 'mDSC_approved',
           'mDSC_referred', 'sensitivity_dsc<0.5', 'false_safe_rate_dsc<0.5', 'failure_rate_no_triage_dsc<0.5',
           'referral_rate_with_safeguard', 'false_safe_rate_with_safeguard',
           'thr_vh', 'thr_h', 'thr_u']
    out = pd.DataFrame(index=mean.index)
    for k in key:
        if k in mean:
            out[k] = [f'{m:.4f} ± {s:.4f}' if not np.isnan(s) else f'{m:.4f}' for m, s in zip(mean[k], sd[k])]
    out.insert(0, 'n_models', num.size() // 1)
    out.to_csv(os.path.join(opt.out_dir, 'summary_mean_sd.csv'))
    with open(os.path.join(opt.out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(f'# Results over {len(opt.models)} model(s)\n\nmean ± SD across models; min_size = {opt.min_size}; '
                f'thresholds = {opt.thresholds}\n\n')
        f.write('```\n' + out.to_string() + '\n```\n\n## Per model\n\n```\n'
                + per_model[['model', 'dataset', 'n', 'mIoU', 'mDSC', 'mDSC_CI95_low', 'mDSC_CI95_high', 'recall',
                             'precision', 'F2', 'pooled_F2', 'referral_rate', 'mDSC_approved', 'mDSC_referred',
                             'false_safe_rate_dsc<0.5']].to_string(index=False, float_format=lambda v: f'{v:.4f}')
                + '\n```\n')
    print(f'\nDone. Tables: {os.path.join(opt.out_dir, "summary.md")}')


if __name__ == '__main__':
    main()
