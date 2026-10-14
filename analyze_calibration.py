"""
Step 2 of the calibration and failure-detection analysis (no GPU and no PyTorch needed).

Reads the outputs of infer_save.py for one trained model and produces:
  1. pixel-level calibration (ECE, adaptive ECE, MCE, Brier score and its decomposition, NLL) for the
     whole image and the peri-lesional region, before and after temperature scaling
     (T fitted on the validation images only)                          -> Table 10, Figure 9
  2. case-level quality estimation: expected vs. actual Dice           -> Figure 10
  3. failure detection (DSC < 0.5) by case-level scores: AUROC, AUPRC, Spearman correlation,
     risk-coverage curves and AURC, compared with entropy and MC dropout -> Table 11, Figure 11
  4. triage rule (Algorithm 1): approved vs. referred cases, with and without the no-detection safeguard
  5. sensitivity of the triage rule to each threshold                  -> Figure 8
  6. ID vs. OOD discrimination and PolypGen results per center

The triage thresholds must be those derived by test_paper.py for the same model
(metrics.json -> "thresholds"); pass them with --thresholds_json or --vh_thr / --h_thr / --u_thr.

Example:
  python analyze_calibration.py --in_dir ./calibration_outputs/CFDA_seed6645 ^
      --out_dir ./calibration_results/CFDA_seed6645 --thresholds_json ./results/CFDA_seed6645/metrics.json
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, average_precision_score

from calib_utils import (CalibAccumulator, logit_to_prob, binary_entropy, region_mask, EPS,
                         expected_dice, dice_raw)

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, default='./calibration_outputs/')
parser.add_argument('--out_dir', type=str, default='./calibration_results/')
parser.add_argument('--datasets', nargs='+', default=['Kvasir-SEG', 'CVC-ClinicDB', 'BKAI-IGH', 'PolypGen'])
parser.add_argument('--id_dataset', type=str, default='Kvasir-SEG')
parser.add_argument('--fail_thr', type=float, nargs='+', default=[0.5, 0.7], help='DSC below = failure')
parser.add_argument('--n_bins', type=int, default=15)
parser.add_argument('--region_radius', type=int, default=15)
parser.add_argument('--pix_per_image', type=int, default=3000, help='pixels sampled per image for pixel AUROC')
parser.add_argument('--n_boot', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--thresholds_json', type=str, default='',
                    help='metrics.json written by test_paper.py for the same model (contains the thresholds)')
parser.add_argument('--vh_thr', type=float, default=None, help='theta_VH: refer if very-high share is below this')
parser.add_argument('--h_thr', type=float, default=None, help='theta_H: refer if high share is above this')
parser.add_argument('--u_thr', type=float, default=None, help='theta_U: refer if uncertain share is above this')
opt = parser.parse_args()
if opt.thresholds_json:
    import json
    _t = json.load(open(opt.thresholds_json))['thresholds']
    opt.vh_thr, opt.h_thr, opt.u_thr = _t['vh'], _t['h'], _t['u']
if None in (opt.vh_thr, opt.h_thr, opt.u_thr):
    raise SystemExit('Give the triage thresholds: --thresholds_json <metrics.json of test_paper.py> '
                     'or --vh_thr, --h_thr and --u_thr.')

rng = np.random.default_rng(opt.seed)

# Palette (validated categorical order) and text inks
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'savefig.dpi': 300,
                     'legend.frameon': False})

# Case-level uncertainty scores (oriented: higher = less trustworthy)
SCORES = {
    'Mean confidence (1 - p̄)':      lambda d: 1 - d['mean_conf'],
    'Confidence SD (σp)':            lambda d: d['std_conf'],
    'Low-confidence share (100 - very-high %)': lambda d: 100 - d['very_high_pct'],
    'Entropy, predicted region':     lambda d: d['entropy_fg'],
    'Entropy, boundary band':        lambda d: d['entropy_boundary'],
    'Entropy, whole image':          lambda d: d['entropy_image'],
    '1 - max probability':           lambda d: 1 - d['max_prob'],
    'Predicted DSC (1 - E[DSC])':    lambda d: 1 - d['pred_dsc'],
    'MC-dropout variance':           lambda d: d['mc_var_fg'],
    'MC-dropout mutual information': lambda d: d['mc_mi_fg'],
}
MAIN_SCORES = ['Mean confidence (1 - p̄)', 'Confidence SD (σp)', 'Entropy, predicted region', 'MC-dropout variance']


def ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def stems(ds):
    """Only images listed in this dataset's cases.csv (protects against leftover files)."""
    names = pd.read_csv(os.path.join(opt.in_dir, ds, 'cases.csv'), dtype={'name': str})['name']
    return sorted(str(n) for n in names)


def load(ds, stem):
    z = np.load(os.path.join(opt.in_dir, ds, 'logits', stem + '.npy')).astype(np.float32)
    y = cv2.imread(os.path.join(opt.in_dir, ds, 'gt', stem + '.png'), cv2.IMREAD_GRAYSCALE) > 127
    return z, y.astype(np.float32)


# ----------------------------------------------------------------------------
# 1. Temperature scaling on the validation split
# ----------------------------------------------------------------------------
def fit_temperature():
    if not os.path.isfile(os.path.join(opt.in_dir, 'Validation', 'cases.csv')):
        print('No Validation folder: temperature scaling skipped (T = 1).')
        return 1.0
    zs, ys = [], []
    for s in stems('Validation'):
        z, y = load('Validation', s)
        idx = rng.choice(z.size, size=min(z.size, 50000), replace=False)
        zs.append(z.ravel()[idx])
        ys.append(y.ravel()[idx])
    z, y = np.concatenate(zs), np.concatenate(ys)

    def nll(logT):
        p = np.clip(logit_to_prob(z, np.exp(logT)), EPS, 1 - EPS)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    r = minimize_scalar(nll, bounds=(-3, 3), method='bounded')
    T = float(np.exp(r.x))
    print(f'Temperature fitted on validation split ({len(stems("Validation"))} images): T = {T:.3f}')
    return T


# ----------------------------------------------------------------------------
# 2. Pixel-level pass: calibration + pixel error detection
# ----------------------------------------------------------------------------
UEO_T = np.linspace(0.05, 0.95, 19)


def pixel_pass(ds, T):
    acc = {(scope, ts): CalibAccumulator(opt.n_bins) for scope in ('image', 'region') for ts in ('before', 'after')}
    pu, pe = [], []
    case = []
    inter = np.zeros_like(UEO_T)
    u_sum = np.zeros_like(UEO_T)
    e_sum = 0.0
    for s in stems(ds):
        z, y = load(ds, s)
        p = logit_to_prob(z)
        pt = logit_to_prob(z, T)
        reg = region_mask(p, y, opt.region_radius)
        acc[('image', 'before')].update(p, y)
        acc[('image', 'after')].update(pt, y)
        acc[('region', 'before')].update(p[reg], y[reg])
        acc[('region', 'after')].update(pt[reg], y[reg])
        case.append(dict(name=s, dsc_net=dice_raw(p, y), pred_dsc=expected_dice(p), pred_dsc_ts=expected_dice(pt)))
        ent = binary_entropy(p)
        err = (p > 0.5) != (y > 0.5)
        idx = rng.choice(p.size, size=min(p.size, opt.pix_per_image), replace=False)
        pu.append(ent.ravel()[idx])
        pe.append(err.ravel()[idx])
        u = ent[None] > UEO_T[:, None, None]
        inter += (u & err[None]).sum((1, 2))
        u_sum += u.sum((1, 2))
        e_sum += err.sum()
    pu, pe = np.concatenate(pu), np.concatenate(pe)
    pix_auc = roc_auc_score(pe, pu) if 0 < pe.sum() < pe.size else np.nan
    ueo = 2 * inter / (u_sum + e_sum + 1e-8)
    return acc, pix_auc, float(ueo.max()), float(UEO_T[ueo.argmax()]), pd.DataFrame(case)


# ----------------------------------------------------------------------------
# 3. Case-level helpers
# ----------------------------------------------------------------------------
def risk_coverage(score, dsc):
    order = np.argsort(score, kind='stable')          # most confident first
    risk = 1 - dsc[order]
    cov = np.arange(1, len(dsc) + 1) / len(dsc)
    sel_risk = np.cumsum(risk) / np.arange(1, len(dsc) + 1)
    return cov, sel_risk, float(sel_risk.mean())


def oracle_aurc(dsc):
    return risk_coverage(1 - dsc, dsc)[2]


def boot_auc(y, s):
    vals = []
    n = len(y)
    for _ in range(opt.n_boot):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            vals.append(roc_auc_score(y[i], s[i]))
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def case_analysis(df):
    rows = []
    dsc = df['dsc'].values
    for name, fn in SCORES.items():
        try:
            s = fn(df).values.astype(float)
        except KeyError:
            continue
        r = dict(score=name)
        rho, pval = stats.spearmanr(s, dsc)
        r['spearman_rho'] = rho
        r['spearman_p'] = pval
        for thr in opt.fail_thr:
            y = (dsc < thr).astype(int)
            tag = f'DSC<{thr}'
            r[f'n_fail({tag})'] = int(y.sum())
            if 0 < y.sum() < len(y):
                r[f'AUROC({tag})'] = roc_auc_score(y, s)
                r[f'AUPRC({tag})'] = average_precision_score(y, s)
                r[f'prevalence({tag})'] = y.mean()
                if thr == opt.fail_thr[0]:
                    lo, hi = boot_auc(y, s)
                    r[f'AUROC({tag})_CI95'] = f'[{lo:.3f}, {hi:.3f}]'
        _, _, aurc = risk_coverage(s, dsc)
        r['AURC'] = aurc
        r['E-AURC'] = aurc - oracle_aurc(dsc)
        rows.append(r)
    return pd.DataFrame(rows)


def triage_flags(df, vh=None, h=None, u=None, safeguard=False):
    """Triage rule (Eq. 34): refer if very-high% < vh AND high% > h AND uncertain% > u.
    safeguard=True additionally refers cases whose final mask is empty (no-detection safeguard, Algorithm 1)."""
    vh = opt.vh_thr if vh is None else vh
    h = opt.h_thr if h is None else h
    u = opt.u_thr if u is None else u
    has_fg = df['area_raw'].values > 0
    flag = has_fg & (df['very_high_pct'].values < vh) & (df['high_pct'].values > h) & (df['uncertain_pct'].values > u)
    if safeguard:
        flag = flag | (df['empty_pp'].values.astype(bool))
    return flag


def triage_analysis(df, ds, rule='rule', flag=None):
    dsc, iou = df['dsc'].values, df['iou'].values
    flag = triage_flags(df, safeguard=(rule != 'rule')) if flag is None else flag
    r = dict(dataset=ds, rule=rule, n=len(df), n_flagged=int(flag.sum()), referral_rate=flag.mean())
    r['mDSC_all'] = dsc.mean()
    r['mDSC_approved'] = dsc[~flag].mean() if (~flag).any() else np.nan
    r['mDSC_flagged'] = dsc[flag].mean() if flag.any() else np.nan
    r['mIoU_approved'] = iou[~flag].mean() if (~flag).any() else np.nan
    r['mIoU_flagged'] = iou[flag].mean() if flag.any() else np.nan
    if flag.any() and (~flag).any():
        r['MannWhitney_p'] = stats.mannwhitneyu(dsc[~flag], dsc[flag], alternative='two-sided').pvalue
    for thr in opt.fail_thr:
        fail = dsc < thr
        tag = f'DSC<{thr}'
        tp = (flag & fail).sum()
        r[f'failures({tag})'] = int(fail.sum())
        r[f'sensitivity({tag})'] = tp / fail.sum() if fail.any() else np.nan
        r[f'specificity({tag})'] = (~flag & ~fail).sum() / (~fail).sum() if (~fail).any() else np.nan
        r[f'false_safe_rate({tag})'] = (~flag & fail).sum() / (~flag).sum() if (~flag).any() else np.nan
        r[f'missed_failures({tag})'] = int((~flag & fail).sum())
    r['empty_after_postproc'] = int(df['empty_pp'].sum())
    r['empty_after_postproc_auto_approved'] = int((df['empty_pp'].astype(bool) & ~flag).sum())
    r['empty_raw'] = int(df['empty_raw'].sum())
    return r


def threshold_sensitivity(case_dfs, fail_thr):
    """Vary one threshold at a time (others fixed); referral rate and false-safe rate per dataset."""
    grids = {'vh': np.round(np.linspace(opt.vh_thr - 10, min(opt.vh_thr + 10, 100), 21), 2),
             'h': np.round(np.linspace(max(opt.h_thr - 5, 0), opt.h_thr + 5, 21), 2),
             'u': np.round(np.linspace(max(opt.u_thr - 4, 0), opt.u_thr + 4, 21), 2)}
    rows = []
    for ds, df in case_dfs.items():
        dsc = df['dsc'].values
        fail = dsc < fail_thr
        for key, grid in grids.items():
            for v in grid:
                flag = triage_flags(df, **{key: v})
                appr = ~flag
                rows.append(dict(dataset=ds, threshold=key, value=v, referral_rate=flag.mean(),
                                 false_safe_rate=(appr & fail).sum() / max(appr.sum(), 1),
                                 sensitivity=(flag & fail).sum() / max(fail.sum(), 1),
                                 mDSC_approved=dsc[appr].mean() if appr.any() else np.nan))
    return pd.DataFrame(rows)


def fig_threshold_sensitivity(sens, out, fail_thr):
    names = {'vh': 'Very-high share threshold (%)', 'h': 'High share threshold (%)',
             'u': 'Uncertain share threshold (%)'}
    defaults = {'vh': opt.vh_thr, 'h': opt.h_thr, 'u': opt.u_thr}
    dss = list(dict.fromkeys(sens['dataset']))
    styles = ['-', '--', '-.', ':']
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2), sharex='col')
    for j, key in enumerate(('vh', 'h', 'u')):
        for k, ds in enumerate(dss):
            d = sens[(sens.dataset == ds) & (sens.threshold == key)]
            axes[0, j].plot(d['value'], d['referral_rate'] * 100, color=C[k], ls=styles[k % 4], lw=1.6, label=ds)
            axes[1, j].plot(d['value'], d['false_safe_rate'] * 100, color=C[k], ls=styles[k % 4], lw=1.6, label=ds)
        for i in range(2):
            axes[i, j].axvline(defaults[key], color=INK2, lw=0.8, ls=':')
        axes[1, j].set_xlabel(names[key])
    axes[0, 0].set_ylabel('Referral rate (%)')
    axes[1, 0].set_ylabel(f'False-safe rate (%)\n(approved cases with DSC < {fail_thr})')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc='lower center', ncol=len(dss), fontsize=8, bbox_to_anchor=(0.5, -0.04))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, 'fig_threshold_sensitivity.' + ext), bbox_inches='tight')
    plt.close(fig)


def quality_calibration(df, ds, n_bins=10):
    """Case-level calibration: does the Dice expected from the probability map
    (no ground truth) match the actual per-image Dice?"""
    r = dict(dataset=ds)
    for tag, col in (('before', 'pred_dsc'), ('after', 'pred_dsc_ts')):
        e, a_ = df[col].values, df['dsc_net'].values
        b = np.minimum((e * n_bins).astype(int), n_bins - 1)
        cnt = np.bincount(b, minlength=n_bins)
        gap = np.abs(np.bincount(b, weights=e, minlength=n_bins) - np.bincount(b, weights=a_, minlength=n_bins))
        r[f'quality_ECE_{tag}_TS'] = gap.sum() / cnt.sum()
        r[f'MAE_{tag}_TS'] = np.abs(e - a_).mean()
        r[f'mean_predicted_DSC_{tag}_TS'] = e.mean()
    r['mean_actual_DSC'] = df['dsc_net'].mean()
    r['spearman_rho'] = stats.spearmanr(df['pred_dsc'], df['dsc_net'])[0]
    return r


def fig_quality(case_dfs, out, n_bins=10):
    dss = list(case_dfs)
    fig, axes = plt.subplots(1, len(dss), figsize=(2.8 * len(dss), 3.0), squeeze=False)
    for j, ds in enumerate(dss):
        ax = axes[0, j]
        df = case_dfs[ds]
        e, a_ = df['pred_dsc'].values, df['dsc_net'].values
        ax.scatter(e, a_, s=7, color=C[0], alpha=0.35, lw=0, label='Image')
        b = np.minimum((e * n_bins).astype(int), n_bins - 1)
        cnt = np.bincount(b, minlength=n_bins)
        ok = cnt >= 5
        me = np.bincount(b, weights=e, minlength=n_bins)[ok] / cnt[ok]
        ma = np.bincount(b, weights=a_, minlength=n_bins)[ok] / cnt[ok]
        ax.plot(me, ma, color=C[1], lw=2, marker='o', ms=5, label='Bin mean (n >= 5)')
        ax.plot([0, 1], [0, 1], ls='--', color=INK2, lw=1, label='Perfect agreement')
        rho = stats.spearmanr(e, a_)[0]
        ax.set_title(f'{ds}  (ρ = {rho:.2f})', color=INK)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.set_xlabel('Predicted DSC (from probabilities)')
        if j == 0:
            ax.set_ylabel('Actual DSC')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.13))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, 'fig_quality_calibration.' + ext), bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_reliability(curves, results, out):
    dss = list(curves)
    fig, axes = plt.subplots(2, len(dss), figsize=(2.6 * len(dss), 5.4), squeeze=False)
    w = 1.0 / opt.n_bins
    for j, ds in enumerate(dss):
        for i, ts in enumerate(('before', 'after')):
            ax = axes[i, j]
            centers, mean_p, freq, cnt = curves[ds][ts]
            ok = ~np.isnan(freq)
            ax.bar(centers[ok], freq[ok], width=w * 0.92, color=C[0], label='Observed frequency')
            ax.bar(centers[ok], mean_p[ok] - freq[ok], bottom=freq[ok], width=w * 0.92, color='none',
                   edgecolor=C[1], hatch='///', linewidth=0.8, label='Gap')
            ax.plot([0, 1], [0, 1], ls='--', color=INK2, lw=1, label='Perfect calibration')
            res = results[(ds, 'image', ts)]
            ax.text(0.04, 0.96, f"ECE {res['ECE']:.4f}\nBrier {res['Brier']:.4f}", transform=ax.transAxes,
                    va='top', fontsize=8, color=INK)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(ds, color=INK)
            if j == 0:
                ax.set_ylabel(('Before' if ts == 'before' else 'After') + ' temperature scaling\nObserved polyp frequency')
            if i == 1:
                ax.set_xlabel('Predicted polyp probability')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'fig_reliability.{ext}'), bbox_inches='tight')
    plt.close(fig)


def fig_scatter(case_dfs, out, score='Entropy, predicted region'):
    dss = list(case_dfs)
    fig, axes = plt.subplots(1, len(dss), figsize=(2.8 * len(dss), 2.8), squeeze=False)
    for j, ds in enumerate(dss):
        ax = axes[0, j]
        df = case_dfs[ds]
        s = SCORES[score](df).values
        flag = triage_flags(df)
        ax.scatter(s[~flag], df['dsc'].values[~flag], s=9, color=C[0], alpha=0.6, lw=0, label='Auto-approved')
        ax.scatter(s[flag], df['dsc'].values[flag], s=12, color=C[1], alpha=0.8, marker='^', lw=0, label='Referred')
        rho = stats.spearmanr(s, df['dsc'].values)[0]
        ax.set_title(f'{ds}  (ρ = {rho:.2f})', color=INK)
        ax.set_xlabel(score)
        ax.set_ylim(-0.02, 1.02)
        if j == 0:
            ax.set_ylabel('Per-image DSC')
    axes[0, 0].legend(loc='lower left', fontsize=7)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, 'fig_uncertainty_vs_dsc.' + ext), bbox_inches='tight')
    plt.close(fig)


def fig_risk_coverage(case_dfs, out):
    dss = list(case_dfs)
    fig, axes = plt.subplots(1, len(dss), figsize=(2.8 * len(dss), 2.9), squeeze=False)
    styles = ['-', '--', '-.', ':']
    for j, ds in enumerate(dss):
        ax = axes[0, j]
        df = case_dfs[ds]
        dsc = df['dsc'].values
        k = 0
        for name in MAIN_SCORES:
            try:
                s = SCORES[name](df).values
            except KeyError:
                continue
            cov, risk, aurc = risk_coverage(s, dsc)
            ax.plot(cov, risk, color=C[k], ls=styles[k], lw=1.6, label=f'{name}')
            k += 1
        cov, risk, _ = risk_coverage(1 - dsc, dsc)
        ax.plot(cov, risk, color=INK2, lw=1, ls=(0, (1, 1)), label='Oracle')
        ax.axhline((1 - dsc).mean(), color=INK2, lw=0.8, ls='--', alpha=0.6, label='No referral / random')
        flag = triage_flags(df)
        if (~flag).any():
            ax.plot((~flag).mean(), (1 - dsc[~flag]).mean(), marker='o', ms=8, color=INK,
                    mec='white', mew=2, ls='none', label='Triage rule')
        ax.set_title(ds, color=INK)
        ax.set_xlabel('Coverage (fraction auto-approved)')
        ax.set_xlim(0, 1)
        if j == 0:
            ax.set_ylabel('Mean error (1 - DSC) of approved cases')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, 'fig_risk_coverage.' + ext), bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def fmt(df):
    try:
        return df.to_markdown(index=False, floatfmt='.4f')
    except ImportError:                      # 'tabulate' not installed
        return '```\n' + df.to_string(index=False, float_format=lambda v: f'{v:.4f}') + '\n```'


def derive_thresholds():
    """Mean share of very-high / high / uncertain pixels over the validation images with a
    non-empty prediction (Eq. 33); printed as a check against the thresholds in use."""
    p = os.path.join(opt.in_dir, 'Validation', 'cases.csv')
    if not os.path.isfile(p):
        return None
    v = pd.read_csv(p)
    nz = v[v['area_raw'] > 0]
    r = dict(n_validation=len(v), n_with_prediction=len(nz),
             vh_mean_nonempty=nz['very_high_pct'].mean(), h_mean_nonempty=nz['high_pct'].mean(),
             u_mean_nonempty=nz['uncertain_pct'].mean(),
             vh_mean_all=v['very_high_pct'].mean(), h_mean_all=v['high_pct'].mean(),
             u_mean_all=v['uncertain_pct'].mean(),
             vh_used=opt.vh_thr, h_used=opt.h_thr, u_used=opt.u_thr)
    print('Validation-derived thresholds (mean shares over images with a prediction): '
          f"very-high {r['vh_mean_nonempty']:.2f}, high {r['h_mean_nonempty']:.2f}, uncertain {r['u_mean_nonempty']:.2f}")
    print(f'Thresholds used: very-high {opt.vh_thr}, high {opt.h_thr}, uncertain {opt.u_thr}')
    return r


def main():
    out = ensure(opt.out_dir)
    T = fit_temperature()
    thr = derive_thresholds()
    dss = []
    for d in opt.datasets:
        if os.path.isfile(os.path.join(opt.in_dir, d, 'cases.csv')):
            dss.append(d)
        else:
            print(f'WARNING: {d} skipped (no cases.csv: inference missing or unfinished)')
    n_val = len(stems('Validation')) if os.path.isfile(os.path.join(opt.in_dir, 'Validation', 'cases.csv')) else 0
    print(f'Validation images used: {n_val}')

    calib_rows, curves, results, pix_rows = [], {}, {}, []
    case_dfs, case_tables, triage_rows, quality_rows = {}, [], [], []
    for ds in dss:
        print(f'Pixel-level pass: {ds}')
        acc, pix_auc, ueo, ueo_t, qdf = pixel_pass(ds, T)
        curves[ds] = {ts: acc[('image', ts)].curve() for ts in ('before', 'after')}
        for (scope, ts), a in acc.items():
            res = a.result()
            results[(ds, scope, ts)] = res
            calib_rows.append(dict(dataset=ds, region=scope, temperature_scaling=ts, T=T if ts == 'after' else 1.0,
                                   **{k: v for k, v in res.items()}))
        pix_rows.append(dict(dataset=ds, pixel_error_AUROC_entropy=pix_auc, UEO_max=ueo, UEO_threshold=ueo_t))

        df = pd.read_csv(os.path.join(opt.in_dir, ds, 'cases.csv'), dtype={'name': str})
        df = df.merge(qdf, on='name', how='left')
        case_dfs[ds] = df
        quality_rows.append(quality_calibration(df, ds))
        ct = case_analysis(df)
        ct.insert(0, 'dataset', ds)
        case_tables.append(ct)
        triage_rows.append(triage_analysis(df, ds, 'rule'))
        triage_rows.append(triage_analysis(df, ds, 'rule + no-detection safeguard'))

    calib = pd.DataFrame(calib_rows)
    case = pd.concat(case_tables, ignore_index=True)
    triage = pd.DataFrame(triage_rows)
    pix = pd.DataFrame(pix_rows)

    sens = threshold_sensitivity(case_dfs, opt.fail_thr[0])
    sens.to_csv(os.path.join(out, 'table_threshold_sensitivity.csv'), index=False)
    fig_threshold_sensitivity(sens, out, opt.fail_thr[0])

    # ID vs OOD discrimination
    ood_rows = []
    if opt.id_dataset in case_dfs:
        id_df = case_dfs[opt.id_dataset]
        for ds in dss:
            if ds == opt.id_dataset:
                continue
            for name, fn in SCORES.items():
                try:
                    s = np.r_[fn(id_df).values, fn(case_dfs[ds]).values]
                except KeyError:
                    continue
                y = np.r_[np.zeros(len(id_df)), np.ones(len(case_dfs[ds]))]
                ood_rows.append(dict(id=opt.id_dataset, ood=ds, score=name, AUROC=roc_auc_score(y, s)))
    ood = pd.DataFrame(ood_rows)

    # PolypGen per center
    center = None
    for ds in dss:
        df = case_dfs[ds]
        if 'center' in df and df['center'].notna().any() and (df['center'].astype(str).str.len() > 0).any():
            df = df.assign(flag_rule=triage_flags(df))
            g = df[df['center'].astype(str).str.len() > 0].groupby('center')
            center = g.agg(n=('dsc', 'size'), mDSC=('dsc', 'mean'), mIoU=('iou', 'mean'),
                           referral_rate=('flag_rule', 'mean'),
                           failure_rate_DSC_lt_05=('dsc', lambda x: (x < 0.5).mean())).reset_index()
            center.insert(0, 'dataset', ds)

    # Save tables
    calib.to_csv(os.path.join(out, 'table_calibration.csv'), index=False)
    case.to_csv(os.path.join(out, 'table_uncertainty_vs_error.csv'), index=False)
    triage.to_csv(os.path.join(out, 'table_triage_approved_vs_flagged.csv'), index=False)
    pix.to_csv(os.path.join(out, 'table_pixel_uncertainty_error.csv'), index=False)
    if len(ood):
        ood.to_csv(os.path.join(out, 'table_id_vs_ood.csv'), index=False)
    if center is not None:
        center.to_csv(os.path.join(out, 'table_polypgen_per_center.csv'), index=False)

    # Figures
    fig_reliability(curves, results, out)
    fig_quality(case_dfs, out)
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(os.path.join(out, 'table_case_quality_calibration.csv'), index=False)
    fig_scatter(case_dfs, out)
    fig_risk_coverage(case_dfs, out)

    # One readable summary
    with open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(f'# Calibration and uncertainty analysis\n\nTemperature (fitted on validation split): T = {T:.4f}\n\n')
        f.write('## Pixel-level calibration\n\n' + fmt(calib) + '\n\n')
        f.write('## Case-level quality calibration (predicted vs actual DSC)\n\n' + fmt(quality) + '\n\n')
        f.write('## Uncertainty vs. segmentation error (case level)\n\n' + fmt(case) + '\n\n')
        f.write('## Triage rule: auto-approved vs. referred\n\n' + fmt(triage) + '\n\n')
        if thr:
            f.write('## Threshold derivation on the validation split\n\n' + fmt(pd.DataFrame([thr])) + '\n\n')
        f.write(f'Rule thresholds: very-high < {opt.vh_thr}, high > {opt.h_thr}, uncertain > {opt.u_thr}\n\n')
        f.write('## Threshold sensitivity (one threshold varied at a time)\n\n'
                + fmt(sens[sens['value'].isin(sens.groupby('threshold')['value'].apply(lambda v: v.iloc[::5]).explode().values)]) + '\n\n')
        f.write('## Pixel-level uncertainty vs. error\n\n' + fmt(pix) + '\n\n')
        if len(ood):
            f.write('## ID vs. OOD discrimination (AUROC)\n\n' + fmt(ood) + '\n\n')
        if center is not None:
            f.write('## PolypGen per center\n\n' + fmt(center) + '\n\n')
    print(f'\nDone. Results in {out}  (summary.md, tables and figures)')


if __name__ == '__main__':
    main()
