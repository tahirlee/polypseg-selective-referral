"""
Paired comparison of two methods on the same test images (Section 4.5 of the paper):
mean DSC of both, paired difference with a 95% bootstrap confidence interval, and the
Wilcoxon signed-rank test on the per-image DSC.

Input: two per-image CSV files with the columns 'name' and 'dsc'
       (test.py writes them as <out_dir>/<model>/per_image_<dataset>.csv).

Example (CFDA-Net vs. Fu-TransHNet on PolypGen, same training seed):
  python compare_methods.py results/CFDA_seed6645/per_image_PolypGen.csv ^
      results_futranshnet/FuT_seed6645/per_image_PolypGen.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

a = pd.read_csv(sys.argv[1], dtype={'name': str})[['name', 'dsc']]
b = pd.read_csv(sys.argv[2], dtype={'name': str})[['name', 'dsc']]
m = a.merge(b, on='name', suffixes=('_A', '_B'))
if len(m) < len(a):
    print(f'WARNING: only {len(m)} of {len(a)} images matched by name.')
d = (m.dsc_A - m.dsc_B).values
rng = np.random.default_rng(0)
boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
w = stats.wilcoxon(m.dsc_A, m.dsc_B)
print(f'n = {len(m)} paired images')
print(f'A: mDSC {m.dsc_A.mean():.4f}   ({sys.argv[1]})')
print(f'B: mDSC {m.dsc_B.mean():.4f}   ({sys.argv[2]})')
print(f'Difference A - B: {d.mean():+.4f}  (95% CI {np.percentile(boot, 2.5):+.4f} to {np.percentile(boot, 97.5):+.4f})')
print(f'Wilcoxon signed-rank: W = {w.statistic:.1f}, p = {w.pvalue:.2e}')
