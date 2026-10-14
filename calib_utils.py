"""
NumPy helpers for the calibration and failure-detection analysis (Sections 4.4.2-4.4.3 of the paper).

Used by:
  infer_save.py          runs the model and computes the per-image quantities
  analyze_calibration.py reads the saved outputs and produces the tables and figures
"""
import numpy as np
import cv2

EPS = 1e-7


# ----------------------------------------------------------------------------
# Basic conversions
# ----------------------------------------------------------------------------
def prob_to_logit(p):
    p = np.clip(p.astype(np.float32), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def logit_to_prob(z, T=1.0):
    return 1.0 / (1.0 + np.exp(-z.astype(np.float32) / T))


def binary_entropy(p):
    """Entropy in bits, in [0, 1]."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# ----------------------------------------------------------------------------
# Post-processing and segmentation metrics (same definitions as test_paper.py, Section 4.3)
# ----------------------------------------------------------------------------
def postprocess(prob, threshold=0.5, min_size=300):
    binary = (prob > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[labels == i] = 1
    return out


def seg_metrics(gt, pred):
    gt = (gt > 0.5).astype(np.float64)
    pred = (pred > 0.5).astype(np.float64)
    tp = (gt * pred).sum()
    fp = ((1 - gt) * pred).sum()
    fn = (gt * (1 - pred)).sum()
    dice = 2 * tp / (gt.sum() + pred.sum() + 1e-5)
    iou = tp / (gt.sum() + pred.sum() - tp + 1e-5)
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    f2 = 5 * precision * recall / (4 * precision + recall + 1e-5)
    return dict(dsc=dice, iou=iou, precision=precision, recall=recall, f2=f2)


# ----------------------------------------------------------------------------
# Case-level confidence features
# ----------------------------------------------------------------------------
def confidence_features(prob, binary_pp, threshold=0.5):
    """
    Case-level features computed from the probability map (before post-processing):
      - confidence descriptors of the reliability assessment module (Section 3.2.2):
        mean and SD of the confidence in the predicted region and the shares of
        uncertain / high / very-high confidence pixels (Eqs. 30-32);
      - entropy-based scores and image-level indicators used for comparison (Section 4.4.3).
    """
    fg = prob > threshold
    n_fg = int(fg.sum())
    f = {}
    f['area_raw'] = n_fg
    f['area_pp'] = int(binary_pp.sum())
    f['empty_raw'] = int(n_fg == 0)
    f['empty_pp'] = int(binary_pp.sum() == 0)
    f['max_prob'] = float(prob.max())
    ent = binary_entropy(prob)
    f['entropy_image'] = float(ent.mean())
    if n_fg > 0:
        c = prob[fg]
        f['mean_conf'] = float(c.mean())
        f['std_conf'] = float(c.std())
        f['min_conf'] = float(c.min())
        f['max_conf'] = float(c.max())
        f['uncertain_pct'] = float(((c >= 0.5) & (c < 0.7)).mean() * 100)
        f['high_pct'] = float(((c >= 0.7) & (c < 0.9)).mean() * 100)
        f['very_high_pct'] = float((c >= 0.9).mean() * 100)
        f['entropy_fg'] = float(ent[fg].mean())
    else:
        # No predicted foreground: worst possible confidence (so these cases rank as most uncertain)
        f.update(mean_conf=0.0, std_conf=0.0, min_conf=0.0, max_conf=0.0,
                 uncertain_pct=0.0, high_pct=0.0, very_high_pct=0.0, entropy_fg=1.0)
    # Entropy in a band around the predicted boundary (boundary ambiguity)
    if n_fg > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg8 = fg.astype(np.uint8)
        band = (cv2.dilate(fg8, k) - cv2.erode(fg8, k)) > 0
        f['entropy_boundary'] = float(ent[band].mean()) if band.any() else 0.0
    else:
        f['entropy_boundary'] = 1.0
    return f



# ----------------------------------------------------------------------------
# Streaming calibration statistics (pixel level)
# ----------------------------------------------------------------------------
class CalibAccumulator:
    """
    Accumulates pixel-level calibration statistics over many images without keeping the pixels.

    Two common definitions are reported:
      * foreground-probability calibration (calibration curve): bins on p = P(polyp),
        compares mean predicted p with the observed fraction of polyp pixels in each bin;
      * top-label calibration: bins on confidence max(p, 1-p) in [0.5, 1],
        compares confidence with pixel accuracy.
    Also Brier score and negative log-likelihood (proper scoring rules).
    """

    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.fg_cnt = np.zeros(n_bins)
        self.fg_sum_p = np.zeros(n_bins)
        self.fg_sum_y = np.zeros(n_bins)
        self.top_cnt = np.zeros(n_bins)
        self.top_sum_c = np.zeros(n_bins)
        self.top_sum_acc = np.zeros(n_bins)
        self.brier = 0.0
        self.nll = 0.0
        self.n = 0
        # fine histogram on the logit scale (resolves the huge mass of pixels near 0 and 1);
        # used for adaptive (equal-mass) ECE and the Brier-score decomposition
        self.f_edges = np.linspace(-16, 16, 2001)
        self.f_cnt = np.zeros(2000)
        self.f_sum_p = np.zeros(2000)
        self.f_sum_y = np.zeros(2000)

    def update(self, p, y):
        p = p.ravel().astype(np.float64)
        y = y.ravel().astype(np.float64)
        b = np.minimum((p * self.n_bins).astype(int), self.n_bins - 1)
        self.fg_cnt += np.bincount(b, minlength=self.n_bins)
        self.fg_sum_p += np.bincount(b, weights=p, minlength=self.n_bins)
        self.fg_sum_y += np.bincount(b, weights=y, minlength=self.n_bins)
        conf = np.maximum(p, 1 - p)
        acc = ((p > 0.5) == (y > 0.5)).astype(np.float64)
        bt = np.minimum(((conf - 0.5) * 2 * self.n_bins).astype(int), self.n_bins - 1)
        self.top_cnt += np.bincount(bt, minlength=self.n_bins)
        self.top_sum_c += np.bincount(bt, weights=conf, minlength=self.n_bins)
        self.top_sum_acc += np.bincount(bt, weights=acc, minlength=self.n_bins)
        self.brier += ((p - y) ** 2).sum()
        pc = np.clip(p, EPS, 1 - EPS)
        self.nll += -(y * np.log(pc) + (1 - y) * np.log(1 - pc)).sum()
        self.n += p.size
        z = np.log(pc) - np.log1p(-pc)
        fb = np.clip(np.searchsorted(self.f_edges, z, side='right') - 1, 0, 1999)
        self.f_cnt += np.bincount(fb, minlength=2000)
        self.f_sum_p += np.bincount(fb, weights=p, minlength=2000)
        self.f_sum_y += np.bincount(fb, weights=y, minlength=2000)

    @staticmethod
    def _ece_mce(cnt, sum_pred, sum_obs):
        nz = cnt > 0
        gap = np.zeros_like(cnt)
        gap[nz] = np.abs(sum_pred[nz] / cnt[nz] - sum_obs[nz] / cnt[nz])
        ece = (cnt * gap).sum() / cnt.sum()
        mce = gap[nz].max() if nz.any() else 0.0
        return ece, mce

    def adaptive_ece(self, n_bins=None):
        """Equal-mass ECE: bins contain (approximately) the same number of pixels.
        Robust to the extreme concentration of segmentation probabilities near 0 and 1."""
        n_bins = n_bins or self.n_bins
        cum = np.cumsum(self.f_cnt)
        group = np.minimum((cum - 0.5 * self.f_cnt) / cum[-1] * n_bins, n_bins - 1).astype(int)
        cnt = np.bincount(group, weights=self.f_cnt, minlength=n_bins)
        sp = np.bincount(group, weights=self.f_sum_p, minlength=n_bins)
        sy = np.bincount(group, weights=self.f_sum_y, minlength=n_bins)
        return self._ece_mce(cnt, sp, sy)[0], int((cnt > 0).sum())

    def brier_decomposition(self):
        """Murphy decomposition: Brier ~ Reliability - Resolution + Uncertainty.
        Reliability (lower = better calibrated); Resolution (higher = better separation of
        polyp and background); Uncertainty = base-rate term, fixed by the data."""
        nz = self.f_cnt > 0
        N = self.f_cnt.sum()
        pb = self.f_sum_p[nz] / self.f_cnt[nz]
        yb = self.f_sum_y[nz] / self.f_cnt[nz]
        ybar = self.f_sum_y.sum() / N
        rel = (self.f_cnt[nz] * (pb - yb) ** 2).sum() / N
        res = (self.f_cnt[nz] * (yb - ybar) ** 2).sum() / N
        unc = ybar * (1 - ybar)
        return rel, res, unc

    def result(self):
        ece_fg, mce_fg = self._ece_mce(self.fg_cnt, self.fg_sum_p, self.fg_sum_y)
        ece_top, mce_top = self._ece_mce(self.top_cnt, self.top_sum_c, self.top_sum_acc)
        aece, n_eff = self.adaptive_ece()
        rel, res, unc = self.brier_decomposition()
        return dict(ECE=ece_fg, ECE_adaptive=aece, MCE=mce_fg, ECE_toplabel=ece_top, MCE_toplabel=mce_top,
                    Brier=self.brier / self.n, Brier_reliability=rel, Brier_resolution=res,
                    Brier_uncertainty=unc, NLL=self.nll / self.n, n_pixels=self.n,
                    adaptive_bins_effective=n_eff)

    def curve(self):
        """Reliability-diagram data for the foreground-probability definition."""
        nz = self.fg_cnt > 0
        edges = np.linspace(0, 1, self.n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        mean_p = np.where(nz, self.fg_sum_p / np.maximum(self.fg_cnt, 1), np.nan)
        freq = np.where(nz, self.fg_sum_y / np.maximum(self.fg_cnt, 1), np.nan)
        return centers, mean_p, freq, self.fg_cnt.copy()


def region_mask(pred, gt, radius=15):
    """Pixels near the polyp: dilated union of prediction and ground truth.
    Excludes the large, trivially easy background that dominates whole-image ECE."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    u = ((pred > 0.5) | (gt > 0.5)).astype(np.uint8)
    return cv2.dilate(u, k) > 0


def expected_dice(p, threshold=0.5):
    """Dice expected from the probability map alone (no ground truth), treating each pixel label
    as Bernoulli(p_i): E[DSC] ~= 2 * sum_{i in M} p_i / (|M| + sum_i p_i), M = {p > threshold}.
    If the probabilities are calibrated, this predicts the per-image DSC."""
    m = p > threshold
    denom = m.sum() + p.sum()
    return float(2 * p[m].sum() / denom) if denom > 0 else 1.0


def dice_raw(p, y, threshold=0.5):
    m = p > threshold
    g = y > 0.5
    d = m.sum() + g.sum()
    return float(2 * (m & g).sum() / d) if d > 0 else 1.0
