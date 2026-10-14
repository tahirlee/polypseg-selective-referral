"""
Training of CFDA-Net (Section 4.3 of the paper).

Protocol (all defaults below equal the values reported in the paper)
--------------------------------------------------------------------
Training data   : Kvasir-SEG only. 792 training images, expanded offline to 3,960 images
                  (five versions per image), in --image_root.
Validation data : 88 Kvasir-SEG images held out from the 880-image training split, in --val_image_root
                  (not augmented; used for early stopping and, at test time, for the triage thresholds).
                  The 120-image Kvasir-SEG test set and the target datasets are never read by this script.
Input size      : 256 x 256
On-the-fly aug. : horizontal and vertical flip (p = 0.5 each), rotation (+/-20 deg), scaling (0.8-1.2),
                  brightness, contrast and saturation factors (0.8-1.2), hue shift (+/-0.05)
                  (MixStyle and DropBlock are part of the network and active only in training mode)
Loss            : L = L_BCE + L_Dice (Eqs. 35-37)
Optimizer       : Adam, lr = 1e-4, beta1 = 0.9, beta2 = 0.999, batch size 8
Schedule        : cosine annealing, at most 500 epochs
Early stopping  : validation DSC, patience 20 epochs; the weights with the highest validation DSC are kept
Seeds           : 6645, 1234 and 2024 (three independent runs)

Example (the three runs of the paper)
-------------------------------------
  python train_paper.py --seed 6645 --save_dir ./trained_models/CFDA_seed6645/
  python train_paper.py --seed 1234 --save_dir ./trained_models/CFDA_seed1234/
  python train_paper.py --seed 2024 --save_dir ./trained_models/CFDA_seed2024/

Outputs (in --save_dir)
-----------------------
  best_model.pth                    weights with the highest validation DSC
  split_train.txt / split_val.txt   file names of the training and validation images
  config.json                       all hyper-parameters, seed, library and GPU versions
  history.csv                       per-epoch learning rate, losses, validation DSC and IoU
"""

import argparse
import csv
import json
import os
import platform
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model.cfda_net import CFDANet

# ----------------------------------------------------------------------------
# Arguments (defaults = values reported in the paper)
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--image_root', type=str, default='./dataset/train/Kvasir-SEG/images/',
                    help='training images (3,960 = 792 originals expanded offline)')
parser.add_argument('--gt_root', type=str, default='./dataset/train/Kvasir-SEG/masks_binary/')
parser.add_argument('--val_image_root', type=str, default='./dataset/validation/Kvasir-SEG/images/',
                    help='the 88 validation images')
parser.add_argument('--val_gt_root', type=str, default='./dataset/validation/Kvasir-SEG/masks_binary/')
parser.add_argument('--save_dir', type=str, default='./trained_models/CFDA_seed6645/')
parser.add_argument('--epoch', type=int, default=500, help='maximum number of epochs')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--trainsize', type=int, default=256)
parser.add_argument('--patience', type=int, default=20, help='early stopping patience (epochs, validation DSC)')
parser.add_argument('--seed', type=int, default=6645, help='paper: 6645, 1234 and 2024')
parser.add_argument('--num_workers', type=int, default=0, help='0 is safest on Windows')
parser.add_argument('--pretrained', type=int, default=1, help='ImageNet-pretrained MobileNetV2 encoder')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--arch', type=str, default='cfda', choices=['cfda', 'futranshnet'],
                    help="'cfda' = CFDA-Net (proposed); 'futranshnet' = Fu-TransHNet baseline (Luo et al., 2026)")
parser.add_argument('--deit_ckpt', type=str, default='./pretrained/deit_small_patch16_224-cd65a155.pth',
                    help='ImageNet DeiT-S weights (only for --arch futranshnet)')
parser.add_argument('--hardnet_ckpt', type=str, default='./pretrained/hardnet68.pth',
                    help='ImageNet HarDNet68 weights (only for --arch futranshnet)')
opt = parser.parse_args()


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
IMG_EXT = ('.jpg', '.jpeg', '.png')


def list_pairs(image_root, gt_root):
    """Pairs every image with the mask of the same file name."""
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(image_root) if f.lower().endswith(IMG_EXT)}
    gts = {os.path.splitext(f)[0]: f for f in os.listdir(gt_root) if f.lower().endswith(IMG_EXT)}
    stems = sorted(set(imgs) & set(gts))
    missing = set(imgs) ^ set(gts)
    if missing:
        print(f'WARNING: {len(missing)} images without a matching mask (or vice versa) were skipped.')
    return [(os.path.join(image_root, imgs[s]), os.path.join(gt_root, gts[s])) for s in stems]


class PolypDataset(Dataset):
    def __init__(self, pairs, size, train):
        self.pairs, self.size, self.train = pairs, size, train
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def augment(img, gt):
        """On-the-fly augmentation of Section 4.3."""
        # geometric, applied jointly to image and mask
        if random.random() < 0.5:
            img, gt = TF.hflip(img), TF.hflip(gt)
        if random.random() < 0.5:
            img, gt = TF.vflip(img), TF.vflip(gt)
        angle = random.uniform(-20.0, 20.0)
        scale = random.uniform(0.8, 1.2)
        img = TF.affine(img, angle=angle, translate=[0, 0], scale=scale, shear=[0.0],
                        interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        gt = TF.affine(gt, angle=angle, translate=[0, 0], scale=scale, shear=[0.0],
                       interpolation=TF.InterpolationMode.NEAREST, fill=0)
        # photometric, image only
        img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
        img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
        img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
        img = TF.adjust_hue(img, random.uniform(-0.05, 0.05))
        return img, gt

    def __getitem__(self, i):
        img_path, gt_path = self.pairs[i]
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        img = TF.resize(img, [self.size, self.size], interpolation=TF.InterpolationMode.BILINEAR)
        gt = TF.resize(gt, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)
        if self.train:
            img, gt = self.augment(img, gt)
        img = TF.normalize(TF.to_tensor(img), self.mean, self.std)
        gt = (TF.to_tensor(gt) > 0.5).float()
        return img, gt


# ----------------------------------------------------------------------------
# Loss: L = L_BCE + L_Dice (Eqs. 35-37)
# ----------------------------------------------------------------------------
class BCEDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, prob, target):
        # both networks end with a sigmoid, so the output is already a probability.
        # A list of side outputs (deep supervision of the comparison model) gives the sum of their losses.
        if isinstance(prob, (list, tuple)):
            return sum(self.forward(p, target) for p in prob)
        if prob.shape[-2:] != target.shape[-2:]:
            prob = F.interpolate(prob, size=target.shape[-2:], mode='bilinear', align_corners=False)
        prob = prob.clamp(1e-7, 1 - 1e-7)
        bce = F.binary_cross_entropy(prob, target)
        p, t = prob.flatten(1), target.flatten(1)
        dice = (1 - (2 * (p * t).sum(1) + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)).mean()
        return bce + dice


# ----------------------------------------------------------------------------
# Train / validate
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total, t0 = 0.0, time.time()
    for b, (images, gts) in enumerate(loader, 1):
        images, gts = images.to(device), gts.to(device)
        optimizer.zero_grad()
        out = model(images)
        # multi-view models (Fu-TransHNet) weight the losses of their views themselves
        loss = model.view_loss(out, gts, criterion) if hasattr(model, 'view_loss') else criterion(out, gts)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
        if b % 50 == 0 or b == len(loader):
            print(f'    epoch {epoch:03d}  batch {b:4d}/{len(loader)}  loss {loss.item():.4f}  '
                  f'({time.time() - t0:.0f}s)', flush=True)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validation loss and per-image mean DSC / IoU at threshold 0.5."""
    model.eval()
    total, dscs, ious = 0.0, [], []
    for images, gts in loader:
        images, gts = images.to(device), gts.to(device)
        prob = model(images)
        total += criterion(prob, gts).item() * images.size(0)
        pred, t = (prob > 0.5).float().flatten(1), gts.flatten(1)
        inter = (pred * t).sum(1)
        dscs += ((2 * inter) / (pred.sum(1) + t.sum(1) + 1e-8)).cpu().tolist()
        ious += (inter / (pred.sum(1) + t.sum(1) - inter + 1e-8)).cpu().tolist()
    return total / len(loader.dataset), float(np.mean(dscs)), float(np.mean(ious))


def main():
    set_seed(opt.seed)
    os.makedirs(opt.save_dir, exist_ok=True)
    device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() else 'cpu')

    # ---- data ----
    train_pairs = list_pairs(opt.image_root, opt.gt_root)
    val_pairs = list_pairs(opt.val_image_root, opt.val_gt_root)
    overlap = {os.path.basename(p[0]) for p in train_pairs} & {os.path.basename(p[0]) for p in val_pairs}
    if overlap:
        raise SystemExit(f'ERROR: {len(overlap)} images are in BOTH the training and the validation folder.')
    if len(train_pairs) != 3960 or len(val_pairs) != 88:
        print(f'NOTE: the paper uses 3,960 training and 88 validation images; '
              f'found {len(train_pairs)} and {len(val_pairs)}.')
    for name, subset in [('split_train.txt', train_pairs), ('split_val.txt', val_pairs)]:
        with open(os.path.join(opt.save_dir, name), 'w') as f:
            f.writelines(os.path.basename(p[0]) + '\n' for p in subset)
    print(f'Train: {len(train_pairs)} images | Validation: {len(val_pairs)} images')

    g = torch.Generator()
    g.manual_seed(opt.seed)
    train_loader = DataLoader(PolypDataset(train_pairs, opt.trainsize, train=True),
                              batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers,
                              worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(PolypDataset(val_pairs, opt.trainsize, train=False),
                            batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers)

    # ---- model, loss, optimizer, schedule ----
    if opt.arch == 'futranshnet':
        from model.baseline_futranshnet import FuTransHNet
        for f in (opt.deit_ckpt, opt.hardnet_ckpt):
            if opt.pretrained and not os.path.isfile(f):
                raise SystemExit(f'ERROR: ImageNet weights not found: {f}')
        model = FuTransHNet(img_size=opt.trainsize, deit_ckpt=opt.deit_ckpt if opt.pretrained else None,
                            hardnet_ckpt=opt.hardnet_ckpt if opt.pretrained else None).to(device)
    else:
        model = CFDANet(num_channels=3, num_classes=1, pretrained=bool(opt.pretrained)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params:,}')
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch)

    # ---- record configuration ----
    config = vars(opt).copy()
    config.update({
        'n_train': len(train_pairs), 'n_val': len(val_pairs), 'n_params': n_params,
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'torchvision': str(torchvision.__version__), 'cuda': str(torch.version.cuda),
        'cudnn': str(torch.backends.cudnn.version()),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'started': datetime.now().isoformat(timespec='seconds'),
    })
    with open(os.path.join(opt.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ---- training with early stopping on validation DSC ----
    hist_path = os.path.join(opt.save_dir, 'history.csv')
    with open(hist_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'val_dsc', 'val_iou', 'time_s'])

    best_dsc, best_epoch, wait = -1.0, 0, 0
    for epoch in range(1, opt.epoch + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_dsc, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0
        with open(hist_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{lr:.3e}', f'{train_loss:.5f}', f'{val_loss:.5f}',
                                    f'{val_dsc:.5f}', f'{val_iou:.5f}', f'{dt:.1f}'])
        msg = (f'[{datetime.now():%H:%M:%S}] Epoch {epoch:03d}/{opt.epoch} | lr {lr:.2e} | '
               f'train {train_loss:.4f} | val loss {val_loss:.4f} | val DSC {val_dsc:.4f} | '
               f'val IoU {val_iou:.4f} | {dt:.0f}s')
        if val_dsc > best_dsc:
            best_dsc, best_epoch, wait = val_dsc, epoch, 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_dsc': val_dsc, 'val_iou': val_iou, 'config': config},
                       os.path.join(opt.save_dir, 'best_model.pth'))
            msg += '  <- best (saved)'
        else:
            wait += 1
            msg += f'  (no improvement {wait}/{opt.patience})'
        print(msg, flush=True)
        if wait >= opt.patience:
            print(f'Early stopping at epoch {epoch}.')
            break

    print(f'Best validation DSC {best_dsc:.4f} at epoch {best_epoch}. '
          f'Model: {os.path.join(opt.save_dir, "best_model.pth")}')
    config.update({'best_epoch': best_epoch, 'best_val_dsc': best_dsc,
                   'finished': datetime.now().isoformat(timespec='seconds')})
    with open(os.path.join(opt.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    main()
