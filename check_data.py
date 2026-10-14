"""
Check the dataset folders before training (image/mask pairing and train-validation-test leakage).

Usage:  python check_data.py
Prints: number of images / masks per folder, images without a mask (and vice versa),
        files that appear in both train and validation, and validation images whose
        name is contained in a training file name (e.g. offline-augmented copies such as
        '<name>_aug3.png'), which would indicate leakage.
"""
import os
from collections import Counter

ROOTS = {
    'train': ('./dataset/train/Kvasir-SEG/images/', './dataset/train/Kvasir-SEG/masks_binary/'),
    'validation': ('./dataset/validation/Kvasir-SEG/images/', './dataset/validation/Kvasir-SEG/masks_binary/'),
    'test': ('./dataset/test/Kvasir-SEG/images/', './dataset/test/Kvasir-SEG/masks_binary/'),
}
EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def stems(folder):
    if not os.path.isdir(folder):
        return None
    return {os.path.splitext(f)[0]: f for f in os.listdir(folder) if f.lower().endswith(EXT)}


sets = {}
for name, (img_dir, gt_dir) in ROOTS.items():
    imgs, gts = stems(img_dir), stems(gt_dir)
    if imgs is None or gts is None:
        print(f'[{name}] folder missing: {img_dir if imgs is None else gt_dir}')
        continue
    paired = set(imgs) & set(gts)
    only_img, only_gt = sorted(set(imgs) - set(gts)), sorted(set(gts) - set(imgs))
    sets[name] = paired
    print(f'\n[{name}] images {len(imgs)} | masks {len(gts)} | paired {len(paired)}')
    print(f'   images without mask: {len(only_img)}  e.g. {only_img[:5]}')
    print(f'   masks without image: {len(only_gt)}  e.g. {only_gt[:5]}')
    exts = Counter(os.path.splitext(f)[1].lower() for f in imgs.values())
    print(f'   image extensions: {dict(exts)}')

if 'train' in sets and 'validation' in sets:
    both = sorted(sets['train'] & sets['validation'])
    print(f'\nSame file name in train AND validation: {len(both)}  e.g. {both[:10]}')
    # augmented copies: a validation name contained in a longer training name
    tr = list(sets['train'])
    leak = sorted({v for v in sets['validation'] for t in tr if v != t and v in t})
    print(f'Validation images with (augmented?) copies in train: {len(leak)}  e.g. {leak[:10]}')

if 'train' in sets and 'test' in sets:
    both = sorted(sets['train'] & sets['test'])
    tr = list(sets['train'])
    leak = sorted({v for v in sets['test'] for t in tr if v != t and v in t})
    print(f'\nSame file name in train AND test: {len(both)}  e.g. {both[:10]}')
    print(f'Test images with (augmented?) copies in train: {len(leak)}  e.g. {leak[:10]}')
