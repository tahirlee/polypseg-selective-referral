# Polyp Segmentation (will update after paper acceptance)

Code for the paper *"After paper acceptance"*.



## Repository

| File | Purpose | Paper |
|---|---|---|
| `model/cfda_net.py` | CFDA-Net (MobileNetV2 encoder, DFA, MSA, EAA, decoder with SN, MixStyle, AdaIN, DropBlock) and ablation switches | Sec. 3.2.1, Table 8 |
| `train.py` | training | Sec. 4.3 |
| `test.py` | segmentation metrics, bootstrap CIs, triage thresholds and triage results, PolypGen per center, speed | Tables 2–7, 9, 12 |
| `infer_save.py` + `analyze_calibration.py` + `calib_utils.py` | calibration, temperature scaling, failure detection, risk–coverage, threshold sensitivity | Tables 10–11, Figs. 8–11 |
| `compare_methods.py` | paired Wilcoxon test between two methods | Sec. 4.5 |
| `check_data.py` | checks image/mask pairs and train/validation/test leakage | Sec. 4.1 |
| `model/baseline_futranshnet.py` | wrapper to retrain Fu-TransHNet under the same protocol | Sec. 4.5 |

## Installation

```
pip install -r requirements.txt
```
The paper used Python 3.9.25, PyTorch 2.6.0, torchvision 0.21.0, CUDA 11.8 and an NVIDIA GeForce RTX 3060 (12 GB).

## Data

```
dataset/
  train/Kvasir-SEG/images, masks_binary          792 training images, offline-augmented to 3,960
  validation/Kvasir-SEG/images, masks_binary     88 validation images (not augmented)
  test/Kvasir-SEG/images, masks_binary           120 test images
  test/CVC-ClinicDB/images, masks_binary
  test/BKAI-IGH/images, masks_binary
  test/PolypGen/images, masks_binary
  test/data_C1 ... data_C6/images, masks_binary  PolypGen per center (optional)
```
The datasets are available from their providers: Kvasir-SEG, CVC-ClinicDB, BKAI-IGH (NeoPolyp) and PolypGen.
Masks are binary (0/255). Run `python check_data.py` before training.

## Training (three runs, Section 4.3)

Input 256 × 256, Adam (lr 1e-4, β = 0.9/0.999), batch size 8, cosine annealing, at most 500 epochs,
early stopping on validation DSC (patience 20), BCE + Dice loss, ImageNet-pretrained MobileNetV2.
The benchmark of Jha et al. (2024), from which the results of the eight compared methods are taken,
used an early-stopping patience of 50 epochs; all other main settings are the same (Section 4.5).
```
python train.py --seed 6645 --save_dir ./trained_models/CFDA_seed6645/
python train.py --seed 1234 --save_dir ./trained_models/CFDA_seed1234/
python train.py --seed 2024 --save_dir ./trained_models/CFDA_seed2024/
```

## Evaluation and triage (Tables 2–7, 9, 12)

```
python test.py --models ./trained_models/CFDA_seed6645/best_model.pth ./trained_models/CFDA_seed1234/best_model.pth ./trained_models/CFDA_seed2024/best_model.pth
```
For each model, the three triage thresholds are derived on its own 88 validation images (Eq. 33)
before any test image is used. Results are written to `./results/` (`summary.md`: mean ± SD over the runs).
PolypGen per center: add `--datasets data_C1 data_C2 data_C3 data_C4 data_C5 data_C6`.

## Calibration and failure detection (Tables 10–11, Figures 8–11)

```
python infer_save.py --checkpoint ./trained_models/CFDA_seed6645/best_model.pth --out_dir ./calibration_outputs/CFDA_seed6645
python analyze_calibration.py --in_dir ./calibration_outputs/CFDA_seed6645 --out_dir ./calibration_results/CFDA_seed6645 --thresholds_json ./results/CFDA_seed6645/metrics.json
```

## Trained models and results of the paper

The three trained models and all outputs reported in the paper are included:

| Folder | Content |
|---|---|
| `trained_models/revision/CFDA_baseline256_seed{6645,1234,2024}/` | `best_model.pth`, `config.json` (all settings and library versions), `history.csv` (per-epoch validation DSC), `split_train.txt`, `split_val.txt` |
| `results_revision/` | output of `test.py`: `summary.md` / `summary_mean_sd.csv` (mean ± SD, Tables 2–6 and 9), `summary_per_model.csv` (per run, Supplementary Tables S1–S2), and per run `metrics.json` and `per_image_<dataset>.csv` |
| `calibration_results/CFDA_baseline256_seed*/` | output of `analyze_calibration.py` (Tables 10–11, Figures 8–11) |
| `logs/` | original console logs of training, testing and the calibration analysis |

The tag `CFDA_baseline256` denotes CFDA-Net at 256 × 256 input. Rates are fractions (0.228 = 22.8%).
The calibration outputs were generated with an earlier version of `analyze_calibration.py`, in which the triage
rule was labelled "submitted"; it is the same rule (Eq. 34) with the validation-derived thresholds of each model.

To evaluate the provided models:
```
python test.py --models ./trained_models/revision/CFDA_baseline256_seed6645/best_model.pth ./trained_models/revision/CFDA_baseline256_seed1234/best_model.pth ./trained_models/revision/CFDA_baseline256_seed2024/best_model.pth
```

## Ablation (Table 8)

`model/cfda_net.py` provides `ablation_config('add' | 'drop', component)` for the additive and
leave-one-out variants. `python model/cfda_net.py` prints the parameters and GMACs of every variant.

## Fu-TransHNet baseline

Place the authors' `lib/` folder (https://github.com/ChenxiLuo-code1/Fu-TransHNet) in the project folder
and the ImageNet weights in `./pretrained/` (DeiT-S `deit_small_patch16_224-cd65a155.pth`, `hardnet68.pth`). Then:
```
python train.py --arch futranshnet --seed 6645 --save_dir ./trained_models/FuT_seed6645/
python test.py --arch futranshnet --models ./trained_models/FuT_seed6645/best_model.pth --out_dir ./results_futranshnet/
```
The Fu-TransHNet models reported in the paper were trained on the 792 original training images with
on-the-fly augmentation only (no offline expansion). To reproduce them, set `--image_root` and `--gt_root`
to a folder that contains only the 792 original images and masks (seeds 6645, 1234 and 2024).

## Citation

Citation details will be added after publication.

## License

The code is released under the MIT License (see `LICENSE`). The trained models were trained on Kvasir-SEG and are
provided for research use only. The datasets are not included and remain subject to the licenses of their providers.
The Fu-TransHNet code (`lib/`) is not included and is subject to its authors' terms.
This software is intended for research only and is not a medical device.
