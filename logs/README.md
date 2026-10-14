# Logs of the three training runs reported in the paper

Original, unedited console logs (seeds 6645, 1234, 2024; folder tag `CFDA_baseline256` = 256 x 256 input).

| Log | Content |
|---|---|
| `train_seed*.log` | training: 3,960 training / 88 validation images, 2,085,248 parameters, validation DSC per epoch, early stopping (best epochs 72, 21, 49) |
| `test_all.log` | `test_paper.py`: validation-derived thresholds, speed, mDSC with 95% CI, referral rate, mDSC approved vs. referred, false-safe rate (Tables 2–6, 9, 12) |
| `infer_*.log` | `infer_save.py`: per-dataset inference for the calibration analysis |
| `analyze_*.log` | `analyze_calibration.py`: fitted temperature (1.878, 1.604, 1.748) and the thresholds used |

Note: the logs were produced with an earlier version of the scripts. The line "flagged by submitted rule"
in `infer_*.log` refers to fixed thresholds from an earlier version of the method and is not used in the paper;
all triage results in the paper use the validation-derived thresholds shown in `test_all.log` and `analyze_*.log`.
