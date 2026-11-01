# EDSR paper-size / SEV2MOD dual-band x12

This adaptation trains the paper-size EDSR architecture from random initialization:
32 residual blocks, 256 feature channels, residual scaling 0.1, two input/output
bands, and a jointly trained x2 -> x2 -> x3 PixelShuffle head. No RGB checkpoint
is downloaded or loaded. These are architectural settings, not a claim to
reproduce the original paper's dataset, augmentation, or published scores.

The original RGB entry point remains `src/main.py`. Use the separate
`src/train_sev2mod.py` entry point for this task. It does not call the RGB
image loader, byte quantizer, luminance metrics, or PNG result writer.

## Data contract

Point `--data-root` to the directory containing `train`, `val`, `test_16`,
`test_32`, and `test_64`. Each split contains `LR/<id>.npy` and `HR/<id>.npy`
with identical stems. Arrays are real finite CHW values, cast to float32.

- train / val / test_16: (2,16,16) -> (2,192,192).
- test_32: (2,32,32) -> (2,384,384).
- test_64: (2,64,64) -> (2,768,768).

No synthetic LR, RGB conversion, band duplication, PAN input, PNG preprocessing,
random crop or augmentation is used. A training item is the complete 16x16 LR
and its paired 192x192 HR. Missing pairs, wrong layouts, NaN/Inf and float32
overflow fail explicitly. Spatial registration and band physical interpretation
must be established by the dataset provider; shape checks do not establish them.

Normalization is fit once by scanning **all train LR and HR** arrays. Each band
uses shared LR/HR min and max. Validation/test reuse those bounds, never fit
their own bounds and are never clipped. A constant band raises an error.
The complete bounds and sample counts are stored in config.json and checkpoints.
Initial normalization reads the complete training split and can take some time.

## Train

Activate the `edsr` environment, enter this repository, then run:

```powershell
python src/train_sev2mod.py train --data-root "C:/Users/10115/Datasets/SEV2MOD_X12/SEV2MOD_X12" --output runs/edsr-paper-2band-x12 --epochs 100 --batch-size 4 --workers 0 --amp
```

On the rented Linux server, replace the data path and activate the matching
environment. The local environment used Python 3.10, torch 2.5.1+cu121,
NumPy 1.26.4 and scikit-image 0.24.0. `--device cpu` supports debugging;
`--amp` requires CUDA. Batch size 4 is a starting setting, not a tested optimum
for the server's 4090. Adjust after measuring memory and throughput there.

Training uses L1, Adam (lr 1e-4, betas 0.9/0.999), and StepLR (halve after
every 200 epochs). With 100 epochs no scheduled reduction occurs. The epoch
count is configurable and is not a guaranteed convergence budget.

Each epoch evaluates **val only**, in float32. `best.pt` is selected by minimum
mean normalized per-band validation MSE. Held-out test splits are never used
for model selection. `history.jsonl` records training seconds and throughput;
warmup, batch size and hardware affect these numbers.

Outputs include config.json, history.jsonl, latest.pt and best.pt. Each checkpoint
contains model, normalization, optimizer, scheduler, AMP scaler, epoch, best
validation error, and Python/NumPy/PyTorch/CUDA/shuffle-generator RNG states.
Checkpoint writes use a temporary file and atomic replacement. A nonempty output
directory is refused for fresh training; original data and output may not overlap.

## Resume

```powershell
python src/train_sev2mod.py train --data-root "C:/Users/10115/Datasets/SEV2MOD_X12/SEV2MOD_X12" --output runs/edsr-paper-2band-x12 --epochs 200 --batch-size 4 --workers 0 --amp --resume runs/edsr-paper-2band-x12/latest.pt
```

`--epochs` is the total target epoch, not an additional count. Resume must use
the original output directory, data root, batch size, worker count and AMP mode. Optimizer,
scheduler, normalization and RNG state come from the checkpoint. For closest
repeatability keep hardware/software and worker count unchanged. No claim of
bitwise identity across CUDA devices is made. Only load trusted checkpoints:
they include Python/NumPy training state and use PyTorch pickle deserialization.

## Held-out evaluation and output

```powershell
python src/train_sev2mod.py evaluate --data-root "C:/Users/10115/Datasets/SEV2MOD_X12/SEV2MOD_X12" --checkpoint runs/edsr-paper-2band-x12/best.pt --split test_16 --output runs/edsr-eval-test16 --save-predictions
```

Repeat with separate output directories for test_32 and test_64. Evaluation uses
batch size 1 and float32. No quantization, clipping, RGB/Y conversion, or border
crop precedes metrics. Report both bands separately:

- PSNR from aggregate pixel MSE, normalized data_range=1 (the train-fitted span).
- SSIM with data_range=1, averaged over images.
- MAE/RMSE in the original array's numeric units; those units are not inferred.

Perfect reconstruction has null PSNR plus `perfect_reconstruction: true`, meaning
+infinity, so JSON remains standards compliant. `metrics.json` includes the metric
protocol and checkpoint epoch. Optional predictions are inverse-normalized
two-band float32 NPY, preserving out-of-range predictions rather than clipping.
No RGB PNG is substituted for the numeric result.

## Tests by feature

- tests/model/: x12 output/gradients, paper configuration, legacy upsamplers,
  missing/incompatible checkpoint rejection.
- tests/data/: train-only bounds, native CHW and test sizes, missing pairs,
  bad layout and nonfinite data rejection.
- tests/evaluation/: analytic per-band metrics, sub-byte error precision,
  perfect reconstruction, original-unit float32 NPY export.
- tests/training/: data-to-optimizer integration, checkpoint roundtrip and
  optional full-size CUDA command-line train/resume/evaluate integration.

```powershell
python -B -m unittest discover -s tests -t . -v
python -B -m tests.training.smoke_cli
```

The CUDA integration test uses one temporary synthetic pair per split with the
full 32-block/256-channel network. It runs an AMP training epoch, resumes to the
second epoch, evaluates the validation-selected checkpoint and checks NPY export.
It does not train on the real dataset or establish convergence/quality. It was
run locally on the available 4060 Laptop; the rented 4090 has not been benchmarked.

## Weight compatibility change

EDSR now honors `shift_mean=False`; RGB mean layers remain available for the old
three-channel path. The state-dict loader no longer silently skips incompatible
tail tensor shapes. The new entry point checks the complete model configuration
and strictly loads its own checkpoints. Old cross-scale partial-loading behavior
is intentionally removed to prevent unnoticed random reconstruction heads.
