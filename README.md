# NASMark

Reference implementation of **NASMark: Enabling Stable and Efficient
Watermarking for NAS-Generated Vision Models**.

This release currently provides the SPOS MobileNet search space on CIFAR-10.
It embeds a black-box watermark into a weight-sharing NAS supernet so that
searched subnets inherit the watermark without `recover()`, post-search
fine-tuning, or per-subnet watermark embedding.

> **Research release.** The implementation and end-to-end workflow are
> available, but the repository does not yet include pretrained paper
> checkpoints or a complete reproduction table. Short smoke runs are execution
> checks and must not be interpreted as paper results.

## Highlights

- SPOS single-path training with MobileNetV2-style candidate blocks.
- One residual watermark module per selected layer, shared by every candidate
  path at that layer.
- Full convolutional watermark bottleneck:
  `1x1 pointwise -> 3x3 depthwise -> 1x1 pointwise`.
- Two explicit training stages matching the NASMark procedure.
- First-order Taylor scoring for selecting low-contribution backbone tensors.
- Resource-constrained search over frozen weights.
- Standalone subnet export with inherited watermark modules.
- Deterministic CIFAR-10 patch-trigger construction and black-box WSR
  verification.

## Method overview

The default CIFAR-10 supernet has ten searchable layers. Each layer contains
four MobileNet candidates with expansion ratios `1`, `3`, `6`, and `4`.
Downsampling occurs at layers `1`, `3`, and `6`.

Following the paper setting `k=3`, NASMark attaches watermark modules to the
last three searchable positions: layers `7`, `8`, and `9`.

```text
                                      selected candidate fi
                                    ┌────────────────────────┐
layer input hi-1 ───────────────────┤                        ├── (+) ── hi
                                    └─ shared WM module gi ──┘
```

There is exactly one `gi` at a watermarked layer. All four candidates at that
layer share its parameters. In the default supernet this means three watermark
modules, not twelve.

For convolutional features, each watermark module is:

```text
1x1 Conv -> BN -> ReLU
3x3 Depthwise Conv -> BN -> ReLU
1x1 Conv -> BN -> ReLU
```

Its output is added through a scaled residual connection:

```text
hi = fi(hi-1; alpha_i) + watermark_scale * gi(hi-1)
```

The release keeps the final ReLU and default residual scale `0.2` from the
author's earlier implementation. Both are isolated in the model definition for
future ablation.

## Training protocol

### Stage 1: watermark supernet training

The watermark modules participate in the forward pass from the first
iteration. For every SPOS step, one candidate is sampled at each layer and all
three parameter groups are optimized on clean data:

- searchable backbone parameters `theta`;
- fixed shared parameters `psi` (stem and classifier in this implementation);
- watermark module parameters `phi`.

Stage 1 does **not** use trigger data.

### Stage 2: watermark embedding

Starting from the converged Stage 1 checkpoint, the implementation:

1. estimates `|<w, dLmain/dw>|` for searchable convolution weight tensors;
2. selects the lowest-scoring fraction `rho` as `theta_cpl`;
3. freezes the remaining searchable backbone;
4. optimizes `theta_cpl`, the shared part, and watermark modules using
   `Lmain + lambda_wm * Lwm`.

The shared stem and classifier use reduced learning rates in this release. This
stabilization preserves the paper's optimization objective while reducing the
risk that a small, single-label trigger set overwrites the main classifier.

### Architecture search

After Stage 2, all parameters are frozen. Candidate paths satisfying the target
parameter budget are ranked by clean validation accuracy. WSR is reported for
analysis but is not part of the architecture selection criterion. The selected
backbone path, shared part, and watermark modules are copied directly into a
standalone subnet.

Search contains no recovery, retraining, or additional watermark embedding.

## Default CIFAR-10 configuration

The paper explicitly specifies 100 private watermark samples, a fixed 3x3
corner patch, and watermark modules at the last three backbone positions. The
training defaults below combine those settings with the SPOS release schedule
and the requested watermark-loss balance.

| Setting | Default |
|---|---:|
| Stage 1 epochs | 200 |
| Stage 1 learning rate | 0.025 |
| Stage 2 epochs | 20 |
| Stage 2 backbone/watermark learning rate | 0.01 |
| Stage 2 shared stem learning rate | 0.0005 (`0.05x`) |
| Stage 2 shared classifier learning rate | 0.001 (`0.1x`) |
| Momentum | 0.9 |
| Weight decay | `3e-4` |
| Scheduler | cosine annealing |
| `lambda_wm` | 0.5 |
| Low-contribution fraction `rho` | 0.01 |
| Contribution-scoring batches | 100 |
| Private watermark samples | 100 |
| Trigger | fixed white 3x3 bottom-right patch |
| Watermarked layers | final 3 layers |
| Watermark residual scale | 0.2 |

The manuscript defines the objective and structural settings but does not list
a complete optimizer table. The learning-rate values and multipliers are
therefore documented as release defaults rather than claimed as verbatim paper
hyperparameters.

## Repository layout

```text
nasmark/
  models/blocks.py       MobileNet candidate and watermark module
  models/supernet.py     SPOS supernet and standalone subnet export
  data.py                CIFAR-10 transforms and trigger construction
  training.py            Stage 1, Taylor scoring, Stage 2, evaluation
  search.py              frozen resource-constrained random search
  checkpoint.py          portable checkpoint I/O
train_supernet.py        Stage 1 command-line entry point
embed_watermark.py       Stage 2 command-line entry point
search.py                search and subnet extraction entry point
verify.py                clean accuracy and WSR verification
tests/                   structural and pipeline tests
```

## Installation

Requirements:

- Python 3.10 or newer
- PyTorch 1.13 or newer
- torchvision 0.14 or newer
- CUDA-capable GPU recommended for full training

Install in an isolated environment:

```bash
git clone <YOUR-NASMARK-REPOSITORY-URL>
cd nasmark-release
python -m pip install -e .
```

The existing workspace Conda environment can also be used:

```powershell
conda run -n env python -m unittest discover -s tests -v
```

## Reproduce the CIFAR-10 workflow

Run commands from the repository root.

### 1. Train the watermark supernet

```bash
python train_supernet.py \
  --data ./data \
  --download \
  --epochs 200 \
  --learning-rate 0.025 \
  --output checkpoints/stage1_supernet.pt
```

### 2. Embed the watermark

```bash
python embed_watermark.py \
  --data ./data \
  --input checkpoints/stage1_supernet.pt \
  --output checkpoints/nasmark_supernet.pt \
  --epochs 20 \
  --watermark-weight 0.5 \
  --rho 0.01 \
  --learning-rate 0.01 \
  --stem-lr-scale 0.05 \
  --classifier-lr-scale 0.1
```

### 3. Search and export a small subnet

The following example evaluates 100 frozen candidates with at most 800,000
parameters:

```bash
python search.py \
  --data ./data \
  --input checkpoints/nasmark_supernet.pt \
  --output checkpoints/best_small_subnet.pt \
  --samples 100 \
  --max-parameters 800000
```

### 4. Verify one supernet path

```bash
python verify.py \
  --data ./data \
  --input checkpoints/nasmark_supernet.pt \
  --choice 0,1,2,3,0,1,2,3,0,1
```

To reuse the dataset in the original local project, replace `./data` with:

```text
../finalversion/dataset/cifar10
```

## Fast execution check

`--smoke-test` uses four synthetic samples and a tiny version of the same
search space. It verifies execution only:

```bash
python train_supernet.py --smoke-test --device cpu --epochs 1 \
  --output checkpoints/smoke_stage1.pt
python embed_watermark.py --smoke-test --device cpu --epochs 1 \
  --score-batches 1 --rho 0.25 \
  --input checkpoints/smoke_stage1.pt \
  --output checkpoints/smoke_nasmark.pt
python search.py --smoke-test --device cpu --samples 3 \
  --input checkpoints/smoke_nasmark.pt \
  --output checkpoints/smoke_subnet.pt
```

## Evaluating stable inheritance

A high WSR alone is insufficient: a model that predicts the target class for
ordinary inputs is not a valid watermarked model. A complete evaluation should
report:

- Stage 1 and Stage 2 clean validation accuracy;
- WSR for multiple high-performing searched subnets;
- mean and variance of top-k subnet WSR;
- FPR on independently trained unwatermarked models;
- clean prediction distribution to detect target-class collapse;
- parameter count, FLOPs, or latency constraint used during search.

Pretrained checkpoints, full seed-controlled reproduction scripts, FLOPs and
latency constraints, and the complete paper result table are planned follow-up
work.

## License

Released under the [MIT License](LICENSE).

