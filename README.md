# NASMark

This repository provides an open-source implementation of **NASMark: Enabling
Stable and Efficient Watermarking for NAS-Generated Visual Models**.

The current release implements NASMark with an SPOS MobileNet search space on
CIFAR-10. Watermark modules are inserted as residual branches and shared by all
candidate paths at the same searchable layer. Each module uses a
`1x1 pointwise -> 3x3 depthwise -> 1x1 pointwise` structure.

## Workflow

The code is organized into three stages:

1. Train the SPOS supernet on the main task while jointly training the shared
   watermark modules.
2. Embed the trigger behavior into the supernet using the dedicated watermark
   class.
3. Search the trained supernet and directly export a watermarked subnet.

## Code structure

| Path | Purpose |
|---|---|
| `train_supernet.py` | Trains the SPOS supernet and shared watermark modules on CIFAR-10. |
| `embed_watermark.py` | Loads the trained supernet and performs watermark embedding. |
| `search.py` | Searches frozen candidate architectures and exports the selected subnet. |
| `verify.py` | Evaluates clean accuracy and watermark success rate for a supernet path. |
| `nasmark/models/blocks.py` | Defines MobileNet candidate blocks and watermark modules. |
| `nasmark/models/supernet.py` | Defines the SPOS supernet, shared residual watermark branches, eleventh-class head, and subnet export. |
| `nasmark/data.py` | Builds CIFAR-10 loaders and deterministic trigger datasets. |
| `nasmark/training.py` | Implements supernet training, contribution scoring, watermark training, and evaluation. |
| `nasmark/search.py` | Implements resource-constrained architecture search without retraining. |
| `nasmark/checkpoint.py` | Provides checkpoint loading and saving utilities. |
| `nasmark/cli.py` | Contains shared command-line and construction helpers. |
| `tests/` | Contains model-structure and end-to-end pipeline tests. |

## Installation

```bash
python -m pip install -e .
```

The project requires Python 3.10 or newer, PyTorch, and torchvision.

## Usage

Train the supernet:

```bash
python train_supernet.py --download
```

Embed the watermark:

```bash
python embed_watermark.py
```

Search and export a subnet:

```bash
python search.py
```

Verify a supernet path:

```bash
python verify.py --choice 0,1,2,3,0,1,2,3,0,1
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Additional options are available through `python <script>.py --help`.
