# CapSRA

<div align="center">

## LMM-Guided Caption Semantic Retrieval Aggregation for Hateful Meme Detection

[![Paper](https://img.shields.io/badge/Paper-Pattern%20Recognition-blue)](https://www.sciencedirect.com/science/article/abs/pii/S0031320326007466)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](./requirements.txt)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Code-orange)](./README.md)

</div>

## Table of Contents

- [Overview](#overview)
- [Framework](#framework)
- [Main Results](#main-results)
- [Quick Start](#quick-start)
- [Reproduction](#reproduction)
- [Documentation](#documentation)

## Overview

CapSRA is a retrieval-augmented framework for hateful meme detection. It combines:

- LMM-guided meme-aware caption generation
- caption-enhanced multimodal retrieval over semantic neighbors
- graph attention based neighbor aggregation
- information bottleneck based feature compression
- plug-in integration with multiple base architectures

## Framework

<div align="center">
  <img src="./pic.png" alt="CapSRA framework" width="100%">
</div>

## Highlights

- End-to-end CapSRA workflow from caption generation to semantic retrieval
- Retrieval-augmented multimodal reasoning for hateful meme detection
- Reproducible project layout for paper-oriented experimentation
- Release assets for mainline runs, patch-based baselines, and documentation

## Related Work

| Model family | Paper / Code |
| --- | --- |
| `PromptHate` | [GitHub](https://github.com/Social-AI-Studio/PromptHate) |
| `Pro-Cap` | [GitHub](https://github.com/Social-AI-Studio/Pro-Cap) |
| `CapSRA` | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320326007466) |

## Main Results

CapSRA consistently improves strong multimodal hateful meme detectors across `FHM`, `HarMeme`, and `MAMI`, with gains observed on `Accuracy`, `AUC`, and `Macro-F1` over multiple backbone families and baseline architectures.

## Repository Layout

```text
CapSRA-master/
├── README.md
├── LICENSE
├── CITATION.cff
├── pic.png
├── requirements.txt
├── run_capsra.py
├── main.py
├── train_eval.py
├── baselines/
├── baseline_patches/
├── configs/
├── data/
├── docs/
├── examples/
├── models/
├── scripts/
└── utils/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Format

Each split is expected as a JSONL file with one sample per line:

```json
{"id": "sample_id", "img": "sample.png", "text": "meme text", "label": 0}
```

The mainline loader accepts:

- image field: `img` or `image`
- text field: `text`, `caption`, or `description`
- label field: `label` or `labels`

## Quick Start

### 1. Generate meme-aware captions

```bash
python scripts/preprocess/generate_captions.py \
  --jsonl_file /path/to/train_fix.jsonl \
  --img_folder /path/to/img \
  --output_file outputs/captions/train_captions.json \
  --device cuda:0
```

### 2. Build retrieval features and neighbors

```bash
python scripts/preprocess/build_retrieval.py \
  --base_model /path/to/backbone_choice \
  --train_jsonl /path/to/train_fix.jsonl \
  --query_jsonl /path/to/val_fix.jsonl \
  --train_caption_json outputs/captions/train_captions.json \
  --query_caption_json outputs/captions/val_captions.json \
  --image_dir /path/to/img \
  --train_feature_out outputs/features/train_feats.pt \
  --query_feature_out outputs/features/val_feats.pt \
  --train_neighbor_out outputs/neighbors/train_neighbors.json \
  --query_neighbor_out outputs/neighbors/val_neighbors.json \
  --top_k 10 \
  --exclude_self
```

### 3. Train CapSRA

```bash
python run_capsra.py \
  --data_dir /path/to/dataset_root \
  --feature_path_train outputs/features/train_feats.pt \
  --feature_path_val outputs/features/val_feats.pt \
  --feature_path_test outputs/features/test_feats.pt \
  --neighbor_path_train outputs/neighbors/train_neighbors.json \
  --neighbor_path_val outputs/neighbors/val_neighbors.json \
  --neighbor_path_test outputs/neighbors/test_neighbors.json \
  --output_dir runs/capsra
```

## Reproduction

- Mainline preprocessing and training are documented in `docs/reproduction.md`
- Patch-based baseline integration is documented in `baseline_patches/README.md`
- Config presets are provided in `configs/`

## Documentation

- [Reproduction Guide](./docs/reproduction.md)
- [Project Structure](./docs/project_structure.md)
- [Results](./docs/results.md)
- [Baseline Patch Bundle](./baseline_patches/README.md)
- [Config Presets](./configs/README.md)
- [Examples](./examples/README.md)
- [Contributing](./CONTRIBUTING.md)

## Citation

If you use this repository in your research, please cite the CapSRA paper.

```bibtex
@article{capsra2026,
  title = {CapSRA: LMM-guided caption semantic retrieval aggregation for hateful meme detection},
  journal = {Pattern Recognition},
  year = {2026}
}
```