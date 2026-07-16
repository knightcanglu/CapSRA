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
- [Quick Start](#quick-start)
- [Supported Base Models](#supported-base-models)
- [Baseline Patches](#baseline-patches)
- [Results](#reference-results)
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

- Unified mainline implementation with both `CLIP` and `ViLT` backbones
- Retrieval construction pipeline from captions to neighbor graphs
- Unified experiment runner for `clip`, `vilt`, `prompthate`, and `procap`
- Ready-to-extend plugin structure for baseline-specific integrations
- Config presets and sample JSONL files for quick adaptation

## Official Baselines

| Model family | Paper / Code |
| --- | --- |
| `PromptHate` | [GitHub](https://github.com/Social-AI-Studio/PromptHate) |
| `Pro-Cap` | [GitHub](https://github.com/Social-AI-Studio/Pro-Cap) |
| `CapSRA` | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320326007466) |

## Baseline Patches

`baseline_patches/` stores the CapSRA-side integration files for `PromptHate` and `Pro-Cap`.

- It is not a full mirror of the upstream baseline repositories.
- Full reproduction still requires the official upstream codebases.
- The patch bundle is included to document the plugin logic and preserve the model-side modifications used in this project.

Merge a patch bundle into an official checkout with:

```bash
python scripts/merge_baseline_patch.py \
  --baseline prompthate \
  --repo_dir external/PromptHate \
  --overwrite
```

```bash
python scripts/merge_baseline_patch.py \
  --baseline procap \
  --repo_dir external/Pro-Cap \
  --overwrite
```

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

## Supported Base Models

| Base model | Command entry | Description |
| --- | --- | --- |
| `clip` | `python run_capsra.py --base_model clip ...` | CapSRA mainline with frozen CLIP |
| `vilt` | `python run_capsra.py --base_model vilt ...` | CapSRA mainline with frozen ViLT |
| `prompthate` | `python run_capsra.py --base_model prompthate ...` | PromptHate with CapSRA retrieval aggregation |
| `procap` | `python run_capsra.py --base_model procap ...` | Pro-Cap with CapSRA retrieval aggregation |

## Reference Results

| Model | Dataset | AUC | Accuracy | Notes |
| --- | --- | ---: | ---: | --- |
| PromptHate | HarM | 91.93 | 87.29 | reference reproduction |
| PromptHate | Mem | 82.51 | 72.40 | reference reproduction |

## Benchmark Sheet

| Variant | FHM | HarM | MAMI | Notes |
| --- | ---: | ---: | ---: | --- |
| CapSRA-CLIP | - | - | - | mainline release target |
| CapSRA-ViLT | - | - | - | mainline release target |
| PromptHate + CapSRA | - | 91.93 / 87.29 | 82.51 / 72.40 | AUC / Accuracy reference runs |
| Pro-Cap + CapSRA | - | - | - | plugin release target |

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
  --base_model clip \
  --train_jsonl /path/to/train_fix.jsonl \
  --query_jsonl /path/to/val_fix.jsonl \
  --train_caption_json outputs/captions/train_captions.json \
  --query_caption_json outputs/captions/val_captions.json \
  --image_dir /path/to/img \
  --train_feature_out outputs/features/train_clip_feats.pt \
  --query_feature_out outputs/features/val_clip_feats.pt \
  --train_neighbor_out outputs/neighbors/train_clip_neighbors.json \
  --query_neighbor_out outputs/neighbors/val_clip_neighbors.json \
  --top_k 10 \
  --exclude_self
```

### 3. Train CapSRA

```bash
python run_capsra.py \
  --base_model clip \
  --data_dir /path/to/dataset_root \
  --feature_path_train outputs/features/train_clip_feats.pt \
  --feature_path_val outputs/features/train_clip_feats.pt \
  --feature_path_test outputs/features/train_clip_feats.pt \
  --neighbor_path_train outputs/neighbors/train_clip_neighbors.json \
  --neighbor_path_val outputs/neighbors/val_clip_neighbors.json \
  --neighbor_path_test outputs/neighbors/test_clip_neighbors.json \
  --output_dir runs/clip_capsra
```

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