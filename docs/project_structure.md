# Project Structure

This document summarizes the role of each major component in the repository.

## Top-Level Entries

| Path | Purpose |
| --- | --- |
| `run_capsra.py` | Unified runner across all supported base models |
| `main.py` | Mainline CapSRA training entry |
| `train_eval.py` | Training and evaluation loops |
| `README.md` | GitHub landing page for the public release |
| `CITATION.cff` | Citation metadata for GitHub and downstream users |
| `pic.png` | Main framework figure used by the README |

## Mainline Components

| Path | Purpose |
| --- | --- |
| `data/` | Dataset loading for the unified mainline |
| `models/backbones.py` | Backbone registry and frozen CLIP / ViLT encoders |
| `models/layers.py` | Graph aggregation and fusion modules |
| `models/model.py` | Mainline CapSRA model |
| `utils/process_caption.py` | Caption generation pipeline |
| `utils/process_feature.py` | Retrieval feature extraction and neighbor construction |
| `utils/utils.py` | Common utilities |

## Public Script Entry Points

| Path | Purpose |
| --- | --- |
| `scripts/preprocess/generate_captions.py` | Public entry for caption generation |
| `scripts/preprocess/build_retrieval.py` | Public entry for retrieval construction |
| `scripts/merge_baseline_patch.py` | Utility for merging CapSRA patch bundles into official baseline checkouts |
| `scripts/smoke_check.py` | Lightweight release checklist for required repository assets |

## Configs And Examples

| Path | Purpose |
| --- | --- |
| `configs/` | Ready-to-edit argument presets for supported experiment families |
| `examples/` | Minimal JSONL schema examples for local adaptation |

## Baseline Wrappers

| Path | Purpose |
| --- | --- |
| `baselines/mainline.py` | Wrapper for the unified CLIP / ViLT mainline |
| `baselines/prompthate_capsra/launcher.py` | PromptHate plugin wrapper |
| `baselines/procap_capsra/launcher.py` | Pro-Cap plugin wrapper |
| `baselines/registry.py` | Base model registry |
| `baselines/common.py` | Shared runner utilities for baseline wrappers |

## Baseline Patch Bundle

| Path | Purpose |
| --- | --- |
| `baseline_patches/README.md` | Explains the scope of the patch bundle |
| `baseline_patches/prompthate_capsra_patch/` | CapSRA-side integration files for PromptHate |
| `baseline_patches/procap_capsra_patch/` | CapSRA-side integration files for Pro-Cap |

The merge workflow for these patch bundles is documented in `docs/reproduction.md`.

## Supported Experiment Families

### Mainline

- `clip`
- `vilt`

### Plugin Integrations

- `prompthate`
- `procap`

## Patch Scope

The baseline patch bundle is not a full copy of PromptHate or Pro-Cap. It preserves the model-side CapSRA integration files and should be used together with the official upstream repositories.

## Upload Checklist

The public release should keep the following groups:

- top-level metadata and entry files
- `baselines/`
- `baseline_patches/`
- `configs/`
- `data/`
- `docs/`
- `examples/`
- `models/`
- `scripts/`
- `utils/`