# Reproduction Guide

This document describes a standard end-to-end workflow for reproducing CapSRA experiments.

## 1. Data Preparation

Organize each dataset as:

```text
dataset_root/
├── train_fix.jsonl
├── val_fix.jsonl
├── test_fix.jsonl
└── img/
    ├── sample_1.png
    └── ...
```

Each JSONL line should contain at least:

```json
{"id": "sample_id", "img": "sample.png", "text": "meme text", "label": 0}
```

## 2. Caption Generation

Training split:

```bash
python scripts/preprocess/generate_captions.py \
  --jsonl_file dataset_root/train_fix.jsonl \
  --img_folder dataset_root/img \
  --output_file outputs/captions/train_captions.json \
  --device cuda:0
```

Validation split:

```bash
python scripts/preprocess/generate_captions.py \
  --jsonl_file dataset_root/val_fix.jsonl \
  --img_folder dataset_root/img \
  --output_file outputs/captions/val_captions.json \
  --device cuda:0
```

Test split:

```bash
python scripts/preprocess/generate_captions.py \
  --jsonl_file dataset_root/test_fix.jsonl \
  --img_folder dataset_root/img \
  --output_file outputs/captions/test_captions.json \
  --device cuda:0
```

## 3. Retrieval Construction

### CLIP Retrieval Space

Train-to-train neighbors:

```bash
python scripts/preprocess/build_retrieval.py \
  --base_model clip \
  --train_jsonl dataset_root/train_fix.jsonl \
  --query_jsonl dataset_root/train_fix.jsonl \
  --train_caption_json outputs/captions/train_captions.json \
  --query_caption_json outputs/captions/train_captions.json \
  --image_dir dataset_root/img \
  --train_feature_out outputs/features/train_clip_feats.pt \
  --query_feature_out outputs/features/train_clip_query_feats.pt \
  --train_neighbor_out outputs/neighbors/train_clip_neighbors.json \
  --query_neighbor_out outputs/neighbors/train_clip_query_neighbors.json \
  --top_k 10 \
  --exclude_self
```

Validation-to-train neighbors:

```bash
python scripts/preprocess/build_retrieval.py \
  --base_model clip \
  --train_jsonl dataset_root/train_fix.jsonl \
  --query_jsonl dataset_root/val_fix.jsonl \
  --train_caption_json outputs/captions/train_captions.json \
  --query_caption_json outputs/captions/val_captions.json \
  --image_dir dataset_root/img \
  --train_feature_out outputs/features/train_clip_feats.pt \
  --query_feature_out outputs/features/val_clip_feats.pt \
  --train_neighbor_out outputs/neighbors/train_clip_neighbors.json \
  --query_neighbor_out outputs/neighbors/val_clip_neighbors.json \
  --top_k 10 \
  --exclude_self
```

Test-to-train neighbors:

```bash
python scripts/preprocess/build_retrieval.py \
  --base_model clip \
  --train_jsonl dataset_root/train_fix.jsonl \
  --query_jsonl dataset_root/test_fix.jsonl \
  --train_caption_json outputs/captions/train_captions.json \
  --query_caption_json outputs/captions/test_captions.json \
  --image_dir dataset_root/img \
  --train_feature_out outputs/features/train_clip_feats.pt \
  --query_feature_out outputs/features/test_clip_feats.pt \
  --train_neighbor_out outputs/neighbors/train_clip_neighbors.json \
  --query_neighbor_out outputs/neighbors/test_clip_neighbors.json \
  --top_k 10 \
  --exclude_self
```

### ViLT Retrieval Space

Use the same commands as above, replacing:

- `--base_model clip` with `--base_model vilt`
- output file names with the `vilt` variant

## 4. Mainline Training

### CLIP Backbone

```bash
python run_capsra.py \
  --base_model clip \
  --data_dir dataset_root \
  --feature_path_train outputs/features/train_clip_feats.pt \
  --feature_path_val outputs/features/train_clip_feats.pt \
  --feature_path_test outputs/features/train_clip_feats.pt \
  --neighbor_path_train outputs/neighbors/train_clip_neighbors.json \
  --neighbor_path_val outputs/neighbors/val_clip_neighbors.json \
  --neighbor_path_test outputs/neighbors/test_clip_neighbors.json \
  --output_dir runs/clip_capsra \
  --aggregation_type GAT \
  --fusion_type Gated \
  --use_ib \
  --monitor_metric auc \
  --batch_size 32 \
  --epochs 20 \
  --device cuda:0
```

### ViLT Backbone

```bash
python run_capsra.py \
  --base_model vilt \
  --data_dir dataset_root \
  --feature_path_train outputs/features/train_vilt_feats.pt \
  --feature_path_val outputs/features/train_vilt_feats.pt \
  --feature_path_test outputs/features/train_vilt_feats.pt \
  --neighbor_path_train outputs/neighbors/train_vilt_neighbors.json \
  --neighbor_path_val outputs/neighbors/val_vilt_neighbors.json \
  --neighbor_path_test outputs/neighbors/test_vilt_neighbors.json \
  --output_dir runs/vilt_capsra \
  --aggregation_type GAT \
  --fusion_type Gated \
  --use_ib \
  --monitor_metric auc \
  --batch_size 32 \
  --epochs 20 \
  --device cuda:0
```

## 5. PromptHate Plugin

### Merge Patch Into Official PromptHate

`baseline_patches/prompthate_capsra_patch/` is a patch bundle, not a full PromptHate repository. Use it together with the official upstream codebase.

```bash
git clone https://github.com/Social-AI-Studio/PromptHate.git external/PromptHate
python scripts/merge_baseline_patch.py \
  --baseline prompthate \
  --repo_dir external/PromptHate \
  --overwrite
```

After merging, the PromptHate working tree should contain the CapSRA-enabled files:

```text
external/PromptHate/PromptHate-Code/
├── main.py
├── train.py
├── config.py
├── dataset.py
├── baselineGAT.py
├── classifier.py
└── utils.py
```

Then prepare CapSRA retrieval files from this repository and pass them to the merged PromptHate entry:

```bash
python run_capsra.py \
  --base_model prompthate \
  --DATASET mem \
  --neighbor_path_train /path/to/train_neighbors.json \
  --feature_path_train /path/to/train_feats.pt \
  --neighbor_path_dev /path/to/test_neighbors.json \
  --feature_path_dev /path/to/train_feats.pt
```

## 6. Pro-Cap Plugin

### Merge Patch Into Official Pro-Cap

`baseline_patches/procap_capsra_patch/` is a patch bundle, not a full Pro-Cap repository. Use it together with the official upstream codebase.

```bash
git clone https://github.com/Social-AI-Studio/Pro-Cap.git external/Pro-Cap
python scripts/merge_baseline_patch.py \
  --baseline procap \
  --repo_dir external/Pro-Cap \
  --overwrite
```

After merging, the Pro-Cap working tree should contain the CapSRA-enabled files:

```text
external/Pro-Cap/codes/scr/
├── main.py
├── main_ib.py
├── train.py
├── train_ib.py
├── config.py
├── dataset.py
├── pbm_gat.py
├── pbm_gat_ib.py
├── classifier.py
└── utils.py
```

Then prepare CapSRA retrieval files from this repository and pass them to the merged Pro-Cap entry:

```bash
python run_capsra.py \
  --base_model procap \
  --DATASET mem \
  --train_neighbor_path /path/to/train_neighbors.json \
  --train_feature_path /path/to/train_feats.pt \
  --test_neighbor_path /path/to/test_neighbors.json \
  --test_feature_path /path/to/train_feats.pt
```

IB variant:

```bash
python run_capsra.py --base_model procap --main_ib ...
```

## 7. Patch Bundle Notes

- The patch files are intentionally scoped to the model-side CapSRA integration.
- They do not replace upstream dataset assets, caption assets, pretrained checkpoints, or project-level scripts.
- If the upstream repository changes, merge the patch files manually and resolve conflicts around `main.py`, `train.py`, `config.py`, `dataset.py`, and the CapSRA model files.
- Keep generated outputs outside the upstream checkout or add them to `.gitignore`.

## 8. Outputs

The mainline runner saves:

- `run_config.json`
- `training_log.txt`
- `best_model.pt`

under the specified `output_dir`.