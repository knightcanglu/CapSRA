# Baseline Patch Bundle

This directory stores the CapSRA-specific patch files for the official `PromptHate` and `Pro-Cap` baselines.

## Important Note

These files are not a full copy of the upstream repositories.

- `baseline_patches/prompthate_capsra_patch/` contains the PromptHate-side CapSRA integration files.
- `baseline_patches/procap_capsra_patch/` contains the Pro-Cap-side CapSRA integration files.
- Full reproduction still requires the official upstream repositories:
  - PromptHate: https://github.com/Social-AI-Studio/PromptHate
  - Pro-Cap: https://github.com/Social-AI-Studio/Pro-Cap

## Recommended Usage

1. Clone the official upstream baseline repository.
2. Copy or merge the corresponding files from this directory into the upstream project checkout.
3. Prepare CapSRA retrieval features and neighbor files from this repository.
4. Launch the baseline experiment with the CapSRA patch enabled.

### PromptHate Merge Example

```bash
git clone https://github.com/Social-AI-Studio/PromptHate.git external/PromptHate
python scripts/merge_baseline_patch.py \
  --baseline prompthate \
  --repo_dir external/PromptHate \
  --overwrite
```

### Pro-Cap Merge Example

```bash
git clone https://github.com/Social-AI-Studio/Pro-Cap.git external/Pro-Cap
python scripts/merge_baseline_patch.py \
  --baseline procap \
  --repo_dir external/Pro-Cap \
  --overwrite
```

## Scope

This bundle is intentionally limited to the CapSRA-related integration code and nearby entry files required to document the plugin logic.