# Examples

This directory provides minimal examples for preparing a local CapSRA experiment.

## Files

- `sample_data/train_fix.jsonl`
- `sample_data/val_fix.jsonl`
- `sample_data/test_fix.jsonl`

## Purpose

These files document the expected JSONL schema for the unified CapSRA mainline. They are not benchmark data and are not intended for reporting paper results.

## Minimal Workflow

1. Replace the placeholder image names with files that exist in your own `img/` directory.
2. Generate captions with `scripts/preprocess/generate_captions.py`.
3. Build retrieval files with `scripts/preprocess/build_retrieval.py`.
4. Train the mainline with `python run_capsra.py --base_model clip ...` or `python run_capsra.py --base_model vilt ...`.