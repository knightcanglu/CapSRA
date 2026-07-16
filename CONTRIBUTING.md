# Contributing

Thank you for your interest in improving CapSRA.

## Scope

This repository is organized around the CapSRA research workflow, including:

- data preparation and caption generation
- retrieval construction and neighbor aggregation
- mainline training and evaluation
- patch-based integration with external baseline repositories

Please keep contributions aligned with the overall CapSRA pipeline and public release quality.

## Recommended Workflow

1. Open an issue or describe the intended change before large refactors.
2. Keep public-facing files and directory names in English.
3. Prefer small, reviewable pull requests over large mixed changes.
4. Update `README.md` or `docs/` when changing user-facing behavior.
5. Keep generated outputs, checkpoints, and logs out of version control.

## Code Style

- Follow the existing Python style in nearby files.
- Preserve compatibility with the current command-line interfaces.
- Avoid introducing baseline-specific assumptions into the unified mainline.
- Keep patch bundle changes isolated to `baseline_patches/` whenever possible.

## Validation

Before submitting a change, verify at least one of the following:

- the mainline command-line interface still parses as expected
- the preprocessing entry points still route correctly
- the baseline patch merge instructions remain accurate

## Questions

If a change touches the paper-facing presentation, repository structure, or release assets, prefer discussing it first so the public release stays consistent.