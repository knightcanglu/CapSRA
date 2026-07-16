# Results

This page collects reproducible results and release notes for CapSRA experiments.

## Main Results

CapSRA consistently improves strong multimodal hateful meme detectors across `FHM`, `HarMeme`, and `MAMI`, with gains observed on `Accuracy`, `AUC`, and `Macro-F1` over multiple backbone families and baseline architectures.

## Ablation Template

| Variant | Retrieval | Aggregation | IB | AUC | Accuracy |
| --- | --- | --- | --- | ---: | ---: |
| Base model | - | - | - | - | - |
| + Caption retrieval | yes | mean | no | - | - |
| + GAT aggregation | yes | GAT | no | - | - |
| + Information bottleneck | yes | GAT | yes | - | - |