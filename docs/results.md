# Results

This page collects reproducible results and release notes for CapSRA experiments.

## Main Results

| Model | Dataset | AUC | Accuracy | Notes |
| --- | --- | ---: | ---: | --- |
| PromptHate | HarM | 91.93 | 87.29 | reference reproduction |
| PromptHate | Mem | 82.51 | 72.40 | reference reproduction |
| CapSRA-CLIP | FHM / HarM / MAMI | - | - | to be released |
| CapSRA-ViLT | FHM / HarM / MAMI | - | - | to be released |
| Pro-Cap + CapSRA | FHM / HarM / MAMI | - | - | to be released |

## Release Matrix

| Variant | FHM | HarM | MAMI | Notes |
| --- | ---: | ---: | ---: | --- |
| CapSRA-CLIP | - | - | - | mainline benchmark |
| CapSRA-ViLT | - | - | - | mainline benchmark |
| PromptHate + CapSRA | - | 91.93 / 87.29 | 82.51 / 72.40 | AUC / Accuracy reference runs |
| Pro-Cap + CapSRA | - | - | - | plugin benchmark |

## Ablation Template

| Variant | Retrieval | Aggregation | IB | AUC | Accuracy |
| --- | --- | --- | --- | ---: | ---: |
| Base model | - | - | - | - | - |
| + Caption retrieval | yes | mean | no | - | - |
| + GAT aggregation | yes | GAT | no | - | - |
| + Information bottleneck | yes | GAT | yes | - | - |