BASELINE_REGISTRY = {
    "clip": "baselines.mainline:run_mainline",
    "vilt": "baselines.mainline:run_mainline",
    "prompthate": "baselines.prompthate_capsra.launcher:run",
    "procap": "baselines.procap_capsra.launcher:run",
}


def get_baseline_entry(base_model):
    key = base_model.lower()
    if key not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unsupported base_model '{base_model}'. Supported: {sorted(BASELINE_REGISTRY)}"
        )
    return BASELINE_REGISTRY[key]