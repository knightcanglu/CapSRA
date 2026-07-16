import argparse
import importlib

from baselines.registry import get_baseline_entry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified CapSRA runner across clip, vilt, PromptHate, and Pro-Cap."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        choices=["clip", "vilt", "prompthate", "procap"],
        help="Choose which base model family to run with the CapSRA plugin.",
    )
    args, remaining = parser.parse_known_args()
    return args, remaining


def resolve_callable(entry_string):
    module_name, function_name = entry_string.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def main():
    args, remaining = parse_args()
    entry = get_baseline_entry(args.base_model)
    runner = resolve_callable(entry)

    forwarded = list(remaining)
    if args.base_model in {"clip", "vilt"}:
        forwarded = ["--base_model", args.base_model] + forwarded
    runner(forwarded)


if __name__ == "__main__":
    main()