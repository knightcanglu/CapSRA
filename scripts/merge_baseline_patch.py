import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_SPECS = {
    "prompthate": {
        "patch_dir": REPO_ROOT / "baseline_patches" / "prompthate_capsra_patch",
        "target_candidates": [
            Path("PromptHate-Code"),
            Path("."),
        ],
        "required_files": ["main.py", "train.py", "config.py", "dataset.py", "baselineGAT.py"],
    },
    "procap": {
        "patch_dir": REPO_ROOT / "baseline_patches" / "procap_capsra_patch",
        "target_candidates": [
            Path("codes") / "scr",
            Path("scr"),
            Path("."),
        ],
        "required_files": ["main.py", "main_ib.py", "train.py", "train_ib.py", "config.py", "dataset.py", "pbm_gat.py"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge CapSRA baseline patch files into an official PromptHate or Pro-Cap checkout."
    )
    parser.add_argument(
        "--baseline",
        choices=sorted(PATCH_SPECS),
        required=True,
        help="Target baseline family.",
    )
    parser.add_argument(
        "--repo_dir",
        type=Path,
        required=True,
        help="Path to the cloned official upstream repository.",
    )
    parser.add_argument(
        "--target_subdir",
        type=Path,
        default=None,
        help="Optional explicit target subdirectory inside repo_dir.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned file copies without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the target checkout.",
    )
    return parser.parse_args()


def resolve_target_dir(repo_dir, target_subdir, spec):
    repo_dir = repo_dir.resolve()
    if target_subdir is not None:
        target = (repo_dir / target_subdir).resolve()
        if not target.exists():
            raise FileNotFoundError(f"Target subdirectory does not exist: {target}")
        return target

    for candidate in spec["target_candidates"]:
        target = repo_dir / candidate
        if target.exists():
            return target.resolve()

    candidates = ", ".join(str(item) for item in spec["target_candidates"])
    raise FileNotFoundError(
        f"Could not infer target directory under {repo_dir}. Tried: {candidates}. "
        "Use --target_subdir to specify it explicitly."
    )


def collect_patch_files(patch_dir):
    return sorted(path for path in patch_dir.iterdir() if path.is_file())


def merge_patch_files(patch_files, target_dir, overwrite, dry_run):
    copied = []
    skipped = []

    for source in patch_files:
        destination = target_dir / source.name
        if destination.exists() and not overwrite:
            skipped.append(destination)
            continue
        if dry_run:
            copied.append(destination)
            continue
        shutil.copy2(source, destination)
        copied.append(destination)

    return copied, skipped


def main():
    args = parse_args()
    spec = PATCH_SPECS[args.baseline]
    patch_dir = spec["patch_dir"]
    if not patch_dir.exists():
        raise FileNotFoundError(f"Patch directory does not exist: {patch_dir}")
    if not args.repo_dir.exists():
        raise FileNotFoundError(f"Upstream repository does not exist: {args.repo_dir}")

    target_dir = resolve_target_dir(args.repo_dir, args.target_subdir, spec)
    patch_files = collect_patch_files(patch_dir)
    copied, skipped = merge_patch_files(
        patch_files=patch_files,
        target_dir=target_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    action = "Would copy" if args.dry_run else "Copied"
    print(f"{action} {len(copied)} files into {target_dir}")
    for path in copied:
        print(f"  + {path}")

    if skipped:
        print(f"Skipped {len(skipped)} existing files. Re-run with --overwrite to replace them.")
        for path in skipped:
            print(f"  - {path}")

    missing = [name for name in spec["required_files"] if not (target_dir / name).exists()]
    if missing:
        print("Warning: expected files are still missing after merge:")
        for name in missing:
            print(f"  ! {name}")
    else:
        print("Patch merge check passed.")


if __name__ == "__main__":
    main()