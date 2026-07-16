import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _detect_patch_root():
    candidates = []
    for path in REPO_ROOT.iterdir():
        if not path.is_dir():
            continue
        names = {child.name for child in path.iterdir() if child.is_dir()}
        if {"prompthate_capsra_patch", "procap_capsra_patch"}.issubset(names):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "Could not locate the baseline patch bundle containing PromptHate and Pro-Cap integration files."
        )
    return sorted(candidates)[0]


PATCH_ROOT = _detect_patch_root()


def run_python_entry(script_path, argv):
    script_path = Path(script_path).resolve()
    script_dir = str(script_path.parent)

    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        sys.argv = [str(script_path)] + list(argv)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path = old_path