from pathlib import Path

from baselines.common import PATCH_ROOT, run_python_entry


def run(argv):
    script_path = (
        PATCH_ROOT
        / "prompthate_capsra_patch"
        / "main.py"
    )
    run_python_entry(Path(script_path), argv)