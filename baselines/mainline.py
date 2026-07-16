from pathlib import Path

from baselines.common import run_python_entry


def run_mainline(argv):
    script_path = Path(__file__).resolve().parent.parent / "main.py"
    run_python_entry(script_path, argv)