from pathlib import Path

from baselines.common import PATCH_ROOT, run_python_entry


def run(argv):
    main_ib = "--use_ib" in argv or "--main_ib" in argv
    cleaned_argv = [arg for arg in argv if arg != "--main_ib"]
    script_name = "main_ib.py" if main_ib else "main.py"
    script_path = (
        PATCH_ROOT
        / "procap_capsra_patch"
        / script_name
    )
    run_python_entry(Path(script_path), cleaned_argv)