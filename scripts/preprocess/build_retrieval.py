from pathlib import Path
import runpy


def main():
    script_path = Path(__file__).resolve().parents[2] / "utils" / "process_feature.py"
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()