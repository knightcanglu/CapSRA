from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "LICENSE",
    REPO_ROOT / "requirements.txt",
    REPO_ROOT / "run_capsra.py",
    REPO_ROOT / "main.py",
    REPO_ROOT / "baseline_patches" / "README.md",
    REPO_ROOT / "docs" / "reproduction.md",
    REPO_ROOT / "configs" / "clip_mainline.yaml",
    REPO_ROOT / "examples" / "README.md",
]


def main():
    missing = [path for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        print("Smoke check failed. Missing paths:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    print("Smoke check passed. Required release files are present.")
    for path in REQUIRED_PATHS:
        print(f"  + {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()