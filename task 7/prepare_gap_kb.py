from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent.parent
SRC = BASE_DIR / "knowledge_base"
DST = BASE_DIR / "task 7" / "knowledge_base_gap"

# Artificial gaps: remove key entities from evaluation corpus.
REMOVED = {
    "01_ксарн_вэлгор.md",
    "14_синт_поток.md",
    "27_нексус_пустоты.md",
}


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for file in DST.glob("*.md"):
        file.unlink()

    copied = 0
    for path in sorted(SRC.glob("*.md")):
        if path.name in REMOVED:
            continue
        shutil.copy2(path, DST / path.name)
        copied += 1

    print(f"Gap KB prepared: copied={copied}, removed={len(REMOVED)}")


if __name__ == "__main__":
    main()
