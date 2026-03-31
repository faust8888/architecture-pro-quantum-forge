import json
from pathlib import Path


def replace_terms_in_text(text: str, mapping: dict[str, str]) -> str:
    result = text
    # Replace longer keys first to avoid partial overlaps.
    for src in sorted(mapping.keys(), key=len, reverse=True):
        result = result.replace(src, mapping[src])
    return result


def process_directory(input_dir: Path, output_dir: Path, mapping_path: Path) -> None:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in input_dir.glob("*.md"):
        original = path.read_text(encoding="utf-8")
        transformed = replace_terms_in_text(original, mapping)
        (output_dir / path.name).write_text(transformed, encoding="utf-8")


if __name__ == "__main__":
    # Example:
    # python scripts/replace_terms.py
    # Expects source docs in raw_knowledge_base/ and writes to knowledge_base/
    process_directory(
        input_dir=Path("raw_knowledge_base"),
        output_dir=Path("knowledge_base"),
        mapping_path=Path("terms_map.json"),
    )
