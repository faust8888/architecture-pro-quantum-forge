import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = Path(os.getenv("RAG_SOURCE_DIR", str(BASE_DIR / "knowledge_base")))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", str(BASE_DIR / "task 3" / "artifacts")))
STATE_DIR = BASE_DIR / "task 6" / "state"
LOG_DIR = BASE_DIR / "task 6" / "logs"
STATE_PATH = STATE_DIR / "source_state.json"
JSON_LOG_PATH = LOG_DIR / "update_index.jsonl"
TEXT_LOG_PATH = LOG_DIR / "update_index.log"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dataclass
class SourceEntry:
    path: str
    sha256: str
    mtime: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("update_index")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.INFO)
    logger.addHandler(stream)

    text_file = logging.FileHandler(TEXT_LOG_PATH, encoding="utf-8")
    text_file.setLevel(logging.INFO)
    logger.addHandler(text_file)
    return logger


def load_previous_state() -> dict[str, SourceEntry]:
    if not STATE_PATH.exists():
        return {}
    raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    out: dict[str, SourceEntry] = {}
    for item in raw.get("files", []):
        out[item["path"]] = SourceEntry(
            path=item["path"],
            sha256=item["sha256"],
            mtime=float(item["mtime"]),
        )
    return out


def scan_source() -> dict[str, SourceEntry]:
    current: dict[str, SourceEntry] = {}
    for path in sorted(SOURCE_DIR.glob("*.md")):
        rel = str(path.relative_to(BASE_DIR))
        current[rel] = SourceEntry(
            path=rel,
            sha256=sha256_file(path),
            mtime=path.stat().st_mtime,
        )
    return current


def diff_state(prev: dict[str, SourceEntry], curr: dict[str, SourceEntry]) -> tuple[list[str], list[str], list[str]]:
    prev_keys = set(prev.keys())
    curr_keys = set(curr.keys())
    new_files = sorted(curr_keys - prev_keys)
    deleted_files = sorted(prev_keys - curr_keys)
    changed_files = sorted(
        k for k in (curr_keys & prev_keys) if curr[k].sha256 != prev[k].sha256
    )
    return new_files, changed_files, deleted_files


def load_docs(curr: dict[str, SourceEntry]) -> list[dict]:
    docs: list[dict] = []
    for rel in sorted(curr.keys()):
        path = BASE_DIR / rel
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = text.splitlines()[0].replace("#", "").strip()
        docs.append({"path": rel, "title": title or path.stem, "text": text})
    return docs


def chunk_docs(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for idx, piece in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": f'{Path(doc["path"]).stem}_{idx:03d}',
                    "source": doc["path"],
                    "title": doc["title"],
                    "chunk_index": idx,
                    "text": piece,
                }
            )
    return chunks


def build_index(chunks: list[dict]) -> int:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(BASE_DIR / ".cache" / "fastembed"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    embedder = TextEmbedding(model_name=MODEL_NAME)
    vectors = np.array(list(embedder.embed([c["text"] for c in chunks])), dtype="float32")
    faiss.normalize_L2(vectors)
    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    np.save(INDEX_DIR / "vectors.npy", vectors)
    with (INDEX_DIR / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (INDEX_DIR / "index_build_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": MODEL_NAME,
                "embedding_dim": dim,
                "chunks_count": len(chunks),
                "knowledge_base_path": str(SOURCE_DIR.relative_to(BASE_DIR)) + "/",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return len(chunks)


def save_state(curr: dict[str, SourceEntry]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {"path": v.path, "sha256": v.sha256, "mtime": v.mtime}
            for _, v in sorted(curr.items())
        ],
    }
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_json_log(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with JSON_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    logger = setup_logging()
    started = datetime.now(timezone.utc)
    logger.info("update_index started: %s", started.isoformat())

    errors: list[str] = []
    try:
        prev = load_previous_state()
        curr = scan_source()
        new_files, changed_files, deleted_files = diff_state(prev, curr)
        docs = load_docs(curr)
        chunks = chunk_docs(docs)
        total_chunks = build_index(chunks)
        save_state(curr)

        changed_set = set(new_files + changed_files)
        changed_chunk_count = sum(1 for c in chunks if c["source"] in changed_set)
        index_size = (INDEX_DIR / "faiss.index").stat().st_size if (INDEX_DIR / "faiss.index").exists() else 0

        logger.info(
            "update_index done: new_files=%d changed_files=%d deleted_files=%d new_chunks=%d total_chunks=%d index_size_bytes=%d",
            len(new_files),
            len(changed_files),
            len(deleted_files),
            changed_chunk_count,
            total_chunks,
            index_size,
        )

        finished = datetime.now(timezone.utc)
        append_json_log(
            {
                "started_at_utc": started.isoformat(),
                "finished_at_utc": finished.isoformat(),
                "new_files": len(new_files),
                "changed_files": len(changed_files),
                "deleted_files": len(deleted_files),
                "new_chunks": changed_chunk_count,
                "total_chunks": total_chunks,
                "index_size_bytes": index_size,
                "errors": errors,
            }
        )
    except Exception as exc:
        errors.append(str(exc))
        finished = datetime.now(timezone.utc)
        logger.exception("update_index failed")
        append_json_log(
            {
                "started_at_utc": started.isoformat(),
                "finished_at_utc": finished.isoformat(),
                "new_files": 0,
                "changed_files": 0,
                "deleted_files": 0,
                "new_chunks": 0,
                "total_chunks": 0,
                "index_size_bytes": 0,
                "errors": errors,
            }
        )
        raise


if __name__ == "__main__":
    main()
