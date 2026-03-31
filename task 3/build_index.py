import json
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "knowledge_base"
ARTIFACTS_DIR = BASE_DIR / "task 3" / "artifacts"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_URL = "https://huggingface.co/BAAI/bge-small-en-v1.5"


def load_documents(kb_dir: Path) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = text.splitlines()[0].replace("#", "").strip()
        docs.append(
            {
                "path": str(path.relative_to(BASE_DIR)),
                "title": title or path.stem,
                "text": text,
            }
        )
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for doc in docs:
        doc_chunks = splitter.split_text(doc["text"])
        for idx, chunk_text in enumerate(doc_chunks):
            chunks.append(
                {
                    "chunk_id": f'{Path(doc["path"]).stem}_{idx:03d}',
                    "source": doc["path"],
                    "title": doc["title"],
                    "chunk_index": idx,
                    "text": chunk_text,
                }
            )
    return chunks


def build_and_save_index(chunks: list[dict], artifacts_dir: Path) -> tuple[int, float]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    model = TextEmbedding(model_name=MODEL_NAME)
    vectors = list(model.embed([c["text"] for c in chunks]))
    embeddings = np.array(vectors, dtype="float32")
    faiss.normalize_L2(embeddings)

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(artifacts_dir / "faiss.index"))
    np.save(artifacts_dir / "vectors.npy", embeddings)
    with (artifacts_dir / "faiss_store.pkl").open("wb") as f:
        pickle.dump(
            {
                "embedding_model": MODEL_NAME,
                "embedding_dim": dim,
                "vectors_path": "task 3/artifacts/vectors.npy",
            },
            f,
        )

    with (artifacts_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed_sec = time.perf_counter() - started
    with (artifacts_dir / "index_build_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "embedding_model": MODEL_NAME,
                "embedding_model_url": MODEL_URL,
                "embedding_dim": dim,
                "chunks_count": len(chunks),
                "knowledge_base_path": "knowledge_base/",
                "elapsed_seconds": round(elapsed_sec, 3),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return dim, elapsed_sec


def run_sample_query(artifacts_dir: Path, query: str, top_k: int = 3) -> dict:
    index = faiss.read_index(str(artifacts_dir / "faiss.index"))
    rows: list[dict] = []
    with (artifacts_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    model = TextEmbedding(model_name=MODEL_NAME)
    q_vec = np.array(list(model.embed([query])), dtype="float32")
    faiss.normalize_L2(q_vec)
    scores, ids = index.search(q_vec, top_k)

    matches = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        item = rows[int(idx)]
        matches.append(
            {
                "score": float(score),
                "chunk_id": item["chunk_id"],
                "source": item["source"],
                "title": item["title"],
                "text_preview": item["text"][:240],
            }
        )

    payload = {"query": query, "top_k": top_k, "matches": matches}
    with (artifacts_dir / "sample_query_result.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    docs = load_documents(KB_DIR)
    chunks = chunk_documents(docs)
    dim, elapsed = build_and_save_index(chunks, ARTIFACTS_DIR)
    sample = run_sample_query(
        ARTIFACTS_DIR,
        query="Кто такой Ксарн Вэлгор и какую роль он играл в Империи Штиля?",
        top_k=3,
    )

    print("Index built successfully")
    print(f"Documents: {len(docs)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding dim: {dim}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print("Sample query saved to task 3/artifacts/sample_query_result.json")
    print(f"Top match source: {sample['matches'][0]['source'] if sample['matches'] else 'n/a'}")


if __name__ == "__main__":
    main()
