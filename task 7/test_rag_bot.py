import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "task 7" / "artifacts"
OUT_DIR = BASE_DIR / "task 7"
QUESTIONS_PATH = OUT_DIR / "golden_questions.json"
LOG_PATH = OUT_DIR / "logs.jsonl"
REPORT_PATH = OUT_DIR / "evaluation_report.md"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MIN_SCORE = 0.74


def load_rows() -> list[dict]:
    rows = []
    with (INDEX_DIR / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def retrieve(query: str, embedder: TextEmbedding, index, rows: list[dict], top_k: int = 3):
    q_vec = np.array(list(embedder.embed([query])), dtype="float32")
    faiss.normalize_L2(q_vec)
    scores, ids = index.search(q_vec, top_k)
    found = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        row = rows[int(idx)]
        found.append(
            {
                "score": float(score),
                "source": row["source"],
                "text": row["text"],
            }
        )
    return found


def build_answer(matches: list[dict]) -> str:
    if not matches or matches[0]["score"] < MIN_SCORE:
        return "Я не знаю."
    first = matches[0]["text"].splitlines()
    if len(first) > 1:
        return first[1]
    return matches[0]["text"]


def is_success(answer: str, expected_found: bool, expected_keywords: list[str]) -> bool:
    if not expected_found:
        return answer.strip().lower() == "я не знаю."
    if answer.strip().lower() == "я не знаю.":
        return False
    if expected_keywords:
        lowered = answer.lower()
        return any(k.lower() in lowered for k in expected_keywords)
    return len(answer.strip()) > 10


def main() -> None:
    embedder = TextEmbedding(model_name=MODEL_NAME)
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    rows = load_rows()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    results = []
    with LOG_PATH.open("w", encoding="utf-8") as logf:
        for q in questions:
            matches = retrieve(q["query"], embedder, index, rows, top_k=3)
            answer = build_answer(matches)
            status = is_success(answer, q["expected_found"], q["expected_keywords"])
            entry = {
                "query": q["query"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chunks_found": len(matches) > 0,
                "answer_length": len(answer),
                "successful_answer": status,
                "sources": [{"source": m["source"], "score": round(m["score"], 3)} for m in matches],
                "answer": answer,
                "expected_found": q["expected_found"],
                "topic": q["topic"],
            }
            logf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            results.append(entry)

    total = len(results)
    ok = sum(1 for r in results if r["successful_answer"])
    failed = total - ok

    topic_fails: dict[str, int] = {}
    for r in results:
        if not r["successful_answer"]:
            topic_fails[r["topic"]] = topic_fails.get(r["topic"], 0) + 1

    report = [
        "# Task 7 Evaluation Report",
        "",
        f"- Всего вопросов: {total}",
        f"- Успешных ответов: {ok}",
        f"- Неуспешных ответов: {failed}",
        "",
        "## Темы с наибольшими пробелами",
    ]
    if topic_fails:
        for t, c in sorted(topic_fails.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {t}: {c}")
    else:
        report.append("- Пробелов не обнаружено")

    report += [
        "",
        "## Рекомендации",
        "- Вернуть удаленные сущности в базу: Ксарн Вэлгор, Синт-Поток, Нексус Пустоты.",
        "- Добавить документы по внешним терминам (Ти'лора, HyperRelay), если они важны для пользователей.",
        "- Ввести регулярный регрессионный прогон golden-набора после каждого обновления индекса.",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"Done: logs -> {LOG_PATH}, report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
