# Задание 3: Индексация базы знаний

- Модель эмбеддингов: `BAAI/bge-small-en-v1.5`
- Ссылка на модель: https://huggingface.co/BAAI/bge-small-en-v1.5
- Размер эмбеддингов: `384`
- База знаний: `knowledge_base/` (30 документов)

## Артефакты

- `task 3/artifacts/faiss.index` — FAISS-индекс
- `task 3/artifacts/faiss_store.pkl` — сериализованный индекс (альтернативный формат)
- `task 3/artifacts/vectors.npy` — матрица эмбеддингов чанков
- `task 3/artifacts/chunks.jsonl` — чанки с метаданными (`source`, `title`, `chunk_id`, `chunk_index`)
- `task 3/artifacts/index_build_meta.json` — метаданные сборки индекса
- `task 3/artifacts/sample_query_result.json` — пример поискового запроса и найденные чанки
