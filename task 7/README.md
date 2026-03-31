# Задание 7: Оценка полноты базы знаний

## Что сделано

- Создана база с искусственными пробелами: `task 7/knowledge_base_gap/`
- Удалены 3 ключевые сущности из копии базы:
  - `01_ксарн_вэлгор.md`
  - `14_синт_поток.md`
  - `27_нексус_пустоты.md`
- Построен отдельный индекс: `task 7/artifacts/`
- Подготовлен golden set:
  - `task 7/golden_questions.json`
  - `task 7/golden_questions.txt`
- Реализован автотест и логирование:
  - `task 7/test_rag_bot.py`
  - `task 7/logs.jsonl`
  - `task 7/evaluation_report.md`
- Добавлена sequence-диаграмма:
  - `task 7/sequence_diagram.puml`

## Запуск

```bash
source .venv/bin/activate
python "task 7/prepare_gap_kb.py"
RAG_SOURCE_DIR="$(pwd)/task 7/knowledge_base_gap" RAG_INDEX_DIR="$(pwd)/task 7/artifacts" python "task 6/update_index.py"
python "task 7/test_rag_bot.py"
```
