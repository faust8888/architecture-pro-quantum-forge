# Task 4: RAG-бот

## Что реализовано

- Загрузка FAISS-индекса из `task 3/artifacts/faiss.index`
- Прием пользовательского запроса (REPL)
- Эмбеддинг запроса той же моделью (`BAAI/bge-small-en-v1.5`)
- Поиск релевантных чанков в векторной базе
- Формирование промпта с контекстом
- Few-shot примеры внутри промпта
- Chain-of-Thought инструкция в system prompt
- Генерация ответа:
  - через OpenAI API (если задан `OPENAI_API_KEY`)
  - либо локальный fallback (если ключа нет)
- "Я не знаю" при низкой уверенности retrieval

## Запуск

```bash
source .venv/bin/activate
python "task 4/rag_bot.py"
```

Опционально для OpenAI:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
```
