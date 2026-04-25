# Project: context-demo

A learning project that teaches LLM context handling through 4 progressive programs.

## Stack
- Python 3.14
- Ollama (local LLM — llama3)
- requests library

## How to run
```bash
cd /Users/tanviagarwal/Documents/Projects/context-demo
source venv/bin/activate
ollama serve   # make sure Ollama is running
python <program_name>.py
```

## Programs

| File | Concept | What it teaches |
|---|---|---|
| `1_no_context.py` | No context | LLM forgets everything — each message sent alone |
| `2_with_context.py` | Full context | Full conversation history sent every turn |
| `3_sliding_window.py` | Sliding window | Only last 6 messages kept — older ones dropped |
| `4_system_prompt_and_rag.py` | System prompt + RAG | System prompt shapes behaviour, RAG injects relevant facts |

## Key concepts

**Context window** — fixed limit on how much text the LLM can see at once.

**conversation_history** — a list of messages `[{role, content}, ...]` sent to the LLM each turn.

**Sliding window** — `conversation_history[-MAX_MESSAGES:]` keeps context size under control.

**System prompt** — first message with `role: system`, shapes all LLM responses.

**RAG** — search a knowledge base for relevant facts, inject them into the user message before sending to LLM.

## Knowledge base (program 4)
5 hardcoded facts about SkillsApp in `KNOWLEDGE_BASE` list.
Matched using keyword overlap. In real apps, use vector embeddings + ChromaDB.

## GitHub
https://github.com/tanviagg1/context-demo
