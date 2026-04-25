# Context Handling — Learn by doing

4 programs, each teaching one concept. Uses Ollama (local, free, no API key).

## Setup
```bash
# Install Ollama — https://ollama.com
ollama pull llama3

cd context-demo
python3 -m venv venv
source venv/bin/activate
pip install requests
```

## Run

```bash
python 1_no_context.py       # LLM forgets everything
python 2_with_context.py     # LLM remembers everything
python 3_sliding_window.py   # LLM remembers last 6 messages only
python 4_system_prompt_and_rag.py  # system prompt + RAG
```

## The 4 concepts

```
No context       → LLM gets 1 message, forgets everything
With context     → LLM gets full history, remembers everything
Sliding window   → LLM gets last N messages, old ones dropped
System prompt    → LLM gets instructions on who it is
RAG              → LLM gets relevant facts injected per question
```
