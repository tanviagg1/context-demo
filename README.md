# Context Handling — Learn by doing

4 programs, each teaching one concept. Run them in order.

## Setup
```bash
cd context-demo
source venv/bin/activate
export ANTHROPIC_API_KEY=your_key_here
```

## Programs

### 1. No context — the problem
```bash
python 1_no_context.py
```
Say "my name is Alice", then ask "what is my name?"
The bot forgets immediately. This is the problem we solve.

### 2. With context — the fix
```bash
python 2_with_context.py
```
Same test. Now the bot remembers. See history size grow each turn.

### 3. Sliding window — handling limits
```bash
python 3_sliding_window.py
```
Chat for more than 3 exchanges. Early messages get dropped.
Shows how to keep context size under control.

### 4. System prompt + RAG — real world
```bash
python 4_system_prompt_and_rag.py
```
Ask "how much does it cost?" or "what languages do you support?"
Watch [RAG injected: ...] show which fact was found and added to context.

## The 4 concepts at a glance

```
No context       → LLM gets 1 message, forgets everything
With context     → LLM gets full history, remembers everything
Sliding window   → LLM gets last N messages, old ones dropped
System prompt    → LLM gets instructions on who it is
RAG              → LLM gets relevant facts injected per question
```
