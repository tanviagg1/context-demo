# Context Handling — Learn by doing

5 programs, each teaching one concept. Uses Ollama (local, free, no API key).

## Setup
```bash
# Install Ollama — https://ollama.com
ollama pull llama3

cd context-demo
python3 -m venv venv
source venv/bin/activate
pip install requests

# For program 5 (vector RAG) only
pip install chromadb sentence-transformers

# Start Ollama in background
ollama serve
```

---

## Program 1 — No Context

**File:** `1_no_context.py`

**Run:**
```bash
python 1_no_context.py
```

**What it teaches:** The problem. Every message is sent to the LLM alone — no history. The LLM forgets everything immediately.

**How it works:**
```
You type "my name is Alice"
    ↓
LLM gets: [ {user: "my name is Alice"} ]   ← only this message

You type "what is my name?"
    ↓
LLM gets: [ {user: "what is my name?"} ]   ← only this message, forgets Alice
    ↓
Bot: "I don't know your name" ❌
```

**Try this:**
1. Type `my name is Alice`
2. Type `what is my name?`
3. Bot will not know — that's the point

---

## Program 2 — With Context

**File:** `2_with_context.py`

**Run:**
```bash
python 2_with_context.py
```

**What it teaches:** The fix. We keep a list called `conversation_history` and send the full list to the LLM every single time. Now it remembers everything.

**How it works:**
```
You type "my name is Alice"
    ↓
Step 1: Add to history → history = [ {user: "my name is Alice"} ]
Step 2: Send full history to LLM
Step 3: Add reply → history = [ {user: ...}, {assistant: ...} ]

You type "what is my name?"
    ↓
Step 1: Add to history → history = [ msg1, msg2, {user: "what is my name?"} ]
Step 2: Send full history — LLM sees ALL 3 messages
    ↓
Bot: "Your name is Alice" ✅
```

**Try this:**
1. Type `my name is Alice`
2. Type `I am 30 years old`
3. Type `what is my name and age?` — bot will know both ✅

Watch `[History size: X]` grow with every message.

---

## Program 3 — Sliding Window

**File:** `3_sliding_window.py`

**Run:**
```bash
python 3_sliding_window.py
```

**What it teaches:** Context has limits. Sending the full history forever gets expensive and eventually hits the token limit. The fix is to only send the last N messages — older ones get dropped.

**How it works:**
```
Full history:   [ msg1, msg2, msg3, msg4, msg5, msg6, msg7, msg8 ]
                                                     ↑
Window (last 6):                  [ msg3, msg4, msg5, msg6, msg7, msg8 ]
                                                               ↑
LLM only sees these 6 — msg1 and msg2 are dropped
```

**Try this:**
1. Tell the bot your name (e.g. `my name is Alice`)
2. Have 4+ more exchanges about other things
3. Ask `what is my name?` — bot will have forgotten (it was pushed out of the window)

Watch `[Total history: X | Sending to LLM: Y]` — Total grows, Sending stays at 6.

---

## Program 4 — System Prompt + RAG

**File:** `4_system_prompt_and_rag.py`

**Run:**
```bash
python 4_system_prompt_and_rag.py
```

**What it teaches:** Two real-world techniques:
- **System prompt** — tells the LLM who it is and how to behave, loaded once at the start
- **RAG (Retrieval Augmented Generation)** — instead of sending a giant document, search for the relevant bit and inject only that into the context

**How the system prompt works:**
```
Every message sent to LLM looks like:

[ {system: "You are a SkillsApp assistant. Only answer about SkillsApp..."},
  {user: "how much does it cost?"},
  {assistant: "Basic is $10/month..."},
  {user: "what languages do you support?"} ]
    ↑
System prompt shapes ALL responses
```

**How RAG works:**
```
You ask: "how much does it cost?"
    ↓
Search knowledge base for relevant fact
    ↓
Found: "Basic plan $10/month, Pro $30/month..."
    ↓
Inject into message:
  "how much does it cost?
   [Relevant info: Basic plan $10/month, Pro $30/month...]"
    ↓
LLM answers accurately from injected fact ✅
```

**Try this:**
1. `how much does it cost?` — RAG finds the pricing fact
2. `what languages do you support?` — RAG finds the languages fact
3. `what is the weather today?` — bot refuses (outside its role)

Watch `[RAG injected: ...]` to see which fact was found.

**Limitation:** Uses keyword overlap — "cost" won't match "pricing". See Program 5 for the fix.

---

## Program 5 — Vector RAG

**File:** `5_vector_rag.py`

**Run:**
```bash
python 5_vector_rag.py
```

**What it teaches:** The upgraded version of Program 4. Instead of matching by exact words, it converts every fact and every question into a vector (a list of numbers) that captures *meaning*. Two phrases can share no words but still match closely if they mean the same thing.

**How vector search works:**
```
You ask: "what does it cost?"
    ↓
Embed question → [0.23, -0.81, 0.44, ...]   ← 384 numbers representing meaning
    ↓
Compare against stored fact vectors in ChromaDB
    ↓
"SkillsApp pricing: Basic $10/month..."  → distance: 0.31  ← closest match ✅
"SkillsApp supports Python, Java..."     → distance: 1.12
    ↓
Inject closest fact into message → LLM answers accurately
```

**What's different from Program 4:**

| | Program 4 | Program 5 |
|---|---|---|
| Search | Keyword overlap | Cosine similarity (vector distance) |
| "cost" matches "pricing"? | No | Yes |
| Database | In-memory list | ChromaDB on disk (`./chroma_db/`) |
| Persists between runs? | No | Yes — indexed once, reloaded each time |

**Try this:**
1. `what does it cost?` — matches pricing fact even without the word "pricing"
2. `which version control tools work with it?` — matches the GitHub/Bitbucket fact
3. `what is the weather today?` — no close match, bot refuses (system prompt still applies)

Watch `[Vector search — best match distance: X.XXX]` — lower = closer match.

> **First run:** downloads the `all-MiniLM-L6-v2` embedding model (~80MB) and indexes facts into `./chroma_db/`. Subsequent runs load instantly from disk.

---

## The 5 concepts at a glance

```
Program 1 — No context       → LLM gets 1 message, forgets everything
Program 2 — With context     → LLM gets full history, remembers everything
Program 3 — Sliding window   → LLM gets last N messages, old ones dropped
Program 4 — System prompt    → LLM gets instructions on who it is
            + keyword RAG    → relevant facts injected by word matching
Program 5 — System prompt    → same as 4
            + vector RAG     → facts injected by semantic similarity (ChromaDB)
```
