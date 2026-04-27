# CONCEPT 6: Strict RAG — LLM restricted to injected facts only
#
# This is the upgraded version of 5_vector_rag.py.
#
# WHAT CHANGED:
#   Program 5 used a loose system prompt ("never make up information") which still
#   allowed the LLM to answer from its training data if no relevant fact was injected,
#   or if it already knew the answer. This means:
#     → you can't trace WHERE the answer came from
#     → the LLM may silently ignore the injected fact and answer from memory
#     → in regulated domains (legal, medical, finance) this is unacceptable
#
#   This program uses a STRICT system prompt that:
#     → tells the LLM to ONLY use information from [Relevant info: ...] blocks
#     → tells the LLM to refuse if no [Relevant info] block is present
#     → explicitly bans the LLM from using its training data
#
#   Additionally, the distance threshold is tightened from 1.5 → 0.8 so only
#   genuinely close facts are injected (avoiding misleading irrelevant injections).
#
# WHEN TO USE THIS PATTERN:
#   - Customer support bots (answers must be traceable to official docs)
#   - Legal/medical/financial assistants (no hallucination allowed)
#   - Any app where you need full control over what the LLM can say
#
# HOW TO RUN:
#   pip install chromadb sentence-transformers
#   python3 6_strict_rag.py

import requests
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# STRICT SYSTEM PROMPT
#
# This is the key difference from program 5.
#
# Program 5 system prompt:
#   "Never make up information."
#   → LLM interprets this loosely: it still uses training data if it knows the answer.
#
# This system prompt:
#   → Explicitly names the ONLY allowed source: [Relevant info: ...] blocks
#   → Tells the LLM what to say when no fact is injected (refuse, don't guess)
#   → Bans training data explicitly
#
# This makes the LLM's behaviour predictable and auditable — every answer
# can be traced back to a specific fact in your knowledge base.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a SkillsApp support assistant.

STRICT RULES — follow these exactly:
1. ONLY answer using information provided in [Relevant info: ...] blocks in the user's message.
2. If the user's message has no [Relevant info: ...] block, respond with exactly:
   "I don't have information about that. Please contact support."
3. Do NOT use any knowledge from your training data — even if you think you know the answer.
4. Do NOT guess, infer, or expand beyond what the [Relevant info] block says.
5. Keep answers short, accurate, and friendly."""

# ---------------------------------------------------------------------------
# KNOWLEDGE BASE — same 5 facts as programs 4 and 5.
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    "SkillsApp pricing: Basic plan is $10/month, Pro plan is $30/month, Enterprise is custom.",
    "SkillsApp supports Python, Java, and JavaScript projects.",
    "SkillsApp integrates with GitHub, Bitbucket, and GitLab.",
    "To reset your password, go to Settings > Account > Reset Password.",
    "SkillsApp customer support is available Monday to Friday, 9am to 5pm EST.",
]

# ---------------------------------------------------------------------------
# DISTANCE THRESHOLD
#
# Program 5 used 1.5 — too lenient. It injected facts even when the question
# was unrelated, because ChromaDB always returns the nearest neighbor regardless
# of how far it is.
#
# This program uses 0.8 — stricter. Only inject a fact if it is genuinely close
# in meaning to the question. If nothing is close enough, inject nothing and let
# the strict system prompt handle the refusal.
#
# Tune this value based on your data:
#   lower threshold → fewer injections, more refusals, higher precision
#   higher threshold → more injections, fewer refusals, more false positives
# ---------------------------------------------------------------------------
DISTANCE_THRESHOLD = 0.8

# ---------------------------------------------------------------------------
# EMBEDDING FUNCTION + CHROMADB SETUP (same as program 5)
# Uses a separate ChromaDB collection ("skillsapp_facts_v6") so it doesn't
# conflict with the collection created by program 5.
# ---------------------------------------------------------------------------
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="skillsapp_facts_v6",   # separate collection from program 5
    embedding_function=embedding_fn
)

if collection.count() == 0:
    print("[Indexing knowledge base into ChromaDB for the first time...]")
    collection.add(
        documents=KNOWLEDGE_BASE,
        ids=[f"fact_{i}" for i in range(len(KNOWLEDGE_BASE))]
    )
    print("[Done. Facts stored in ./chroma_db/]\n")
else:
    print(f"[Loaded {collection.count()} facts from ChromaDB on disk]\n")


def find_relevant_fact(question: str) -> str:
    """Search ChromaDB for the closest fact to the question.

    Same vector search as program 5, but with a tighter distance threshold (0.8).
    This means only genuinely relevant facts get injected.

    If the closest fact is still too far away (distance > DISTANCE_THRESHOLD),
    we return None — no fact is injected, and the strict system prompt kicks in,
    making the bot refuse to answer rather than guess.

    This is the correct behaviour for a strict RAG system: unknown = refuse,
    not unknown = hallucinate.
    """
    results = collection.query(
        query_texts=[question],
        n_results=1
    )

    best_fact = results["documents"][0][0]
    distance = results["distances"][0][0]

    print(f"[Vector search — best match distance: {distance:.3f} | threshold: {DISTANCE_THRESHOLD}]")

    if distance > DISTANCE_THRESHOLD:
        # Nothing close enough — return None so no [Relevant info] block is injected.
        # The strict system prompt will then make the LLM refuse to answer.
        print("[No relevant fact found — bot will refuse to answer]")
        return None

    return best_fact


def ask(messages: list) -> str:
    """Send the full conversation history to the LLM and return its reply."""
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3",
        "stream": False,
        "messages": messages
    })
    return response.json()["message"]["content"]


# ---------------------------------------------------------------------------
# MAIN CHAT LOOP
# ---------------------------------------------------------------------------
print("=== SkillsApp Assistant (Strict RAG — facts only, no training data) ===")
print("Try: 'what does it cost?' → should answer from injected fact")
print("Try: 'who is the CEO?' → no fact matches, bot should refuse")
print("Type 'quit' to exit\n")

conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # --- STRICT VECTOR RAG STEP ---
    # Search for a relevant fact. With the tighter threshold, many questions
    # will return None — meaning no [Relevant info] block is added to the message.
    # The strict system prompt then instructs the LLM to refuse rather than guess.
    relevant_fact = find_relevant_fact(user_input)
    if relevant_fact:
        # Inject the fact. The system prompt tells the LLM this is its ONLY source.
        message = f"{user_input}\n\n[Relevant info: {relevant_fact}]"
        print(f"[RAG injected: {relevant_fact}]")
    else:
        # No fact injected. The strict system prompt will make the LLM say:
        # "I don't have information about that. Please contact support."
        # Unlike program 5, it will NOT fall back to training knowledge.
        message = user_input

    conversation_history.append({"role": "user", "content": message})

    reply = ask(conversation_history)

    conversation_history.append({"role": "assistant", "content": reply})

    print(f"Bot: {reply}\n")
