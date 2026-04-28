# CONCEPT 7: Versioned RAG — managing embedding versions safely
#
# This is the upgraded version of 6_strict_rag.py.
#
# THE PROBLEM WITH PROGRAMS 5 AND 6:
#   The collection name was hardcoded ("skillsapp_facts_v6").
#   If you:
#     → switch to a better embedding model
#     → add/remove/edit facts in the knowledge base
#     → change how text is chunked
#   ...the old vectors in ChromaDB are now INCOMPATIBLE with the new setup.
#   Vectors from model A cannot be compared against vectors from model B.
#   But the code would silently keep using the old collection — wrong results, no error.
#
# THE FIX — VERSION YOUR COLLECTION NAME:
#   The collection name is built from KB_VERSION + EMBEDDING_MODEL.
#   When either changes, a new collection is automatically created and re-indexed.
#   The old collection stays on disk untouched — instant rollback if needed.
#
# VERSIONING RULES:
#   Bump KB_VERSION when:  facts change, facts are added/removed, chunking changes
#   Change EMBEDDING_MODEL when: you switch to a different sentence-transformer model
#   Either change → new collection name → fresh index automatically
#
# HOW TO RUN:
#   pip install chromadb sentence-transformers
#   python3 7_versioned_rag.py
#
# TO INSPECT ALL VERSIONS ON DISK:
#   python3 7_versioned_rag.py --list-versions
#
# TO DELETE AN OLD VERSION:
#   python3 7_versioned_rag.py --delete-version skillsapp_v1_all-MiniLM-L6-v2

import sys
import requests
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# VERSION CONFIGURATION — the only place you need to change when upgrading
#
# KB_VERSION:      bump this (v1 → v2) whenever the knowledge base content changes.
#                  A new collection is created and facts are re-indexed automatically.
#
# EMBEDDING_MODEL: the sentence-transformers model used to convert text to vectors.
#                  Changing this also creates a new collection, since vectors from
#                  different models are incompatible and cannot be compared.
#
# COLLECTION_NAME: auto-built from both — encodes exactly what's inside the collection.
#                  e.g. "skillsapp_v1_all-MiniLM-L6-v2"
#                  You can read the name and know the model and version instantly.
# ---------------------------------------------------------------------------
KB_VERSION      = "v1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = f"skillsapp_{KB_VERSION}_{EMBEDDING_MODEL.replace('/', '_')}"

# ---------------------------------------------------------------------------
# STRICT SYSTEM PROMPT — same as program 6.
# LLM is restricted to only use [Relevant info: ...] blocks.
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
# KNOWLEDGE BASE
#
# This is KB_VERSION = "v1". When you change these facts, bump KB_VERSION to "v2".
# The new collection will be created and re-indexed with the updated facts.
# The v1 collection stays on disk — you can roll back by changing KB_VERSION back.
#
# Example upgrade scenario:
#   KB_VERSION = "v2"   ← bump
#   KNOWLEDGE_BASE = [  ← updated facts
#       "SkillsApp pricing: Basic $12/month, Pro $35/month...",   ← price changed
#       ... same other facts ...
#       "SkillsApp now supports Ruby projects.",                   ← new fact added
#   ]
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    "SkillsApp pricing: Basic plan is $10/month, Pro plan is $30/month, Enterprise is custom.",
    "SkillsApp supports Python, Java, and JavaScript projects.",
    "SkillsApp integrates with GitHub, Bitbucket, and GitLab.",
    "To reset your password, go to Settings > Account > Reset Password.",
    "SkillsApp customer support is available Monday to Friday, 9am to 5pm EST.",
]

# ---------------------------------------------------------------------------
# DISTANCE THRESHOLD — same as program 6.
# Only inject a fact if it's genuinely close (distance < 0.8).
# Tune lower for higher precision, higher for more recall.
# ---------------------------------------------------------------------------
DISTANCE_THRESHOLD = 0.8

# ---------------------------------------------------------------------------
# CHROMADB + EMBEDDING SETUP
# ---------------------------------------------------------------------------
embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
client = chromadb.PersistentClient(path="./chroma_db")


def list_versions():
    """Print all versioned collections currently stored on disk.

    Useful to see which versions exist before deleting old ones.
    Each collection name encodes the KB version and model used.
    """
    collections = client.list_collections()
    if not collections:
        print("No collections found in ./chroma_db/")
        return
    print("Collections on disk:")
    for col in collections:
        c = client.get_collection(col.name)
        print(f"  {col.name}  ({c.count()} facts)")


def delete_version(name: str):
    """Delete a specific versioned collection from disk.

    Use this to clean up old versions once you're confident the new version is stable.
    This is irreversible — the vectors are deleted and cannot be recovered.
    The knowledge base text (KNOWLEDGE_BASE list above) is always safe in source code.
    """
    try:
        client.delete_collection(name)
        print(f"Deleted collection: {name}")
    except Exception as e:
        print(f"Could not delete '{name}': {e}")


def load_or_create_collection():
    """Load the versioned collection from disk, or create and index it if it doesn't exist.

    This is where versioning pays off:
    - If COLLECTION_NAME already exists on disk → load it instantly (no re-indexing)
    - If it doesn't exist (new KB_VERSION or new EMBEDDING_MODEL) → create it and index facts

    The version is entirely encoded in COLLECTION_NAME, so:
    - Changing KB_VERSION creates a new collection automatically
    - Changing EMBEDDING_MODEL creates a new collection automatically
    - Old collections remain on disk for rollback
    """
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    if collection.count() == 0:
        # New version — index the knowledge base facts for the first time
        print(f"[New version detected: {COLLECTION_NAME}]")
        print(f"[Indexing {len(KNOWLEDGE_BASE)} facts with model '{EMBEDDING_MODEL}'...]")
        collection.add(
            documents=KNOWLEDGE_BASE,
            ids=[f"fact_{i}" for i in range(len(KNOWLEDGE_BASE))]
        )
        print(f"[Done. Stored in ./chroma_db/ as '{COLLECTION_NAME}']\n")
    else:
        # Existing version — load from disk, no re-indexing needed
        print(f"[Loaded existing version: {COLLECTION_NAME} ({collection.count()} facts)]\n")

    return collection


def find_relevant_fact(collection, question: str) -> str:
    """Search the versioned collection for the fact closest in meaning to the question.

    Same semantic search as programs 5 and 6. The key difference here is that
    'collection' is passed in — it always points to the currently active version,
    so search results are always consistent with the active KB_VERSION and model.
    """
    results = collection.query(
        query_texts=[question],
        n_results=1
    )

    best_fact = results["documents"][0][0]
    distance = results["distances"][0][0]

    print(f"[Vector search — version: {COLLECTION_NAME} | distance: {distance:.3f} | threshold: {DISTANCE_THRESHOLD}]")

    if distance > DISTANCE_THRESHOLD:
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
# CLI UTILITIES
# Run with --list-versions or --delete-version <name> for DB management.
# Otherwise, start the chat loop.
# ---------------------------------------------------------------------------
if "--list-versions" in sys.argv:
    list_versions()
    sys.exit(0)

if "--delete-version" in sys.argv:
    idx = sys.argv.index("--delete-version")
    if idx + 1 < len(sys.argv):
        delete_version(sys.argv[idx + 1])
    else:
        print("Usage: python3 7_versioned_rag.py --delete-version <collection_name>")
    sys.exit(0)

# ---------------------------------------------------------------------------
# MAIN CHAT LOOP
# ---------------------------------------------------------------------------
collection = load_or_create_collection()

print(f"=== SkillsApp Assistant (Versioned Strict RAG) ===")
print(f"Active version: {COLLECTION_NAME}")
print("Try: 'what does it cost?' → fact injected, bot answers")
print("Try: 'who is the CEO?' → no match, bot refuses")
print("Type 'quit' to exit\n")

conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    relevant_fact = find_relevant_fact(collection, user_input)
    if relevant_fact:
        message = f"{user_input}\n\n[Relevant info: {relevant_fact}]"
        print(f"[RAG injected: {relevant_fact}]")
    else:
        message = user_input

    conversation_history.append({"role": "user", "content": message})

    reply = ask(conversation_history)

    conversation_history.append({"role": "assistant", "content": reply})

    print(f"Bot: {reply}\n")
