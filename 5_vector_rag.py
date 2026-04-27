# CONCEPT 5: System Prompt + RAG with Vector Embeddings
#
# This is the upgraded version of 4_system_prompt_and_rag.py.
#
# WHAT CHANGED:
#   Program 4 used keyword matching to find relevant facts:
#     → split words, count overlaps, pick the highest score.
#     → problem: "cost" won't match "pricing", "help" won't match "support".
#       It only works when the user uses the exact same words as the fact.
#
#   This program uses vector embeddings instead:
#     → every fact and every question is converted into a list of numbers (a vector)
#       that captures its *meaning*, not just its words.
#     → "how much does it cost?" and "what is the pricing?" produce similar vectors
#       even though they share no words — because they mean the same thing.
#     → we then find the fact whose vector is closest to the question vector (cosine similarity).
#
# TOOLS USED:
#   ChromaDB  — local vector database that stores and searches embeddings on disk.
#               Data persists in ./chroma_db/ so we only index facts once.
#   sentence-transformers — downloads and runs the 'all-MiniLM-L6-v2' model locally
#               to convert text into 384-dimensional vectors. No API key needed.
#
# HOW TO RUN:
#   pip install chromadb sentence-transformers
#   python 5_vector_rag.py

import requests
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — same as program 4.
# Placed as the first message (role="system") in every LLM call.
# It defines the LLM's persona and rules for the whole conversation.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful assistant for a software company called SkillsApp.
- Only answer questions about SkillsApp products.
- If you don't know the answer, say "I don't have that information."
- Keep answers short and friendly.
- Never make up information."""

# ---------------------------------------------------------------------------
# KNOWLEDGE BASE — the facts we want the LLM to be able to retrieve.
# In program 4 these were searched by keyword overlap.
# Here they get embedded as vectors and stored in ChromaDB.
# In a real app this could be thousands of docs loaded from files or a database.
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    "SkillsApp pricing: Basic plan is $10/month, Pro plan is $30/month, Enterprise is custom.",
    "SkillsApp supports Python, Java, and JavaScript projects.",
    "SkillsApp integrates with GitHub, Bitbucket, and GitLab.",
    "To reset your password, go to Settings > Account > Reset Password.",
    "SkillsApp customer support is available Monday to Friday, 9am to 5pm EST.",
]

# ---------------------------------------------------------------------------
# EMBEDDING FUNCTION
# SentenceTransformerEmbeddingFunction downloads 'all-MiniLM-L6-v2' on first run
# and caches it locally. After that it runs fully offline.
# It converts any text string into a 384-number vector that encodes its meaning.
# ChromaDB uses this function automatically when storing and querying documents.
# ---------------------------------------------------------------------------
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# CHROMADB SETUP
# PersistentClient stores the vector database on disk in ./chroma_db/
# so facts are only embedded once. On future runs, the collection is loaded
# from disk — no re-indexing needed.
# ---------------------------------------------------------------------------
client = chromadb.PersistentClient(path="./chroma_db")

# get_or_create_collection:
#   - first run  → creates a new empty collection called "skillsapp_facts"
#   - later runs → loads the existing collection from disk
# The embedding_function tells ChromaDB how to convert text to vectors.
collection = client.get_or_create_collection(
    name="skillsapp_facts",
    embedding_function=embedding_fn
)

# ---------------------------------------------------------------------------
# INDEX FACTS (only if the collection is empty)
# We assign each fact a unique string ID ("fact_0", "fact_1", ...).
# ChromaDB embeds each fact using the embedding_function and stores both
# the vector and the original text. On subsequent runs, this block is skipped.
# ---------------------------------------------------------------------------
if collection.count() == 0:
    print("[Indexing knowledge base into ChromaDB for the first time...]")
    collection.add(
        documents=KNOWLEDGE_BASE,
        ids=[f"fact_{i}" for i in range(len(KNOWLEDGE_BASE))]
        # ChromaDB automatically calls embedding_fn on each document here
    )
    print("[Done. Facts are now stored as vectors in ./chroma_db/]\n")
else:
    print(f"[Loaded {collection.count()} facts from ChromaDB on disk]\n")


def find_relevant_fact(question: str) -> str:
    """Search the vector database for the fact most semantically similar to the question.

    How it works:
    1. ChromaDB embeds the question using the same embedding_fn (all-MiniLM-L6-v2).
    2. It compares that vector against all stored fact vectors using cosine similarity.
       Cosine similarity measures the angle between two vectors — a score of 1.0 means
       identical meaning, 0.0 means completely unrelated.
    3. It returns the n_results closest facts ranked by similarity.

    Unlike keyword matching in program 4, this works even when the user's words
    don't appear in the fact. "What does it cost?" will match the pricing fact
    because "cost" and "pricing" are semantically close in vector space.

    Returns the best matching fact text, or None if the similarity is too low
    (distance > 1.5 means the vectors are far apart — likely no relevant fact).
    """
    results = collection.query(
        query_texts=[question],  # ChromaDB embeds this question automatically
        n_results=1              # return only the single closest fact
    )

    # results["documents"] is a list of lists: [[fact_text], ...]
    # results["distances"] is a list of lists: [[distance_score], ...]
    # Distance here is L2 (Euclidean) distance in embedding space.
    # Lower distance = more similar. We use 1.5 as a cutoff — tune this as needed.
    best_fact = results["documents"][0][0]
    distance = results["distances"][0][0]

    print(f"[Vector search — best match distance: {distance:.3f}]")

    if distance > 1.5:
        # Distance too large — the question is probably unrelated to our knowledge base
        return None

    return best_fact


def ask(messages: list) -> str:
    """Send the full conversation history to the LLM and return its reply.

    'messages' is a list of dicts: [{"role": "system"|"user"|"assistant", "content": "..."}]
    Sending the full list every turn gives the LLM memory of the conversation.
    """
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3",
        "stream": False,
        "messages": messages
    })
    return response.json()["message"]["content"]


# ---------------------------------------------------------------------------
# MAIN CHAT LOOP
# ---------------------------------------------------------------------------
print("=== SkillsApp Assistant (System Prompt + Vector RAG) ===")
print("Try: 'what does it cost?', 'which version control tools work with it?'")
print("Notice: questions don't need exact words from the facts to get a match.")
print("Type 'quit' to exit\n")

# Start conversation history with the system prompt.
# This stays at position 0 and is sent to the LLM on every turn.
conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # --- VECTOR RAG STEP ---
    # 1. Embed the user's question and find the closest fact in ChromaDB.
    # 2. If a relevant fact is found, append it to the user message.
    #    The LLM then sees both the question and the supporting fact,
    #    so it can answer accurately without hallucinating.
    relevant_fact = find_relevant_fact(user_input)
    if relevant_fact:
        # Augment the user message with the retrieved fact.
        # This is the "Augmented" part of Retrieval-Augmented Generation (RAG).
        message = f"{user_input}\n\n[Relevant info: {relevant_fact}]"
        print(f"[RAG injected: {relevant_fact}]")
    else:
        # No close match found — send the question as-is.
        # The system prompt rules still apply (LLM will say "I don't have that info").
        message = user_input

    # Add the (possibly augmented) user message to conversation history
    conversation_history.append({"role": "user", "content": message})

    # Send full history to LLM and get reply
    reply = ask(conversation_history)

    # Add LLM reply to history so future turns remember this exchange
    conversation_history.append({"role": "assistant", "content": reply})

    print(f"Bot: {reply}\n")
