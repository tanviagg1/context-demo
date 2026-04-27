# CONCEPT 4: System Prompt + Simple RAG (keyword matching)
#
# System prompt: tells the LLM WHO it is and HOW to behave.
# RAG: instead of sending a giant document, we search for relevant bits
#      and only send those bits.
#
# LIMITATION OF THIS APPROACH:
# Retrieval here uses keyword overlap — it only matches if the user's words
# appear literally in the fact. "cost" won't match "pricing". "help" won't
# match "support". See 5_vector_rag.py for the upgraded version that uses
# vector embeddings (ChromaDB + sentence-transformers) to match by *meaning*
# instead of exact words.
#
# How this program works:
# 1. A system prompt is set once at the start — it defines the LLM's persona and rules.
# 2. A hardcoded knowledge base holds facts about SkillsApp.
# 3. For every user question, we search the knowledge base for the most relevant fact
#    (using simple keyword overlap — no embeddings needed here).
# 4. If a relevant fact is found, it's injected into the user message before sending to the LLM.
# 5. The full conversation history (system prompt + all turns) is sent each time,
#    so the LLM remembers previous messages.

import requests

# The system prompt is the LLM's "instruction manual".
# It's always the first message in the conversation with role="system".
# It shapes how the LLM behaves for the entire conversation —
# e.g. only answer about SkillsApp, never make up facts.
SYSTEM_PROMPT = """You are a helpful assistant for a software company called SkillsApp.
- Only answer questions about SkillsApp products.
- If you don't know the answer, say "I don't have that information."
- Keep answers short and friendly.
- Never make up information."""

# The knowledge base is our mini "database" of facts.
# In a real RAG system, this would be a vector database (e.g. ChromaDB)
# with thousands of documents stored as embeddings.
# Here we keep it simple: just a list of plain-text strings.
KNOWLEDGE_BASE = [
    "SkillsApp pricing: Basic plan is $10/month, Pro plan is $30/month, Enterprise is custom.",
    "SkillsApp supports Python, Java, and JavaScript projects.",
    "SkillsApp integrates with GitHub, Bitbucket, and GitLab.",
    "To reset your password, go to Settings > Account > Reset Password.",
    "SkillsApp customer support is available Monday to Friday, 9am to 5pm EST.",
]

def find_relevant_fact(question: str) -> str:
    """Simple RAG: find the most relevant fact using keyword matching.

    How it works:
    - Split the question into individual words (a 'set' removes duplicates).
    - For each fact in the knowledge base, count how many question words appear in that fact.
    - Return the fact with the highest overlap score.
    - If no words match at all, return None (no relevant fact found).

    In production RAG, you'd use vector embeddings + cosine similarity instead of word counts,
    which handles synonyms and meaning — not just exact word matches.
    """
    question_words = set(question.lower().split())
    best_fact, best_score = None, 0
    for fact in KNOWLEDGE_BASE:
        # Count words that appear in both the question and this fact
        score = len(question_words & set(fact.lower().split()))
        if score > best_score:
            best_score = score
            best_fact = fact
    return best_fact if best_score > 0 else None

def ask(messages: list) -> str:
    """Send the full conversation history to the LLM and return its reply.

    'messages' is a list of dicts like: [{"role": "system"|"user"|"assistant", "content": "..."}]
    Sending the full history every turn is what gives the LLM memory of the conversation.
    """
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3",
        "stream": False,
        "messages": messages
    })
    return response.json()["message"]["content"]

print("=== SkillsApp Assistant (System Prompt + RAG) ===")
print("Try: 'how much does it cost?' or 'what languages do you support?'")
print("Type 'quit' to exit\n")

# Initialize conversation history with the system prompt as the very first message.
# Every subsequent message (user + assistant) gets appended to this list.
# The system prompt stays at position 0 and is sent to the LLM on every turn.
conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # --- RAG step ---
    # Search the knowledge base for a fact relevant to the user's question.
    relevant_fact = find_relevant_fact(user_input)
    if relevant_fact:
        # Inject the relevant fact directly into the user message.
        # The LLM sees the question + the fact together, so it can answer accurately
        # without having to "know" SkillsApp details from its training data.
        message = f"{user_input}\n\n[Relevant info: {relevant_fact}]"
        print(f"[RAG injected: {relevant_fact}]")  # show what was injected, for learning purposes
    else:
        # No relevant fact found — send the question as-is.
        # The system prompt rules still apply (e.g. "say I don't have that info").
        message = user_input

    # Add the (possibly augmented) user message to history
    conversation_history.append({"role": "user", "content": message})

    # Send the full history to the LLM and get a reply
    reply = ask(conversation_history)

    # Add the LLM's reply to history so future turns remember this exchange
    conversation_history.append({"role": "assistant", "content": reply})

    print(f"Bot: {reply}\n")
