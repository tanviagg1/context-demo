# CONCEPT 4: System Prompt + Simple RAG
#
# System prompt: tells the LLM WHO it is and HOW to behave.
# RAG: instead of sending a giant document, we search for relevant bits
#      and only send those bits.
#
# This example has a tiny "knowledge base" of 5 facts.
# When you ask a question, we find the most relevant fact
# and inject it into the context.

import anthropic

client = anthropic.Anthropic()

# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────
# This shapes how the LLM behaves. Loaded once, sent with every message.

SYSTEM_PROMPT = """You are a helpful assistant for a software company called SkillsApp.
- Only answer questions about SkillsApp products.
- If you don't know the answer, say "I don't have that information."
- Keep answers short and friendly.
- Never make up information."""


# ── KNOWLEDGE BASE (tiny RAG) ───────────────────────────────────────────────
# In real apps this would be a vector database with thousands of documents.
# Here we use a simple list for clarity.

KNOWLEDGE_BASE = [
    "SkillsApp pricing: Basic plan is $10/month, Pro plan is $30/month, Enterprise is custom.",
    "SkillsApp supports Python, Java, and JavaScript projects.",
    "SkillsApp integrates with GitHub, Bitbucket, and GitLab.",
    "To reset your password, go to Settings > Account > Reset Password.",
    "SkillsApp customer support is available Monday to Friday, 9am to 5pm EST.",
]


def find_relevant_fact(question: str) -> str:
    """
    Simple RAG: find the fact most relevant to the question.
    Real apps use embeddings + vector search. We use keyword matching here.
    """
    question_words = set(question.lower().split())

    best_fact = None
    best_score = 0

    for fact in KNOWLEDGE_BASE:
        fact_words = set(fact.lower().split())
        # Score = number of words in common
        score = len(question_words & fact_words)
        if score > best_score:
            best_score = score
            best_fact = fact

    return best_fact if best_score > 0 else None


# ── CHATBOT ─────────────────────────────────────────────────────────────────

print("=== SkillsApp Assistant (System Prompt + RAG) ===")
print("Try asking: 'how much does it cost?' or 'what languages do you support?'")
print("Type 'quit' to exit\n")

conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # Step 1: RAG — find relevant fact from knowledge base
    relevant_fact = find_relevant_fact(user_input)

    # Step 2: If we found a relevant fact, inject it into the user message
    if relevant_fact:
        message_with_context = f"{user_input}\n\n[Relevant info: {relevant_fact}]"
        print(f"[RAG injected: {relevant_fact}]")
    else:
        message_with_context = user_input

    conversation_history.append({
        "role": "user",
        "content": message_with_context
    })

    # Step 3: Send system prompt + history to LLM
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=SYSTEM_PROMPT,        # ← system prompt here
        messages=conversation_history
    )

    reply = response.content[0].text

    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    print(f"Bot: {reply}\n")
