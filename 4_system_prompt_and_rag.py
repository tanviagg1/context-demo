# CONCEPT 4: System Prompt + Simple RAG
#
# System prompt: tells the LLM WHO it is and HOW to behave.
# RAG: instead of sending a giant document, we search for relevant bits
#      and only send those bits.

import requests

SYSTEM_PROMPT = """You are a helpful assistant for a software company called SkillsApp.
- Only answer questions about SkillsApp products.
- If you don't know the answer, say "I don't have that information."
- Keep answers short and friendly.
- Never make up information."""

KNOWLEDGE_BASE = [
    "SkillsApp pricing: Basic plan is $10/month, Pro plan is $30/month, Enterprise is custom.",
    "SkillsApp supports Python, Java, and JavaScript projects.",
    "SkillsApp integrates with GitHub, Bitbucket, and GitLab.",
    "To reset your password, go to Settings > Account > Reset Password.",
    "SkillsApp customer support is available Monday to Friday, 9am to 5pm EST.",
]

def find_relevant_fact(question: str) -> str:
    """Simple RAG: find the most relevant fact using keyword matching."""
    question_words = set(question.lower().split())
    best_fact, best_score = None, 0
    for fact in KNOWLEDGE_BASE:
        score = len(question_words & set(fact.lower().split()))
        if score > best_score:
            best_score = score
            best_fact = fact
    return best_fact if best_score > 0 else None

def ask(messages: list) -> str:
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3",
        "stream": False,
        "messages": messages
    })
    return response.json()["message"]["content"]

print("=== SkillsApp Assistant (System Prompt + RAG) ===")
print("Try: 'how much does it cost?' or 'what languages do you support?'")
print("Type 'quit' to exit\n")

conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}  # system prompt as first message
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # RAG: find relevant fact and inject it
    relevant_fact = find_relevant_fact(user_input)
    if relevant_fact:
        message = f"{user_input}\n\n[Relevant info: {relevant_fact}]"
        print(f"[RAG injected: {relevant_fact}]")
    else:
        message = user_input

    conversation_history.append({"role": "user", "content": message})

    reply = ask(conversation_history)

    conversation_history.append({"role": "assistant", "content": reply})

    print(f"Bot: {reply}\n")
