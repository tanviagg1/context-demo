# CONCEPT 3: Sliding Window
# Problem: if you chat for 1000 messages, the context gets too big.
# Fix: only keep the last N messages. Old ones are dropped.

import requests

MAX_MESSAGES = 6  # only keep the last 6 messages (3 exchanges)

def ask(messages: list) -> str:
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3",
        "stream": False,
        "messages": messages
    })
    return response.json()["message"]["content"]

print("=== Chatbot with SLIDING WINDOW (last 6 messages only) ===")
print(f"Only the last {MAX_MESSAGES} messages are remembered.")
print("Try a long conversation — early messages will be forgotten.\n")
print("Type 'quit' to exit\n")

conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    conversation_history.append({"role": "user", "content": user_input})

    # ✅ Trim: only keep the last MAX_MESSAGES messages
    trimmed_history = conversation_history[-MAX_MESSAGES:]

    reply = ask(trimmed_history)

    conversation_history.append({"role": "assistant", "content": reply})

    total = len(conversation_history)
    sending = len(trimmed_history)

    print(f"Bot: {reply}")
    print(f"[Total history: {total} | Sending to LLM: {sending}]\n")
