# CONCEPT 2: With Context
# Fix: we keep a list of ALL messages and send the full history every time.
# Now the LLM remembers everything you said.

import requests

def ask(messages: list) -> str:
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3",
        "stream": False,
        "messages": messages  # full history sent every time
    })
    return response.json()["message"]["content"]

print("=== Chatbot WITH context ===")
print("Try: say 'my name is Alice', then ask 'what is my name?'")
print("Type 'quit' to exit\n")

# ✅ This list grows with every message
conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # Step 1: Add user message to history
    conversation_history.append({"role": "user", "content": user_input})

    # Step 2: Send FULL history — LLM sees everything
    reply = ask(conversation_history)

    # Step 3: Add reply to history too
    conversation_history.append({"role": "assistant", "content": reply})

    print(f"Bot: {reply}")
    print(f"[History size: {len(conversation_history)} messages]\n")
