# CONCEPT 1: No Context
# Problem: LLM has NO memory. Every message is sent alone.
# Run this and say "my name is Alice", then ask "what is my name?"
# The LLM will not know your name.

import requests

def ask(message: str) -> str:
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3",
        "stream": False,
        "messages": [
            {"role": "user", "content": message}  # only THIS message — no history
        ]
    })
    return response.json()["message"]["content"]

print("=== Chatbot with NO context ===")
print("Try: say 'my name is Alice', then ask 'what is my name?'")
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    reply = ask(user_input)
    print(f"Bot: {reply}\n")
