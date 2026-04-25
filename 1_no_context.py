# CONCEPT 1: No Context
# Problem: LLM has NO memory. Every message is sent alone.
# Run this and say "my name is Alice", then ask "what is my name?"
# The LLM will not know your name.

import anthropic

client = anthropic.Anthropic()

print("=== Chatbot with NO context ===")
print("Try: say 'my name is Alice', then ask 'what is my name?'")
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # ❌ Every message is sent ALONE — no history
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[
            {"role": "user", "content": user_input}  # only THIS message
        ]
    )

    reply = response.content[0].text
    print(f"Bot: {reply}\n")
