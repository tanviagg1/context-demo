# CONCEPT 2: With Context
# Fix: we keep a list of ALL messages and send the full history every time.
# Now the LLM remembers everything you said.

import anthropic

client = anthropic.Anthropic()

print("=== Chatbot WITH context ===")
print("Try: say 'my name is Alice', then ask 'what is my name?'")
print("Type 'quit' to exit\n")

# ✅ This list grows with every message
conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    # Step 1: Add the user message to history
    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    # Step 2: Send the FULL history — LLM sees everything
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=conversation_history  # all messages, not just this one
    )

    reply = response.content[0].text

    # Step 3: Add the reply to history too
    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    print(f"Bot: {reply}")
    print(f"[History size: {len(conversation_history)} messages]\n")
