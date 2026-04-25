# CONCEPT 3: Sliding Window
# Problem: if you chat for 1000 messages, the context gets too big and expensive.
# Fix: only keep the last N messages. Old ones are dropped.

import anthropic

client = anthropic.Anthropic()

MAX_MESSAGES = 6  # only keep the last 6 messages (3 exchanges)

print("=== Chatbot with SLIDING WINDOW (last 6 messages only) ===")
print(f"Only the last {MAX_MESSAGES} messages are remembered.")
print("Try a long conversation — early messages will be forgotten.\n")
print("Type 'quit' to exit\n")

conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    # ✅ Trim: only keep the last MAX_MESSAGES messages
    trimmed_history = conversation_history[-MAX_MESSAGES:]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=trimmed_history
    )

    reply = response.content[0].text

    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    total = len(conversation_history)
    sending = len(trimmed_history)

    print(f"Bot: {reply}")
    print(f"[Total history: {total} | Sending to LLM: {sending}]\n")
