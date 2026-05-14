from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

# =========================
# Load Model
# =========================

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7
)

# =========================
# System Prompt
# =========================

system_prompt = SystemMessage(
    content="""
You are a smart, friendly and helpful AI assistant.
Explain things clearly and simply.
Keep answers concise unless user asks for detail.
"""
)

# =========================
# Chat History
# =========================

chat_history = [system_prompt]

# =========================
# Chatbot UI
# =========================

print("=" * 50)
print("🤖 Advanced AI Chatbot Started")
print("Type 'exit' to quit")
print("=" * 50)

# =========================
# Chat Loop
# =========================

while True:

    user_input = input("\n🧑 You: ")

    # Exit
    if user_input.lower() == "exit":
        print("\n👋 Chatbot Ended.")
        break

    # Save user message
    chat_history.append(
        HumanMessage(content=user_input)
    )

    try:

        # Get response
        response = llm.invoke(chat_history)

        # Print response
        print("\n🤖 AI:", response.content)

        # Save AI response
        chat_history.append(
            AIMessage(content=response.content)
        )

    except Exception as e:
        print("\n❌ Error:", e)