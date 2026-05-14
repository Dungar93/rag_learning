from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

# Load Model
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7
)

# -----------------------------
# Prompt 1 → Explanation
# -----------------------------

explain_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# -----------------------------
# Prompt 2 → Advantages
# -----------------------------

advantages_prompt = ChatPromptTemplate.from_template(
    "What are the advantages of {topic}?"
)

# -----------------------------
# Prompt 3 → Disadvantages
# -----------------------------

disadvantages_prompt = ChatPromptTemplate.from_template(
    "What are the disadvantages of {topic}?"
)

# -----------------------------
# Create Individual Chains
# -----------------------------

explain_chain = explain_prompt | llm

advantages_chain = advantages_prompt | llm

disadvantages_chain = disadvantages_prompt | llm

# -----------------------------
# Parallel Chain
# -----------------------------

parallel_chain = RunnableParallel({

    "Explanation": explain_chain,

    "Advantages": advantages_chain,

    "Disadvantages": disadvantages_chain
})

# -----------------------------
# Run Chain
# -----------------------------

response = parallel_chain.invoke({
    "topic": "Artificial Intelligence"
})

# -----------------------------
# Print Results
# -----------------------------

print("\n===== Explanation =====\n")
print(response["Explanation"].content)

print("\n===== Advantages =====\n")
print(response["Advantages"].content)

print("\n===== Disadvantages =====\n")
print(response["Disadvantages"].content)