# from dotenv import load_dotenv
# load_dotenv()

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate

# llm = ChatOpenAI(model="gpt-4.1-mini")

# prompt = ChatPromptTemplate.from_template(
#     "Explain {topic} in simple words"
# )

# chain = prompt | llm

# response = chain.invoke({
#     "topic": "Blockchain"
# })

# print(response.content)
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=2
)

prompt = PromptTemplate.from_template(
    "Explain {topic} in simple words"
)

final_prompt = prompt.format(topic="blockchain")

response = llm.invoke(final_prompt)

print(response.content)