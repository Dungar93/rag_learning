# from dotenv import load_dotenv
# load_dotenv()

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate

# # Model
# llm = ChatOpenAI(model="gpt-4.1-mini")

# # Prompt Template
# prompt = ChatPromptTemplate.from_template(
#     "Explain {topic} in simple words"
# )

# # Create Chain
# chain = prompt | llm

# # Run Chain
# response = chain.invoke({
#     "topic": "Machine Learning"
# })

# print(response.content)
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Model
llm = ChatOpenAI(model="gpt-4.1-mini")

# Prompt
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one line"
)

# Parser
parser = StrOutputParser()

# Chain
chain = prompt | llm | parser

# Run
response = chain.invoke({
    "topic": "Deep Learning"
})

print(response)