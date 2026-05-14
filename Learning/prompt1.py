from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini")

prompt = PromptTemplate(
    template="Explain what is {topic} and is it good in {area}",
    input_variables=["topic", "area"]
)

# final_prompt = prompt.format(
#     topic="vibe coding",
#     area="software development"
# )

# print(final_prompt)
chain = prompt | llm

response = chain.invoke({
    "topic": "vibe coding",
    "area": "software development"
})

print(response.content)