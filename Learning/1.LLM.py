from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini",temperature=0.7)

response = llm.invoke("write a 5 lines poen on cricket in english?")

print(response.content)