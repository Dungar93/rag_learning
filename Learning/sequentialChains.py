from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# First Prompt
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# Second Prompt
prompt2 = PromptTemplate(
    template='Generate a 5 point summary from the following text:\n{text}',
    input_variables=['text']
)

# OpenAI Model
model = ChatOpenAI(
    model="gpt-3.5-turbo"
)

# Output Parser
parser = StrOutputParser()

# First Chain
report_chain = prompt1 | model | parser

# Second Chain
summary_chain = prompt2 | model | parser

# Final Chain
chain = (
    report_chain
    | (lambda output: {"text": output})
    | summary_chain
)

# Run
result = chain.invoke({
    "topic": "Unemployment in India"
})

print(result)