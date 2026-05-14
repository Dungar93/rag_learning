# =========================================================
# COMPLETE RAG APPLICATION USING LANGCHAIN + CHROMADB
# =========================================================

# FEATURES:
# 1. Document Loading
# 2. Text Chunking
# 3. Embeddings
# 4. Chroma Vector Database
# 5. Retriever
# 6. Prompt Augmentation
# 7. LLM Generation
# 8. Full RAG Pipeline
# =========================================================


# =========================
# IMPORTS
# =========================

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser


# =========================
# LOAD ENV VARIABLES
# =========================

load_dotenv()


# =========================================================
# STEP 1 : LOAD DOCUMENTS
# =========================================================

loader = TextLoader("data.txt")

documents = loader.load()

print("\n================ DOCUMENTS LOADED ================\n")

print(documents[0].page_content[:500])


# =========================================================
# STEP 2 : SPLIT DOCUMENTS INTO CHUNKS
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("\n================ CHUNKS CREATED ================\n")

print("TOTAL CHUNKS :", len(chunks))

print("\nFIRST CHUNK:\n")

print(chunks[0].page_content)


# =========================================================
# STEP 3 : CREATE EMBEDDINGS MODEL
# =========================================================

embeddings = OpenAIEmbeddings()

print("\n================ EMBEDDINGS MODEL READY ================\n")


# =========================================================
# STEP 4 : CREATE VECTOR DATABASE (INDEXING)
# =========================================================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("\n================ CHROMADB CREATED ================\n")


# =========================================================
# STEP 5 : CREATE RETRIEVER
# =========================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("\n================ RETRIEVER READY ================\n")


# =========================================================
# STEP 6 : CREATE LLM
# =========================================================

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

print("\n================ LLM READY ================\n")


# =========================================================
# STEP 7 : CREATE PROMPT TEMPLATE
# =========================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the user's question ONLY from the provided context.

If the answer is not present in the context,
say:
"I could not find the answer in the provided documents."

================ CONTEXT ================

{context}

================ QUESTION ================

{question}

================ ANSWER ================
"""
)

print("\n================ PROMPT TEMPLATE READY ================\n")


# =========================================================
# STEP 8 : OUTPUT PARSER
# =========================================================

parser = StrOutputParser()


# =========================================================
# STEP 9 : START RAG CHATBOT
# =========================================================

print("\n================ RAG CHATBOT STARTED ================\n")

print("Type 'exit' to stop the chatbot.\n")


while True:

    # =====================================================
    # USER INPUT
    # =====================================================

    question = input("Ask Question : ")


    # =====================================================
    # EXIT CONDITION
    # =====================================================

    if question.lower() == "exit":

        print("\nGoodbye!\n")

        break


    # =====================================================
    # RETRIEVAL STEP
    # =====================================================

    retrieved_docs = retriever.invoke(question)

    print("\n================ RETRIEVED DOCUMENTS ================\n")


    # =====================================================
    # CREATE CONTEXT
    # =====================================================

    context_text = ""

    for i, doc in enumerate(retrieved_docs):

        print(f"\n----- DOCUMENT {i+1} -----\n")

        print(doc.page_content)

        print("\n")

        context_text += doc.page_content + "\n"


    # =====================================================
    # AUGMENTATION STEP
    # =====================================================

    final_prompt = prompt.invoke({
        "context": context_text,
        "question": question
    })


    # =====================================================
    # GENERATION STEP
    # =====================================================

    response = llm.invoke(final_prompt)


    # =====================================================
    # PARSE FINAL OUTPUT
    # =====================================================

    final_answer = parser.invoke(response)


    # =====================================================
    # PRINT FINAL ANSWER
    # =====================================================

    print("\n================ FINAL ANSWER ================\n")

    print(final_answer)

    print("\n")


# =========================================================
# END
# =========================================================