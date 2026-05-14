from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma


# ==========================================
# STEP 1 : Load Environment Variables
# ==========================================

load_dotenv()


# ==========================================
# STEP 2 : Load Large Text File
# ==========================================

loader = TextLoader("data.txt")

documents = loader.load()

print("\n================ DOCUMENT LOADED ================\n")

print(documents[0].page_content[:500])

print("\nTotal Documents Loaded :", len(documents))


# ==========================================
# STEP 3 : Split Into Chunks
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("\n================ CHUNKS CREATED ================\n")

print("Total Chunks :", len(chunks))

print("\nFIRST CHUNK:\n")

print(chunks[0].page_content)


# ==========================================
# STEP 4 : Create Embeddings Model
# ==========================================

embeddings = OpenAIEmbeddings()

print("\n================ EMBEDDINGS READY ================\n")


# ==========================================
# STEP 5 : Store Chunks in ChromaDB
# ==========================================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("\n================ DATA STORED IN CHROMADB ================\n")


# ==========================================
# STEP 6 : Create Retriever
# ==========================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("\n================ RETRIEVER CREATED ================\n")


# ==========================================
# STEP 7 : Start Question Answering Loop
# ==========================================

print("\n================ AI DOCUMENT CHAT ================\n")

print("Type 'exit' to stop.\n")


while True:

    query = input("Ask Question : ")

    if query.lower() == "exit":
        print("\nGoodbye!")
        break

    # ==========================================
    # Retrieve Similar Chunks
    # ==========================================

    results = retriever.invoke(query)

    print("\n================ RETRIEVED CHUNKS ================\n")

    for i, doc in enumerate(results):

        print(f"\n----- Result {i+1} -----\n")

        print(doc.page_content)

        print("\n")


# ==========================================
# END
# ==========================================