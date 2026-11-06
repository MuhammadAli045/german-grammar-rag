from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()
#------------------------------LOADING DOCUMENTS--------------------------------
pdf_path = Path(__file__).parent / "Basic_german.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

#split document text into chunks (CHUNKING PHASE) ------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(documents=docs)

#----------------------------VECTOR EMBEDDINGS---------------------------------
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="German_Basics"
    )

print("Vector store created and documents embedded successfully.")

#SUMMERY:
# LOAD file ------> CHUNKS ------> EMBEDDING_Model ------> VECTOR STORE (provide chunk, embeding_model, url, collection_name)