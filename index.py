import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from dedupe import unique_documents


# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "Basic_german.pdf"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "German_Basics"

EMBEDDING_MODEL = "text-embedding-3-large"


# ---------------- ARGUMENTS ----------------

parser = argparse.ArgumentParser(
    description="Index the German grammar PDF."
)

parser.add_argument(
    "--rebuild",
    action="store_true",
    help="Delete and recreate the Qdrant collection."
)

args = parser.parse_args()


# ---------------- ENVIRONMENT ----------------

load_dotenv()

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")


# ---------------- DOCUMENT LOADING ----------------

print("Loading German grammar PDF...")

loader = PyPDFLoader(file_path=str(PDF_PATH))

docs = loader.load()

print(f"Loaded {len(docs)} PDF pages.")


# ---------------- TEXT CHUNKING ----------------

print("Splitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(docs)

print(f"Generated {len(chunks)} chunks.")


# ---------------- DUPLICATE REMOVAL ----------------

chunks, chunk_ids = unique_documents(chunks)

print(f"Unique chunks: {len(chunks)}")


# ---------------- EMBEDDINGS ----------------

embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)


# ---------------- QDRANT VECTOR STORE ----------------

print("Creating Qdrant vector store...")

vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    ids=chunk_ids,
    force_recreate=args.rebuild,
)

print("\nIndexing completed successfully!")

print(f"Total unique chunks stored: {len(chunks)}")

print(f"Collection: {COLLECTION_NAME}")
