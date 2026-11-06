import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


openai_client = OpenAI(api_key=api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="German_Basics",
)

user_query = input("Enter your query: ")
search_results = vector_store.similarity_search(user_query)

print("\nRetrieved chunks from the PDF:")
for i, result in enumerate(search_results, start=1):
    print(f"\n--- Chunk {i} | Page {result.metadata.get('page')} ---")
    print(result.page_content[:400])

if not search_results:
    print("No relevant chunks were found in the PDF for this query.")
    raise SystemExit

context = "\n\n\n".join(
    [
        f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page')}\nFile Location: {result.metadata.get('source')}"
        for result in search_results
    ]
)

system_prompt = f"""
You are a helpful assistant. Answer using ONLY the PDF context provided below.
Do not use outside knowledge. If the answer is not clearly in the PDF context, say:
"The provided PDF does not contain enough information to answer this question."

Context:
{context}
"""

openai_response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ],
)

print("\nAssistant's Response:")
print(openai_response.choices[0].message.content)
