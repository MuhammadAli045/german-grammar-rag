import os

from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from dedupe import unique_search_results


# ---------------- CONFIGURATION ----------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "German_Basics"

EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"


# ---------------- ENVIRONMENT ----------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

openai_client = OpenAI(api_key=api_key)


# ---------------- EMBEDDINGS ----------------

embedding_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=api_key
)


# ---------------- QDRANT VECTOR STORE ----------------

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)


# ---------------- RETRIEVAL ----------------

user_query = input("\nAsk your German grammar question: ").strip()

if not user_query:
    raise SystemExit("Please enter a question.")


retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)

search_results = unique_search_results(
    retriever.invoke(user_query),
    limit=5
)


# ---------------- CHECK RESULTS ----------------

if not search_results:
    print("\nNo relevant information found in the PDF.")
    raise SystemExit


# ---------------- DISPLAY RETRIEVED CHUNKS ----------------

print("\nRetrieved information from the PDF:")

for i, result in enumerate(search_results, start=1):

    # PyPDFLoader uses zero-based page numbering.
    page = result.metadata.get("page")

    display_page = page + 1 if isinstance(page, int) else "Unknown"

    print(f"\n--- Chunk {i} | PDF Page {display_page} ---")

    print(result.page_content[:400])


# ---------------- BUILD CONTEXT ----------------

context_parts = []

for result in search_results:

    page = result.metadata.get("page")

    display_page = page + 1 if isinstance(page, int) else "Unknown"

    context_parts.append(
        f"""
        PDF Content:
        {result.page_content}

        PDF Page: {display_page}
        Source: {result.metadata.get('source', 'Unknown')}
        """
    )

context = "\n\n".join(context_parts)


# ---------------- SYSTEM PROMPT ----------------

system_prompt = f"""
You are an expert German grammar teacher and an intelligent
German language tutor.

Your goal is to help students understand German grammar
clearly, accurately, and deeply.

You have two knowledge sources:

1. PRIMARY SOURCE:
   The German grammar PDF provided in the context.

2. SUPPLEMENTARY KNOWLEDGE:
   Your existing knowledge of German grammar and language.

================ KNOWLEDGE RULES ================

1. Always examine the retrieved PDF context first.

2. Use the PDF as your primary source whenever it contains
   relevant information about the user's question.

3. Understand the information in the PDF and explain it
   in your own words. Do not simply copy the text.

4. You are allowed to use your own knowledge to:
   - Improve explanations.
   - Simplify difficult grammar concepts.
   - Explain concepts in greater depth.
   - Add useful German examples.
   - Provide English translations.
   - Clarify concepts that the book explains briefly.

5. Never pretend that supplementary information came
   from the PDF.

6. If the PDF does not contain enough information,
   clearly say that you are supplementing the explanation
   with your own German grammar knowledge.

7. Never invent PDF content or page references.

8. If you cannot confidently answer a question,
   acknowledge the uncertainty.

================ TEACHING INSTRUCTIONS ================

When explaining a grammar topic:

1. Start with a clear and simple definition.
2. Explain the concept in easy English.
3. Explain how it works in German.
4. Give practical German examples.
5. Translate the examples into English.
6. Explain important grammar rules and patterns.
7. Highlight common mistakes when relevant.
8. Use tables when they make comparisons easier.

For example, if the user asks about adjectives:

- Explain what adjectives are.
- Explain their function in German sentences.
- Explain adjectives before nouns.
- Explain adjective endings when relevant.
- Provide examples with English translations.

Do not explain unrelated grammar topics unless
they help answer the question.

================ RESPONSE STYLE ================

- Be a patient and professional German teacher.
- Use simple English unless the user requests German.
- Explain difficult concepts step by step.
- Prefer practical examples over complicated theory.
- Use headings and tables where useful.
- Do not make every answer unnecessarily long.
- Adapt the explanation to the student's question.

================ SOURCE REFERENCES ================

When using information from the PDF, mention the relevant
PDF page numbers.

If you add information beyond the book, identify it as
additional explanation based on your existing knowledge.

Do not assign PDF page references to information that
does not come from the retrieved context.

================ RETRIEVED PDF CONTEXT ================

{context}
"""


# ---------------- GENERATE ANSWER ----------------

print("\nGenerating explanation...")

response = openai_client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_query
        }
    ],
    temperature=0.3,
)


# ---------------- FINAL RESPONSE ----------------

answer = response.choices[0].message.content

print("\n" + "=" * 60)
print("GERMAN GRAMMAR TUTOR")
print("=" * 60)

print(answer)