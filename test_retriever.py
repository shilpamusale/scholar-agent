# test_retriever.py
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

import configs.settings as settings

# --- 1. Set up the retriever exactly like in core.py ---
print("--- Initializing Retriever ---")
embedding_model = SentenceTransformerEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
vector_store = Chroma(
    persist_directory=str(settings.VECTOR_STORE_PATH),
    embedding_function=embedding_model,
)
retriever = vector_store.as_retriever(search_kwargs={"k": settings.RETRIEVER_TOP_K})
print("Retriever initialized successfully.\n")

# --- 2. Define our test queries ---
long_query = "Summarize the abstract of the paper The Interpretable Dictionary in Sparse Coding"
short_query = "The Interpretable Dictionary in Sparse Coding"

# --- 3. Test the long, conversational query ---
print(f"--- Testing Long Query: '{long_query}' ---")
results_long = retriever.invoke(long_query)
print(f"Found {len(results_long)} documents.")
# Optional: print the results to see what they are
# for i, doc in enumerate(results_long):
#     print(f"  Doc {i+1}: {doc.page_content[:100]}...")
print("-" * 40)

# --- 4. Test the short, keyword-based query ---
print(f"--- Testing Short Query: '{short_query}' ---")
results_short = retriever.invoke(short_query)
print(f"Found {len(results_short)} documents.")
for i, doc in enumerate(results_short):
    print(f"  Doc {i + 1} Title: {doc.metadata.get('title', 'N/A')}")
print("-" * 40)
