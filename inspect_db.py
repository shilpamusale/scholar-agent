# inspect_db.py
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

import configs.settings as settings

print("--- Connecting to ChromaDB ---")
# We need to pass an embedding function to connect, even if we don't use it.
# It's important this is the SAME embedding function used by the app.
embedding_model = SentenceTransformerEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

vector_store = Chroma(
    persist_directory=str(settings.VECTOR_STORE_PATH),
    embedding_function=embedding_model,
)
print("Connection successful.\n")

# --- Get Collection Info ---
# The underlying collection name defaults to "langchain" if not specified.
collection = vector_store.get()
collection_name = vector_store._collection.name
doc_count = vector_store._collection.count()

print(f"--- Inspecting Collection: '{collection_name}' ---")
print(f"Total documents in the database: {doc_count}\n")


# --- Peek at a few documents ---
if doc_count > 0:
    print("--- Peeking at up to 5 documents in the collection ---")
    # Using the raw collection object from Chroma to see the data
    peek_result = vector_store._collection.peek(limit=5)

    ids = peek_result.get("ids", [])
    documents = peek_result.get("documents", [])
    metadatas = peek_result.get("metadatas", [])

    for i in range(len(ids)):
        print(f"  Document ID: {ids[i]}")
        print(f"    Metadata: {metadatas[i]}")
        print(f"    Content Start: {documents[i][:150]}...\n")
else:
    print("Database appears to be empty.")

print("-" * 40)
