# RAG with Qdrant as the vector store (instead of scoring embeddings in plain Python).
# 1) Embed documents and upsert them into Qdrant
# 2) Retrieve the most similar ones for a query via vector search
# 3) Ask an LLM to answer using only that retrieved context.

from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Small knowledge base (the "documents" we retrieve from).
documents = [
    "Machine learning is a field of artificial intelligence that allows computers to learn patterns from data.",
    "Machine learning gives systems the ability to improve their performance without being explicitly programmed.",
    "Instead of following only fixed rules, machine learning discovers hidden relationships in data.",
    "This field combines statistics, algorithms, and computing power to extract knowledge.",
    "The goal is to create models capable of generalizing beyond the examples seen during training.",
    "Machine learning applications range from movie recommendations to medical diagnoses.",
    "Machine learning algorithms transform raw data into useful predictions.",
    "Unlike traditional software, ML adapts as new data arrives.",
    "Learning can be supervised, unsupervised, or reinforcement, depending on the type of problem.",
    "In practice, machine learning is the engine that drives many advances in computer vision and natural language processing.",
    "More than finding patterns, machine learning helps make decisions based on evidence.",
]

# Embedding model: turns text into vectors so we can compare meaning (not just keywords).
model = SentenceTransformer("all-MiniLM-L6-v2")
# Groq client for the generation step (reads GROQ_API_KEY from the environment).
client = Groq()

# Vector DB client. ":memory:" keeps everything in RAM (lost when the process ends).
qdrant = QdrantClient(":memory:")
# path="db/data" persists vectors on disk so you can reuse them across runs.
# qdrant = QdrantClient(path="db/data")

# all-MiniLM-L6-v2 produces 384-dim vectors; Qdrant needs this to size the collection.
vector_size = model.get_embedding_dimension()

# Create a collection: a named bucket of vectors with a fixed size + distance metric.
# COSINE matches the similarity we used manually in rag.py.
qdrant.create_collection(
    collection_name="ml_documents",
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

# Build points: each document becomes an id + vector + payload (original text).
points = []

for idx, doc in enumerate(documents):
    # Flat 1D list (not a nested [[...]]); Qdrant expects one vector per point.
    embedding = model.encode(doc).tolist()
    points.append(PointStruct(id=idx, vector=embedding, payload={"text": doc}))

# wait=True blocks until the upsert is fully applied before we search.
qdrant.upsert(collection_name="ml_documents", points=points, wait=True)


def retrieve(query, top_k=3):
    # Encode the query into the same vector space as the documents.
    # Pass a single string (not [query]) so we get a flat vector, not a multivector.
    query_embedding = model.encode(query).tolist()

    # Qdrant finds the nearest neighbors; no manual cosine loop needed.
    search_result = qdrant.query_points(
        collection_name="ml_documents",
        query=query_embedding,
        limit=top_k,
        # Include the original text stored in payload (not just scores/ids).
        with_payload=True,
    )
    # Return (text, similarity score) pairs, same shape as rag.py's retrieve().
    return [(hit.payload["text"], hit.score) for hit in search_result.points]


def generate_answer(query, retrieve_docs):
    # Build the context string the LLM is allowed to use.
    context = "\n".join([doc for doc, _ in retrieve_docs])

    reponse = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                # Ground the model: answer from context only (reduces hallucination).
                "content": "You are a machine learning specialist. Answer only from the provided context.",
            },
            # English labels keep the reply language aligned with the documents.
            {"role": "user", "content": f"Context: \n{context}\n\nQuestion: {query}"},
        ],
        # Low temperature -> more deterministic, less creative rewriting.
        temperature=0,
    )
    return reponse.choices[0].message.content


def rag(query, top_k=3):
    # Full RAG pipeline: retrieve relevant docs, then generate an answer from them.
    retrieved = retrieve(query, top_k)
    answer = generate_answer(query, retrieved)
    return answer, retrieved


answer, docs = rag("what is machine learning?")
print(answer)
print(docs)

# Show which chunks were retrieved and how similar each was to the query.
for doc, simlarity in docs:
    print(f" - {simlarity:.3f} : {doc}")
