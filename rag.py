# Simple RAG (Retrieval-Augmented Generation) demo.
# 1) Embed documents  2) Retrieve the most similar ones for a query
# 3) Ask an LLM to answer using only that retrieved context.

import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Small in-memory knowledge base (the "documents" we retrieve from).
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

# Precompute embeddings for every document once (cheaper than encoding on each query).
doc_embeddings = model.encode(documents)


def cosine_similarity(a, b):
    # Cosine similarity = how aligned two vectors are (1 = same direction, 0 = orthogonal).
    # Parentheses matter: divide by (||a|| * ||b||), not only by ||a|| then multiply by ||b||.
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Math python cosine similarity
# v1 = np.array([1, 2, 3])
# v2 = np.array([4, 5, 6])
# dot_product = np.dot(v1, v2)
# norm_euclidian = np.linalg.norm(v1) * np.linalg.norm(v2)
# dot_product / norm_euclidian


def retrieve(query, top_k=3):
    # Encode the query into the same vector space as the documents.
    query_embedding = model.encode([query])[0]

    similarities = []

    # Score every document against the query.
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((i, sim))

    # Highest similarity first.
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the top_k documents with their scores.
    return [(documents[i], sim) for i, sim in similarities[:top_k]]


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
