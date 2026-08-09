# Ingest a financial Markdown file into Qdrant for RAG.
# 1) Split the file into paragraph chunks
# 2) Embed each chunk with fastembed and upsert into Qdrant
# 3) Run a sample semantic query and print the top matches

import os
import uuid

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

# Load QDRANT_URL / QDRANT_API_KEY (and any other secrets) from .env.
load_dotenv()

# Same embedding family as the other demos; 384-dim vectors for MiniLM.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Named bucket in Qdrant where this financial text will live.
COLLECTION_NAME = "financial"
# Local Markdown source (e.g. an Apple 10-K Item 1A risk-factors excerpt).
FILE_PATH = "project/AAPL_10-K_1A_temp.md"

# Cloud Qdrant client (URL + API key from the environment).
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Recreate the collection so each run starts clean (demo-friendly, not production).
qdrant.delete_collection(COLLECTION_NAME)
# Collection sized to the embedding model; cosine similarity for retrieval.
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

# Read the full Markdown file as one string.
with open(FILE_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Naive chunking: split on blank lines, keep paragraphs long enough to be useful.
paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

# Embedding model: turns text into vectors so we can compare meaning (not just keywords).
model = TextEmbedding(model_name=MODEL_NAME)

# Build points: each chunk becomes an id + vector + payload (original text + source).
points = []
for chunk in chunks:
    # passage_embed is for documents; returns an iterator — take the first vector.
    embedding = list(model.passage_embed([chunk]))[0].tolist()
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        # Payload keeps the raw text (and file path) so hits are human-readable.
        payload={"text": chunk, "source": FILE_PATH},
    )
    points.append(point)

# Upload all chunk vectors + payloads into the collection.
qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)

# Sample semantic search: embed the question with the query encoder, rank chunks.
query_text = "what are the main financial risks?"
# query_embed (not passage_embed) is the asymmetric counterpart for search queries.
query_embedding = list(model.query_embed([query_text]))[0].tolist()

results = qdrant.query_points(
    collection_name=COLLECTION_NAME, query=query_embedding, limit=3
)

# Inspect the top hits: similarity score and a short preview of each chunk.
for r in results.points:
    print(f"Score: {r.score}")
    print(f"Text: {r.payload['text'][:100]}...")
    print("-" * 80)
