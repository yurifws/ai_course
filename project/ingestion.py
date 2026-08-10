# Ingest a financial Markdown file into Qdrant for hybrid RAG.
# 1) Split the file into paragraph chunks
# 2) Embed each chunk with dense (semantic) + sparse (BM25 keyword) vectors
# 3) Upsert both into Qdrant, then run a hybrid query (RRF) and print top matches

import os
import uuid

from dotenv import load_dotenv
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models

# Load QDRANT_URL / QDRANT_API_KEY (and any other secrets) from .env.
load_dotenv()

# Model IDs passed to fastembed (downloaded on first use).
# Dense MiniLM -> 384-dim semantic vectors; sparse BM25 -> term index/weight pairs.
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
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
# Named vectors: "dense" (cosine, 384-d) + "sparse" (BM25 inverted index).
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
    },
    sparse_vectors_config={"sparse": models.SparseVectorParams()},
)

# Read the full Markdown file as one string.
with open(FILE_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Naive chunking: split on blank lines, keep paragraphs long enough to be useful.
paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

# Two encoders for hybrid retrieval:
# - dense (MiniLM): semantic similarity — paraphrases / related meaning
# - sparse (BM25): keyword/term matching — tickers, legal jargon, exact phrases
dense_model = TextEmbedding(model_name=DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

# Build points: each chunk gets both vectors + payload (original text + source).
points = []
for chunk in chunks:
    # passage_embed is for documents; returns an iterator — take the first vector.
    dense_embedding = list(dense_model.passage_embed([chunk]))[0].tolist()
    # as_object() -> {"indices": [...], "values": [...]} for Qdrant's SparseVector.
    sparse_embedding = list(sparse_model.passage_embed([chunk]))[0].as_object()

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        # Must use the same names as vectors_config / sparse_vectors_config above.
        vector={"dense": dense_embedding, "sparse": sparse_embedding},
        # Payload keeps the raw text (and file path) so hits are human-readable.
        payload={"text": chunk, "source": FILE_PATH},
    )
    points.append(point)

# Upload all chunk vectors + payloads into the collection.
qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)

# Sample hybrid search: embed the question with both query encoders.
query_text = "what are the main financial risks?"

# query_embed (not passage_embed) is the asymmetric counterpart for search queries.
# Dense -> fixed 384-float list; sparse -> {"indices", "values"} term weights (as_object).
query_dense = list(dense_model.query_embed([query_text]))[0].tolist()
query_sparse = list(sparse_model.query_embed([query_text]))[0].as_object()

# Prefetch: two candidate lists (dense semantic + sparse BM25), then RRF merges ranks.
results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        # using= must match the named vectors created in create_collection.
        models.Prefetch(query=query_dense, using="dense", limit=10),
        models.Prefetch(
            # ** unpacks {"indices", "values"} into SparseVector fields.
            query=models.SparseVector(**query_sparse),
            using="sparse",
            limit=10,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=3,
)

# Inspect the top hits: fused RRF score and a short preview of each chunk.
for r in results.points:
    print(f"Score: {r.score}")
    print(f"Text: {r.payload['text'][:100]}...")
    print("-" * 80)
