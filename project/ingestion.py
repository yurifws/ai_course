# Ingest a financial Markdown file into Qdrant for hybrid RAG.
# 1) Split the file into paragraph chunks
# 2) Embed each chunk with dense (semantic) + sparse (BM25) + ColBERT (late interaction)
# 3) Upsert all three into Qdrant
# 4) Hybrid search: dense+sparse candidates fused with RRF, then ColBERT re-ranks the top hits

import os
import uuid

from dotenv import load_dotenv
from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
)
from qdrant_client import QdrantClient, models

# Load QDRANT_URL / QDRANT_API_KEY (and any other secrets) from .env.
load_dotenv()

# Model IDs passed to fastembed (downloaded on first use).
# Dense MiniLM -> 384-dim semantic vectors.
# Sparse BM25 -> term index/weight pairs.
# ColBERT -> multi-vector (one 128-d vector per token); scored with MaxSim at query time.
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"

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
# Named vectors:
# - "dense": single 384-d cosine vector (semantic)
# - "colbert": multi-vector (token-level 128-d), compared with MaxSim
# - "sparse": BM25 inverted index (separate sparse_vectors_config)
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            # Multi-vector: each point stores many token vectors; MaxSim picks best matches.
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    },
    sparse_vectors_config={"sparse": models.SparseVectorParams()},
)

# Read the full Markdown file as one string.
with open(FILE_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Naive chunking: split on blank lines, keep paragraphs long enough to be useful.
paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

# Three encoders for hybrid retrieval:
# - dense (MiniLM): semantic similarity — paraphrases / related meaning
# - sparse (BM25): keyword/term matching — tickers, legal jargon, exact phrases
# - ColBERT: late interaction — token-level MaxSim re-ranking for finer relevance
dense_model = TextEmbedding(model_name=DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(model_name=COLBERT_MODEL)

# Build points: each chunk gets three vectors + payload (original text + source).
points = []
for chunk in chunks:
    # passage_embed is for documents; returns an iterator — take the first vector.
    # as_object() -> {"indices": [...], "values": [...]} for Qdrant's SparseVector.
    # ColBERT passage: list of per-token vectors (multi-vector), not a single embedding.
    dense_embedding = list(dense_model.passage_embed([chunk]))[0].tolist()
    sparse_embedding = list(sparse_model.passage_embed([chunk]))[0].as_object()
    colbert_embedding = list(colbert_model.passage_embed([chunk]))[0].tolist()

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        # Must use the same names as vectors_config / sparse_vectors_config above.
        vector={
            "dense": dense_embedding,
            "sparse": sparse_embedding,
            "colbert": colbert_embedding,
        },
        # Payload keeps the raw text (and file path) so hits are human-readable.
        payload={"text": chunk, "source": FILE_PATH},
    )
    points.append(point)

# Upload all chunk vectors + payloads into the collection.
qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)

# Sample hybrid search: embed the question with all three query encoders.
query_text = "what are the main financial risks?"

# query_embed (not passage_embed) is the asymmetric counterpart for search queries.
# Dense -> fixed 384-float list.
# Sparse -> {"indices", "values"} term weights (as_object).
# ColBERT query -> also multi-vector; used only in the final re-rank stage below.
query_dense = list(dense_model.query_embed([query_text]))[0].tolist()
query_sparse = list(sparse_model.query_embed([query_text]))[0].as_object()
query_colbert = list(colbert_model.query_embed([query_text]))[0].tolist()

# Two-stage retrieval:
# 1) Prefetch: dense + sparse candidate lists, fused with RRF (Reciprocal Rank Fusion)
# 2) Outer query: ColBERT MaxSim re-ranks those fused candidates and returns top 3
results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        models.Prefetch(
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
            # RRF merges the two ranked lists without needing comparable raw scores.
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=20,
        ),
    ],
    # Re-rank the RRF shortlist with ColBERT (more precise, more expensive).
    query=query_colbert,
    using="colbert",
    limit=3,
)

# Scale scores to [0, 1] relative to the best hit (easier to read than raw MaxSim).
max_score = max(result.score for result in results.points)

# Inspect the top hits: ColBERT re-rank score and a short preview of each chunk.
for r in results.points:
    nomalized_score = r.score / max_score
    print(f"Score: {nomalized_score}")
    print(f"Text: {r.payload['text'][:100]}...")
    print("-" * 80)
