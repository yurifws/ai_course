# Embed Docling chunks into Qdrant for RAG.
# Pipeline: convert PDF → HybridChunker → attach paper metadata from
# LangExtract (5-metadatas) → store embeddings locally → run a sample query.

import json

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer

# Same embedding model for token counting, chunk budgets, and Qdrant vectors.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Soft upper bound per chunk; must stay under the embedder's context window.
MAX_TOKENS = 300

# Default converter: same entry point as in the extraction examples.
converter = DocumentConverter()
# Convert a local PDF into a Docling document (structure + text).
result = converter.convert("./2408.09869v5.pdf")
document = result.document

# Wrap the HF tokenizer so HybridChunker can respect MAX_TOKENS.
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL),
    max_tokens=MAX_TOKENS,
)

# Same hybrid settings as 4-hybrid-chunker: structure + token budget + merge.
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

# Materialize chunks so we can embed and upload them by index.
chunks = list(chunker.chunk(document))

# Defaults until we find values in the LangExtract JSONL from 5-metadatas.
paper_title = "N/A"
paper_url = "N/A"

# Read extracted metadata (title / URL) produced by the previous demo.
with open("./test_output/docling_paper_metadata.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        for extraction in doc.get("extractions", []):
            extraction_class = extraction.get("extraction_class")
            extraction_text = extraction.get("extraction_text")

            # Keep the first title / URL only (JSONL may repeat fields).
            if extraction_class == "title" and paper_title == "N/A":
                paper_title = extraction_text

            if extraction_class == "url" and paper_url == "N/A":
                paper_url = extraction_text

# Document-level metadata shared by every chunk payload (for citations).
metadata_document_info = {
    "title": paper_title,
    "url": paper_url,
}

# Local on-disk Qdrant (no server); data lives under db/data.
qdrant = QdrantClient(path="db/data")
# Collection sized to the embedding model; cosine similarity for retrieval.
qdrant.create_collection(
    collection_name="docling_paper",
    vectors_config=models.VectorParams(
        size=qdrant.get_embedding_size(EMBED_MODEL),
        distance=models.Distance.COSINE,
    ),
)

# Parallel lists: payload (text + metadata), vectors, and point ids.
payload = []
embed = []
ids = []

for idx, chunk in enumerate(chunks):
    # Store raw text + paper metadata so query hits are human-readable.
    payload.append({"text": chunk.text, "metadata": metadata_document_info})
    # Document(...) tells Qdrant to embed this text with EMBED_MODEL locally.
    embed.append(models.Document(text=chunk.text, model=EMBED_MODEL))
    ids.append(idx)

# Upload all chunk vectors + payloads into the collection.
qdrant.upload_collection(
    collection_name="docling_paper",
    vectors=embed,
    ids=ids,
    payload=payload,
)

# Sample semantic search: embed the question with the same model, rank chunks.
result = qdrant.query_points(
    collection_name="docling_paper",
    query=models.Document(
        text="what is docling?",
        model=EMBED_MODEL,
    ),
).points

# Inspect the top hit: full payload, chunk text, and paper URL for citation.
result[0].payload
result[0].payload["text"]
result[0].payload["metadata"]["url"]
