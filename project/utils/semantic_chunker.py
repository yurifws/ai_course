# Semantic chunking: group related paragraphs by embedding similarity, then
# pack each group into token-bounded chunks for RAG.
# 1) Split text on blank lines into paragraphs
# 2) Embed paragraphs and cluster with HDBSCAN (similar meaning stays together)
# 3) Pack each cluster into chunks that stay under max_tokens
# 4) Re-cluster leftover "orphan" paragraphs (noise from the first pass)

from collections import defaultdict
from sentence_transformers import SentenceTransformer

import hdbscan
from transformers import AutoTokenizer
import warnings

# HDBSCAN / sklearn may emit FutureWarnings on newer NumPy; ignore for demos.
warnings.simplefilter(action="ignore", category=FutureWarning)


class SemanticChunker:
    """Cluster paragraphs by meaning, then pack them into token-bounded RAG chunks."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_cluster_size: int = 3,
        orphan_cluster_size: int = 2,
        max_tokens: int = 300,
    ):
        # Same MiniLM family as dense retrieval: paragraph vectors live in a
        # comparable semantic space to the ingest embeddings.
        self.model = SentenceTransformer(model_name)
        # Cap encoder context; paragraphs are short, so 512 is enough.
        self.model.max_seq_length = 512
        # HDBSCAN needs at least this many neighbors to form a cluster.
        self.min_cluster_size = min_cluster_size
        # Second-pass clustering of leftovers: smaller groups are allowed.
        self.orphan_cluster_size = orphan_cluster_size
        # Soft upper bound per final chunk (must fit the embedder window).
        self.max_tokens = max_tokens
        # Token counts must match the embedding model's tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _cluster_and_process(self, texts, min_size: int):
        """Embed + HDBSCAN cluster texts; return (packed_chunks, orphan_paragraphs)."""
        # Too few items to cluster — return them as-is (single item is also an "orphan").
        if len(texts) <= 1:
            return texts, texts if len(texts) == 1 else []

        # One dense vector per paragraph for density-based clustering.
        embeddings = self.model.encode(texts, show_progress_bar=False)
        # label == -1 means HDBSCAN treated the paragraph as noise (orphan).
        labels = hdbscan.HDBSCAN(
            min_cluster_size=min_size, metric="euclidean"
        ).fit_predict(embeddings)

        # Bucket paragraphs by cluster id; keep orphans for a second pass.
        clusters = defaultdict(list)
        orphans = []
        for i, label in enumerate(labels):
            if label != -1:
                clusters[label].append(texts[i])
            else:
                orphans.append(texts[i])

        chunks = []
        # Pack each semantic cluster into one or more token-bounded chunks.
        for cluster_paragraphs in clusters.values():
            # Grow a chunk until max_tokens, then start a new one in the same cluster.
            current_chunk = []
            current_tokens = 0

            for paragraph in cluster_paragraphs:
                # Count tokens with the same tokenizer the embedder uses.
                paragraph_tokens = len(
                    self.tokenizer.encode(paragraph, add_special_tokens=False)
                )

                # Flush when adding this paragraph would exceed the budget.
                if (
                    current_tokens + paragraph_tokens > self.max_tokens
                    and current_chunk
                ):
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [paragraph]
                    current_tokens = paragraph_tokens
                else:
                    current_chunk.append(paragraph)
                    current_tokens += paragraph_tokens

            # Flush the last partial chunk for this cluster.
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

        return chunks, orphans

    def create_chunks(self, text_content: str):
        """Split document → cluster → pack; re-cluster HDBSCAN noise at a lower min size."""
        # Paragraphs = blocks separated by blank lines; skip tiny scraps.
        paragraphs = [
            p.strip() for p in text_content.split("\n\n") if len(p.strip()) > 10
        ]
        if not paragraphs:
            return []

        # First pass: larger min_cluster_size → bigger topical groups.
        final_chunks, orphans = self._cluster_and_process(
            paragraphs, self.min_cluster_size
        )

        # Second pass: orphans may still form smaller related groups.
        if len(orphans) > 1:
            orphans_chunks, single_orphans = self._cluster_and_process(
                orphans, self.orphan_cluster_size
            )
            final_chunks.extend(orphans_chunks)
            # True leftovers after pass 2: keep as one-paragraph chunks.
            final_chunks.extend(single_orphans)
        elif orphans:
            # Only one orphan left — nothing to cluster; keep it as its own chunk.
            final_chunks.extend(orphans)

        return final_chunks
