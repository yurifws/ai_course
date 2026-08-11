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
warnings.simplefilter(action='ignore', category=FutureWarning)


class SemanticChunker:
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

    def create_chunks(self, text_content: str):
        # Paragraphs = blocks separated by blank lines; skip tiny scraps.
        paragraphs = [
            p.strip() for p in text_content.split("\n\n") if len(p.strip()) > 10
        ]
        if not paragraphs:
            return []

        # One dense vector per paragraph for density-based clustering.
        embeddings = self.model.encode(paragraphs, show_progress_bar=False)
        # label == -1 means HDBSCAN treated the paragraph as noise (orphan).
        labels = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size, 
            metric="euclidean"
        ).fit_predict(embeddings)

        # Bucket paragraphs by cluster id; keep orphans for a second pass.
        clusters = defaultdict(list)
        orphans = []
        for i, label in enumerate(labels):
            if label != -1:
                clusters[label].append(paragraphs[i])
            else:
                orphans.append(paragraphs[i])

        final_chunks = []
        # Pack each semantic cluster into one or more token-bounded chunks.
        for cluster_paragraphs in clusters.values():
            current_chunk = []
            current_tokens = 0

            for paragraph in cluster_paragraphs:
                paragraph_tokens = len(self.tokenizer.encode(paragraph, add_special_tokens=False))

                # Flush when adding this paragraph would exceed the budget.
                if current_tokens + paragraph_tokens > self.max_tokens and current_chunk:
                    final_chunks.append("\n\n".join(current_chunk))
                    current_chunk = [paragraph]
                    current_tokens = paragraph_tokens
                else:
                    current_chunk.append(paragraph)
                    current_tokens += paragraph_tokens

            if current_chunk:
                final_chunks.append("\n\n".join(current_chunk))

        # Second pass: orphans may still form smaller related groups.
        if len(orphans) > 1:
            orphan_embeddings = self.model.encode(orphans, show_progress_bar=False)
            orphan_labels = hdbscan.HDBSCAN(
                min_cluster_size=self.orphan_cluster_size, metric="euclidean"
            ).fit_predict(orphan_embeddings)

            orphan_clusters = defaultdict(list)
            single_orphans = []

            for i, label in enumerate(orphan_labels):
                if label != -1:
                    orphan_clusters[label].append(orphans[i])
                else:
                    # Still noise after the second pass: keep as a lone chunk.
                    single_orphans.append(orphans[i])

            for orphan_paragraphs in orphan_clusters.values():
                current_chunk = []
                current_tokens = 0

                for paragraph in orphan_paragraphs:
                    paragraph_tokens = len(self.tokenizer.encode(paragraph, add_special_tokens=False))

                    if current_tokens + paragraph_tokens > self.max_tokens and current_chunk:
                        final_chunks.append("\n\n".join(current_chunk))
                        current_chunk = [paragraph]
                        current_tokens = paragraph_tokens
                    else:
                        current_chunk.append(paragraph)
                        current_tokens += paragraph_tokens

                if current_chunk:
                    final_chunks.append("\n\n".join(current_chunk))

            final_chunks.extend(single_orphans)

        elif orphans:
            # Only one leftover paragraph: emit it as its own chunk.
            final_chunks.append(orphans[0])

        return final_chunks
