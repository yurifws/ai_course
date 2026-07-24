# Hybrid chunking with Docling: structure-aware splits + token budget.
# Unlike HierarchicalChunker alone, HybridChunker also respects a max token
# length (useful before embedding) and can merge small sibling chunks.

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

# Embedding model whose tokenizer we use to count tokens (must match what
# you will embed with later, or at least use a compatible tokenizer).
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Soft upper bound per chunk; HybridChunker splits/merges to stay near this.
MAX_TOKENS = 300

converter = DocumentConverter()

# Convert a local PDF into a Docling document first.
result = converter.convert("./2408.09869v5.pdf")
document = result.document

# Alternative tokenizer (commented): another MiniLM-family model; useful to
# inspect model_max_length if you are unsure about context limits.
# tokenizer = AutoTokenizer.from_pretrained(
#     "sentence-transformers/paraphrase-mpnet-base-v2"
# )
# print(tokenizer.model_max_length)

# Wrap the HF tokenizer so Docling can count tokens with our MAX_TOKENS cap.
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL), max_tokens=MAX_TOKENS
)

# merge_peers=True: adjacent small chunks under the same parent can be merged
# when that still fits within MAX_TOKENS (fewer, denser chunks for RAG).
chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)

# Materialize chunks so we can count and iterate them.
chunks = list(chunker.chunk(document))
len(chunks)

# Print each chunk with its token count (helps verify the budget works).
for i, chunk in enumerate(chunks):
    print(f"===={i}====\n")
    txt_tokens = tokenizer.count_tokens(chunk.text)
    print(f"chunk.text ({txt_tokens} tokens): \n{chunk.text!r}")
