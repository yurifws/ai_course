# Hierarchical chunking with Docling.
# After conversion, we split the document into chunks that follow its
# structure (sections, headings, paragraphs) instead of fixed-size windows.

from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

# Default converter: same entry point as in the extraction examples.
converter = DocumentConverter()

# Convert a local PDF into a Docling document (structure + text).
result = converter.convert("./2408.09869v5.pdf")
document = result.document

# HierarchicalChunker walks the document tree and emits one chunk per
# meaningful unit (e.g. a paragraph under its heading context).
chunker = HierarchicalChunker()

# Materialize the generator into a list so we can index and inspect chunks.
chunks = list(chunker.chunk(document))

# In a notebook/REPL, bare `chunks` shows the full list; here we print one.
chunks
# Peek at a single chunk's text (index 4 is just an arbitrary example).
print(chunks[4].text)
