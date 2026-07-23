# First contact with Docling: convert a document into structured content,
# then export it as Markdown.
# Docling downloads layout/OCR models from Hugging Face on first run
# (cached under ~/.cache/huggingface afterward).

from docling.document_converter import DocumentConverter

# Default converter: handles common formats (PDF, HTML, etc.) out of the box.
converter = DocumentConverter()

# Option A: convert a local PDF file.
# result = converter.convert("./2408.09869v5.pdf")

# Option B: convert a remote PDF by URL (Docling fetches it for you).
result = converter.convert("https://arxiv.org/pdf/2408.09869")
# The converted Docling document (structure + text, not just raw bytes).
document = result.document
# Export to Markdown so we can read/print the extracted text easily.
markdown_output = document.export_to_markdown()

# Same pipeline for an HTML page: Docling fetches the page and extracts content.
result = converter.convert("https://docling-project.github.io/docling/")
document = result.document
markdown_output = document.export_to_markdown()
