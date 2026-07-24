# Metadata extraction with Docling + LangExtract.
# First we convert a PDF to Markdown with Docling, then we ask an LLM
# (via LangExtract + Groq) to pull structured fields: title, authors,
# affiliation, version, and repository URLs — grounded in exact document text.

import os
import textwrap
from pathlib import Path

import langextract as lx
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from langextract.providers.openai import OpenAILanguageModel

# Load GROQ_API_KEY (and any other secrets) from ai_course/.env.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

# Default converter: same entry point as in the extraction examples.
converter = DocumentConverter()

# Convert a local PDF into a Docling document (structure + text).
result = converter.convert("./2408.09869v5.pdf")
document = result.document
# Markdown is a clean text form for the LLM to read.
markdown_output = document.export_to_markdown()

# Cap input size: metadata usually lives on the first pages (title, authors).
first_pages = markdown_output[:6000]

# Instruction for LangExtract: what to pull and how (exact spans only).
prompt = textwrap.dedent("""\
Extract metadata from this technical report including title, all authors, 
affiliation, version number, and GitHub repository URLs.
Use exact text from the document.
""")

# Few-shot example: shows the expected extraction_class labels and shape.
# LangExtract uses this to steer the model toward consistent structured output.
examples = [
    lx.data.ExampleData(
        text="Docling Technical Report\nVersion 1.0\nChristoph Auer Maksym Lysak Ahmed Nassar\nAI4K Group, IBM Research\nRüschlikon, Switzerland\ngithub.com/DS4SD/docling",
        extractions=[
            lx.data.Extraction(
                extraction_class="title",
                extraction_text="Docling Technical Report",
                attributes={},
            ),
            lx.data.Extraction(
                extraction_class="author",
                extraction_text="Christoph Auer",
                attributes={},
            ),
            lx.data.Extraction(
                extraction_class="author", extraction_text="Maksym Lysak", attributes={}
            ),
            lx.data.Extraction(
                extraction_class="affiliation",
                extraction_text="AI4K Group, IBM Research",
                attributes={},
            ),
            lx.data.Extraction(
                extraction_class="url",
                extraction_text="github.com/DS4SD/docling",
                attributes={"type": "repository"},
            ),
        ],
    )
]

# OpenAI-compatible client pointed at Groq (Llama 3.1 8B Instant).
model = OpenAILanguageModel(
    model_id="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# Run extraction: prompt + examples + model over the first pages of Markdown.
extraction_result = lx.extract(
    text_or_documents=first_pages,
    prompt_description=prompt,
    examples=examples,
    model=model,
)

# Persist annotated results (JSONL) for later inspection or pipelines.
lx.io.save_annotated_documents(
    [extraction_result], output_name="docling_paper_metadata.jsonl"
)

# Print each extracted field (and optional attributes, e.g. URL type).
print("-" * 80)
for extraction in extraction_result.extractions:
    print(f"{extraction.extraction_class}: {extraction.extraction_text}")
    if extraction.attributes:
        print(f"  Atributos: {extraction.attributes}")
