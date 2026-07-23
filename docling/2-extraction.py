# Docling PDF extraction with image export.
# Same idea as 1-extraction, but we configure the PDF pipeline to keep
# picture images and save each one as a PNG file.

import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem

# PDF-specific options (applied only when the input is a PDF).
pipeline_options = PdfPipelineOptions()
# Upscale factor for extracted images (2.0 = 2x resolution, sharper but larger).
pipeline_options.images_scale = 2.0
# Without this, picture placeholders exist but image bytes are not generated.
pipeline_options.generate_picture_images = True

# Wire the PDF options into the converter via format_options.
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

# Convert a local PDF (needs the pipeline options above for images).
result = converter.convert("./2408.09869v5.pdf")

# Folder where extracted pictures will be written.
os.makedirs("images", exist_ok=True)

picture_counter = 0
# Walk every item in the document tree (text, tables, pictures, ...).
for element, _level in result.document.iterate_items():
    # Keep only picture elements.
    if isinstance(element, PictureItem):
        picture_counter += 1
        # get_image(...) returns a PIL image tied to this conversion result.
        with open(f"images/picture_{picture_counter}.png", "wb") as fp:
            element.get_image(result.document).save(fp, "PNG")
