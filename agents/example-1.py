# Structured extraction with Groq via the OpenAI SDK Responses API.
# Instead of free-form text, the model must fill a Pydantic schema (json_schema).

from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

# Load GROQ_API_KEY / OPENAI_API_KEY from .env into the environment.
load_dotenv()

# OpenAI client pointed at Groq's OpenAI-compatible endpoint (not api.openai.com).
client = OpenAI(base_url="https://api.groq.com/openai/v1")


# Schema for the structured result: field names + types the model must return.
class CaptureEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


# responses.parse asks for structured output matching text_format (Pydantic model).
# Needs a Groq model that supports json_schema (e.g. openai/gpt-oss-*).
response = client.responses.parse(
    # Replacement for retired meta-llama/llama-4-scout-17b-16e-instruct.
    model="openai/gpt-oss-120b",
    # User text to extract from.
    input="Daniel and Alberto are going to record a class video on Tuesday",
    # System-like guidance for the extraction task.
    instructions="Extract information from event.",
    # Target schema: SDK converts this to a json_schema response format.
    text_format=CaptureEvent,
)

# Parsed Pydantic instance (not raw JSON string).
event = response.output_parsed

# Pretty-print the structured event as JSON.
print(event.model_dump_json(indent=2))
