# RAG tool-calling agent with Groq via the OpenAI SDK Responses API.
# The model may request search_kb; we query a local search API and send hits back.

import json

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load GROQ_API_KEY / OPENAI_API_KEY from .env into the environment.
load_dotenv()

# OpenAI client pointed at Groq's OpenAI-compatible endpoint (not api.openai.com).
client = OpenAI(base_url="https://api.groq.com/openai/v1")

# Prefer a Groq model that emits valid OpenAI-style tool calls.
# llama-3.1-8b-instant often fails with tool_use_failed (emits <function=...> XML).
MODEL = "openai/gpt-oss-20b"


def search_kb(query: str):
    # Local tool: POST the query to the knowledge-base search API (must be running).
    response = requests.post(
        "http://localhost:8000/search",
        json={"query": query, "limit": 3},
    )
    return response.json()


# Tool schema advertised to the model (Responses API function tool format).
# Unlike chat.completions, name/description/parameters sit at the top level.
tools = [
    {
        "type": "function",
        "name": "search_kb",
        "description": "Search the knowledge base for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search the knowledge base for",
                }
            },
            "required": ["query"],
        },
    },
]


# Conversation so far: only the user question.
input_list = [{"role": "user", "content": "What are AAPL main financial risks?"}]

# First turn: model may emit a function_call instead of a final answer.
response = client.responses.create(
    model=MODEL,
    tools=tools,
    input=input_list,
)

# Keep the model's output (including function_call items) in the conversation.
input_list += response.output

# Run each requested tool locally and append function_call_output messages.
for item in response.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        result = search_kb(**args)

        # Keep only the retrieved text snippets for the second model turn.
        texts = [r["text"] for r in result["results"]]

        input_list.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({"results": texts}, ensure_ascii=False),
            }
        )

# Second turn: model answers using the tool results.
final_response = client.responses.create(
    model=MODEL,
    input=input_list,
    tools=tools,
    instructions=(
        "Respond with a short analysis of the stock based on the tool information provided."
    ),
)

print(final_response.output_text)
