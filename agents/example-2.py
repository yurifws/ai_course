# Tool-calling agent loop with Groq via the OpenAI SDK Responses API.
# The model may request get_stock; we run it locally and send the result back.

import json

import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

# Load GROQ_API_KEY / OPENAI_API_KEY from .env into the environment.
load_dotenv()

# OpenAI client pointed at Groq's OpenAI-compatible endpoint (not api.openai.com).
client = OpenAI(base_url="https://api.groq.com/openai/v1")

# Prefer a Groq model that emits valid OpenAI-style tool calls.
# llama-3.1-8b-instant often fails with tool_use_failed (emits <function=...> XML).
MODEL = "openai/gpt-oss-20b"


def get_stock(ticker: str):
    # Local tool: fetch basic quote fields from Yahoo Finance.
    stock = yf.Ticker(ticker)
    info = stock.info
    output = {
        "ticker": ticker,
        "company_name": info.get("shortName", ticker),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "currency": info.get("currency"),
    }

    return json.dumps(output)


# Tool schema advertised to the model (Responses API function tool format).
tools = [
    {
        "type": "function",
        "name": "get_stock",
        "description": "Return information about a stock",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "A stock ticker symbol like AAPL or TSLA",
                },
            },
            "required": ["ticker"],
        },
    },
]

# Conversation so far: only the user question.
input_list = [{"role": "user", "content": "What is the stock price of Apple?"}]

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
        result = get_stock(**args)
        input_list.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": result,
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
