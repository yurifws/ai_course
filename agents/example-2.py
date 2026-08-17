import json

import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(base_url="https://api.groq.com/openai/v1")

MODEL = "openai/gpt-oss-20b"


def get_stock(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    output = {
        "ticker": ticker,
        "company_name": info.get("shortName", ticker),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "currency": info.get("currency"),
    }

    return json.dumps(output)


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

input_list = [{"role": "user", "content": "What is the stock price of Apple?"}]

response = client.responses.create(
    model=MODEL,
    tools=tools,
    input=input_list,
)

input_list += response.output

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

final_response = client.responses.create(
    model=MODEL,
    input=input_list,
    tools=tools,
    instructions=(
        "Respond with a short analysis of the stock based on the tool information provided."
    ),
)

print(final_response.output_text)
