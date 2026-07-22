# Same idea as llm-01, but using the OpenAI SDK pointed at Groq's OpenAI-compatible API.
# Uses the newer Responses API (instructions + input) instead of chat.completions.

import openai

# OpenAI client configured to talk to Groq (not api.openai.com).
# Auth still comes from the environment (typically OPENAI_API_KEY or GROQ_API_KEY
# depending on how you set it up; Groq's OpenAI-compatible endpoint accepts Groq keys).
client = openai.OpenAI(base_url="https://api.groq.com/openai/v1")

# Create a response with the Responses API.
response = client.responses.create(
    # Same fast Llama model hosted by Groq.
    model="llama-3.1-8b-instant",
    # System-like guidance for how the model should answer.
    instructions="Reply in a simple way, only a one short paragraph",
    # User prompt as a plain string.
    input="What is machine learning?",
    # Alternative: pass a full message list (developer/user roles) instead of
    # separate instructions + input strings.
    # input=[
    #    {"role": "developer", "content": "Talk like a pirate"},
    #    {"role": "user", "content": "With a simple way, what is machine learning?"},
    # ],
)

# Convenience property: the model's text reply.
print(response.output_text)
