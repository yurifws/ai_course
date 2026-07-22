# First contact with an LLM API using Groq.
# This script sends a chat prompt and prints the model's reply.

from groq import Groq

# Create the API client.
# Groq reads the API key from the GROQ_API_KEY environment variable.
client = Groq()

# Ask the model for a chat completion (generate a response).
response = client.chat.completions.create(
    # Which model to use (fast Llama 3.1 variant hosted by Groq).
    model="llama-3.1-8b-instant",
    # Conversation messages: system sets behavior; user is the question.
    messages=[
        {"role": "system", "content": "Act as a machine learning specialist"},
        {"role": "user", "content": "With a simple way, what is machine learning?"},
    ],
    # temperature: lower = more focused/deterministic; higher = more creative.
    temperature=0.5,
    # top_p (nucleus sampling): considers only the most likely tokens
    # whose probabilities sum to this value (0.8 = top 80%).
    top_p=0.8,
)

# The API may return several alternatives; we print the first one's text.
print(response.choices[0].message.content)
