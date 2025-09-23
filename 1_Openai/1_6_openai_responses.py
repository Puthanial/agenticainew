# Extend the LLM with tools

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    tools=[{"type": "web_search"}],
    input="What was a positive news story from today?"
)

print(response.output_text)