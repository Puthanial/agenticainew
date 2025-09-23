# Analyze an image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    instructions="Talk like a drunk person.",
    input="Are semicolons optional in JavaScript?",
)

print(response.output_text)