# Basic example to ask a model to generate a short story

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    #input="Write a short story of three lines about a an AI Agent who wanted to learn singing."
    # input="Give an interesting problem about machine learning of intermediate level. Then summarize the answer in 5 sentences. Give both, question and answer."
    input = "Write Python code to accept three numbers from the user and find which of them is the largest. Print the largest number."
)

print(response)

# From OpenAI documentation: Some of our official SDKs include an output_text property 
#   on model responses for convenience, which aggregates all text outputs from the model 
#   into a single string. This may be useful as a shortcut to 
#   access text output from the model.
print(response.output_text)