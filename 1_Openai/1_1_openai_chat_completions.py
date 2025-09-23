# pip install openai

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

openai = OpenAI()

# First a very basic question
user_messages = [{
                  "role": "user", 
		    	  "content": "What is 2+2?"
		        }]

response = openai.chat.completions.create(
    model="gpt-4.1-nano",
    messages=user_messages
)

print(response)
print(response.choices[0].message.content)

# Now let us ask a tougher question
question = "Please propose a hard, challenging question to assess someone's IQ. Respond only with the question."
messages = [{"role": "user", "content": question}]

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

# Save the question returned by the LLM into a variable called question
question = response.choices[0].message.content

print(f"OpenAI Question: {question}")  

# now form a new message list
messages = [{
             "role": "user", 
			 "content": question
		   }]

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

answer = response.choices[0].message.content
print(f"GPT answer: {answer} ")

# Now do this using an open source LLM
# Visit https://ollama.com/ and download and install it
# Then open a terminal and run the command ollama pull llama3.2 then ollama list

ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
model_name = "llama3.2:latest"

question = "Please propose a hard, challenging question to assess someone's IQ. Respond only with the question."
messages = [{
              "role": "user", 
              "content": question
           }]

response = ollama.chat.completions.create(
    model=model_name, 
    messages=messages
)
question = response.choices[0].message.content

print(f"Ollama Question: {question}")

# now form a new message list
messages = [{
             "role": "user", 
             "content": question
           }]

# Now ask the LLM to answer the question
response = ollama.chat.completions.create(
    model=model_name, 
    messages=messages
)

answer = response.choices[0].message.content
print(f"Ollama Answer: {answer}")
