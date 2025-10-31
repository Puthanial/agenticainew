# Use house prices chromadb to respond to users' questions 
# via a langgraph agent
# pip install langgraph openai chromadb sentence-transformers python-dotenv

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer

# Load environment variables (expects OPENAI_API_KEY in .env)
load_dotenv(override=True)


# ------------------------------
# State
# ------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, "Conversation messages"]


# ------------------------------
# ChromaDB Setup (already created)
# ------------------------------

client = chromadb.PersistentClient(path="c://code//agenticai//3_langgraph//chroma")
collection = client.get_collection("house_prices")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------
# Tool Function
# ------------------------------

@tool
def search_house_prices(query: str) -> str:
    """Search house prices database using semantic similarity"""
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=10)

    # Filter by cosine similarity threshold (ChromaDB returns distances, not similarities)
    # For cosine distance: similarity = 1 - distance
    # We want high similarity (low distance), so threshold on distance < 0.5
    # which means similarity > 0.5
    SIMILARITY_THRESHOLD = 0.5
    
    out = []
    documents = results["documents"][0]
    distances = results["distances"][0] if "distances" in results else [0] * len(documents)
    
    for i, (doc, dist) in enumerate(zip(documents, distances)):
        similarity = 1 - dist
        if similarity >= SIMILARITY_THRESHOLD:
            out.append(f"{i+1}. [Similarity: {similarity:.2f}] {doc}")
    
    if not out:
        return "No highly relevant results found. Try rephrasing your query or being more specific."
    
    return "\n".join(out)


# ------------------------------
# Build Graph
# ------------------------------

def build_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Agent that knows about our tool
    agent = create_react_agent(
        model=llm,
        tools=[search_house_prices],
    )

    return agent


# ------------------------------
# Run with Streaming
# ------------------------------

if __name__ == "__main__":
    graph = build_graph()

    print("Real Estate Assistant (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting.")
            break

        # Streaming the response
        print("Bot: ", end="", flush=True)
        
        last_content = ""
        for chunk in graph.stream(
            {"messages": [("user", user_input)]},
            stream_mode="values"
        ):
            # Get the last message in the state
            if chunk["messages"]:
                last_msg = chunk["messages"][-1]
                # Check if it's an AI message with content (skip user messages)
                if hasattr(last_msg, "type") and last_msg.type == "ai":
                    if hasattr(last_msg, "content") and last_msg.content:
                        # Only print new content (not already printed)
                        if last_msg.content != last_content:
                            # Print only the new part
                            if last_msg.content.startswith(last_content):
                                new_part = last_msg.content[len(last_content):]
                                print(new_part, end="", flush=True)
                            else:
                                # If content changed completely, print it all
                                if last_content:
                                    print()  # New line for new message
                                print(last_msg.content, end="", flush=True)
                            last_content = last_msg.content
        
        print()  # newline after streaming