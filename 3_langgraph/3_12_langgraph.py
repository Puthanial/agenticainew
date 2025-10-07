import gradio as gr
from typing import TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

# State
class ProductState(TypedDict):
    query: str
    results: str

# Load embeddings once
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.load_local(
    "c:/code/agenticai/3_langgraph/product_embeddings_faiss", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# Memory for HITL
memory = {}

# Node 1: Search
def search_products(state: ProductState) -> ProductState:
    query = state["query"]
    
    # Check memory first
    if query in memory:
        state["results"] = memory[query]
        return state
    
    # Search FAISS
    results = vectordb.similarity_search(query, k=3)
    titles = [doc.metadata["title"] for doc in results]
    state["results"] = "\n".join([f"• {title}" for title in titles])
    return state

# Node 2: Format
def format_response(state: ProductState) -> ProductState:
    if state["results"]:
        state["results"] = f"Found products:\n{state['results']}"
    else:
        state["results"] = "No products found"
    return state

# Build graph
graph = StateGraph(ProductState)
graph.add_node("search", search_products)
graph.add_node("format", format_response)
graph.set_entry_point("search")
graph.add_edge("search", "format")
graph.add_edge("format", END)
runnable = graph.compile()

# Search function
def search(query):
    if not query.strip():
        return "Please enter a query"
    
    result = runnable.invoke({"query": query})
    return result["results"]

# Approve function
def approve(query, results):
    if not query.strip():
        return "No query to approve", format_memory()
    
    memory[query] = results
    return f"✓ Approved and saved for: {query}", format_memory()

# Edit function
def edit(query, edited_results):
    if not query.strip():
        return "No query to edit", format_memory()
    
    memory[query] = edited_results
    return f"✓ Edited and saved for: {query}", format_memory()

# Format memory for display
def format_memory():
    if not memory:
        return "Memory is empty"
    
    lines = []
    for query, results in memory.items():
        lines.append(f"Query: {query}")
        lines.append(results)
        lines.append("-" * 40)
    return "\n".join(lines)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Product Search with HITL Support")
    gr.Markdown("Search for products, then approve or edit the results to save them to memory.")

    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Enter your query", placeholder="e.g., wireless headphones")
            search_btn = gr.Button("Search", variant="primary")
            
            result_box = gr.Textbox(label="Search Results", lines=6)
            
            with gr.Row():
                approve_btn = gr.Button("Approve", variant="secondary")
                edit_btn = gr.Button("Save Edited", variant="secondary")
            
            edit_box = gr.Textbox(label="Edit Results (optional)", lines=6, placeholder="Edit the results above if needed")
            status_box = gr.Textbox(label="Status", lines=2)
        
        with gr.Column():
            memory_display = gr.Textbox(label="Saved Memory", lines=20, interactive=False)

    # Connect functions
    def run_search(query):
        results = search(query)
        return results, results, "", format_memory()

    search_btn.click(
        run_search, 
        inputs=query_input, 
        outputs=[result_box, edit_box, status_box, memory_display]
    )

    approve_btn.click(
        approve, 
        inputs=[query_input, result_box], 
        outputs=[status_box, memory_display]
    )

    edit_btn.click(
        edit, 
        inputs=[query_input, edit_box], 
        outputs=[status_box, memory_display]
    )

if __name__ == "__main__":
    demo.launch()