from langgraph.graph import StateGraph, END
from typing import TypedDict
import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# --------------------------
# Define shared state
# --------------------------
class AgentState(TypedDict):
    topic: str
    headlines: str
    summary: str
    sentiment: str
    final_report: str
    sentiment_label: str  # "positive" / "negative"

# --------------------------
# Initialize LLM
# --------------------------
load_dotenv(override=True)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

news_key = os.getenv("NEWS_API_KEY")

# --------------------------
# Agent 1: Fetch latest news
# --------------------------
def fetch_news_agent(state: AgentState):
    print("Agent 1: Fetching latest news headlines...")
    url = f"https://newsapi.org/v2/everything?q={state['topic']}&apiKey={news_key}&pageSize=5"
    try:
        resp = requests.get(url)
        data = resp.json()
        articles = [a["title"] for a in data.get("articles", [])]
        headlines = "\n".join(articles)
    except Exception as e:
        headlines = f"Error fetching news: {e}"

    return {"headlines": headlines}

# --------------------------
# Agent 2: Summarize news
# --------------------------
def summarizer_agent(state: AgentState):
    print("Agent 2: Summarizing news...")
    prompt = f"""
    Summarize the following news headlines about {state['topic']} in 5 concise bullet points:
    {state['headlines']}
    """
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"summary": resp.content}

# --------------------------
# Agent 3: Analyze sentiment
# --------------------------
def sentiment_agent(state: AgentState):
    print("Agent 3: Analyzing sentiment...")
    prompt = f"""
    Determine the overall sentiment (Positive, Neutral, or Negative)
    of these summarized news points:

    {state['summary']}
    """
    resp = llm.invoke([HumanMessage(content=prompt)])
    sentiment_text = resp.content.strip().lower()
    label = "positive" if "positive" in sentiment_text else "negative"
    return {"sentiment": sentiment_text, "sentiment_label": label}

# --------------------------
# Agent 4a: Investor Summary
# --------------------------
def investor_summary_agent(state: AgentState):
    print("Agent 4A: Creating investor-friendly summary...")
    prompt = f"""
    Based on this summary:
    {state['summary']}
    
    Write a short investor-oriented insight:
    - Focus on risks and opportunities
    - Predict possible market impact
    """
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": resp.content}

# --------------------------
# Agent 4b: Public Summary
# --------------------------
def general_summary_agent(state: AgentState):
    print("Agent 4B: Creating public summary...")
    prompt = f"""
    Based on this summary:
    {state['summary']}

    Write a 5-sentence public news digest in a neutral, friendly tone.
    """
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": resp.content}

# --------------------------
# Build the graph
# --------------------------
workflow = StateGraph(AgentState)

workflow.add_node("fetch_news", fetch_news_agent)
workflow.add_node("summarize", summarizer_agent)
workflow.add_node("analyze_sentiment", sentiment_agent)
workflow.add_node("investor_summary", investor_summary_agent)
workflow.add_node("general_summary", general_summary_agent)

workflow.set_entry_point("fetch_news")
workflow.add_edge("fetch_news", "summarize")
workflow.add_edge("summarize", "analyze_sentiment")

# Branching — sentiment decides next node
workflow.add_conditional_edges(
    "analyze_sentiment",
    lambda state: "investor_summary" if state["sentiment_label"] == "positive" else "general_summary",
    {"investor_summary": "investor_summary", "general_summary": "general_summary"},
)

workflow.add_edge("investor_summary", END)
workflow.add_edge("general_summary", END)
app = workflow.compile()

# --------------------------
# Run it
# --------------------------
if __name__ == "__main__":
    initial_state = {
        "topic": "artificial intelligence",
        "headlines": "",
        "summary": "",
        "sentiment": "",
        "final_report": "",
        "sentiment_label": ""
    }

    result = app.invoke(initial_state)
    print("\n" + "="*70)
    print(result["final_report"])
    print("="*70)
