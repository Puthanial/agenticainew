from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Define the state
class AgentState(TypedDict):
    code_url: str
    code_content: str
    security_issues: str
    suggestions: str
    final_report: str

# Initialize LLM
load_dotenv(override=True)
llm = ChatOpenAI(model="gpt-4", temperature=0)

def fetch_code(repo_url: str, path: str = "") -> str:
    """Recursively fetch Java code from a GitHub repo."""
    repo_path = repo_url.replace("https://github.com/", "")
    api_url = f"https://api.github.com/repos/{repo_path}/contents/{path}"
    code_files = []

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        items = response.json()

        for item in items:
            if item["type"] == "file" and item["name"].endswith(".java"):
                file_content = requests.get(item["download_url"]).text
                code_files.append(f"// {item['path']}\n{file_content}\n")
            elif item["type"] == "dir":
                sub_code = fetch_code(repo_url, item["path"])
                code_files.append(sub_code)

        code_str = "\n".join(code_files)
        # limit to ~50,000 characters for safety
        return code_str[:10000]

    except Exception as e:
        return f"Error fetching code: {e}"


# Agent 1: Security Reviewer
def security_review_function(state: AgentState):
    """Reviews actual code for security vulnerabilities"""
    print("Agent 1: Fetching and reviewing code for security issues...")
    
    code = fetch_code(state["code_url"])
    
    prompt = f"""
    You are a senior application security analyst. 
    Analyze the following **actual Java code** for security vulnerabilities.

    Provide:
    - The vulnerable code snippet or line
    - Type of vulnerability (e.g., SQL Injection, XSS, Insecure Deserialization)
    - Severity (High/Medium/Low)
    - Why it’s a problem
    - How to fix it briefly

    Code to review:
    {code}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "code_content": code,
        "security_issues": response.content
    }

# Agent 2: Solution Provider
def solution_function(state: AgentState):
    """Provides suggestions to fix identified issues"""
    print("Agent 2: Generating solutions...")
    
    prompt = f"""You are a security remediation expert. Based on these security issues, provide specific fixes.

Security Issues Found:
{state['security_issues']}

For each issue, provide:
- Specific code changes needed
- Best practices to follow
- Example of secure code

Be practical and specific."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"suggestions": response.content}

# Agent 3: Finalizer
def finalize_function(state: AgentState):
    """Reviews and creates final report"""
    print("Agent 3: Finalizing report...")
    
    prompt = f"""Create a final executive security report.

Repository: {state['code_url']}

Security Issues:
{state['security_issues']}

Suggested Fixes:
{state['suggestions']}

Create a comprehensive report with:
1. Executive Summary
2. Critical Vulnerabilities (prioritized)
3. Recommended Actions
4. Implementation Roadmap

Format as a professional security assessment report."""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"final_report": response.content}

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("review", security_review_function)
workflow.add_node("suggest", solution_function)
workflow.add_node("finalize", finalize_function)

# Add edges
workflow.set_entry_point("review")
workflow.add_edge("review", "suggest")
workflow.add_edge("suggest", "finalize")
workflow.add_edge("finalize", END)

# Compile the graph
app = workflow.compile()

# Run the workflow
if __name__ == "__main__":
    initial_state = {
        "code_url": "https://github.com/vulnerable-apps/verademo",
        "code_content": "",
        "security_issues": "",
        "suggestions": "",
        "final_report": ""
    }
    
    result = app.invoke(initial_state)
    
    print("\n" + "="*70)
    print(result["final_report"])
    print("="*70)
    