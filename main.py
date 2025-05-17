from dotenv import load_dotenv
import os

load_dotenv() 

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import AzureChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents.agent_types import AgentType
from langchain_core.runnables import RunnableConfig

from typing_extensions import TypedDict
from typing import List

class QueryResults(TypedDict):
    query: str
    results: List  # You can specify the type of list elements if known, e.g. List[str]

# Example usage:
example: QueryResults = {
    "query": "example query",
    "results": []
}


# Setup memory, LLM, and tools (as before)
memory = MemorySaver()

llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.7,
)

tools = load_tools(["serpapi", "llm-math"], llm=llm)
print("Loaded tools:", [tool.name for tool in tools])

# Build your graph nodes manually (e.g., a simple agent node)
def agent_node_fn(inputs: QueryResults) -> QueryResults:
    query = inputs["query"]
    
    # Use the SerpAPI tool
    serp_tool = next(tool for tool in tools if tool.name == "Search")
    serp_result = serp_tool.run(query)

    # Optionally use LLM to summarize or enhance the result
    llm_summary = llm.invoke(f"Summarize this result: {serp_result}")

    return {
        "query": query,
        "results": [serp_result, llm_summary]
    }


graph = StateGraph(QueryResults)
graph.add_node("agent", agent_node_fn)
graph.set_entry_point("agent")
graph.set_finish_point("agent")

app = graph.compile()


if __name__ == "__main__":
    print("Welcome to your query agent. Type 'exit' to quit.")
    
    while True:
        user_input = input("Query: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        input_data = {
            "query": user_input,
            "results": []
        }

        try:
            result = app.invoke(input_data)
            print("\nAnswer:")
            for item in result["results"]:
                print("-", item)
        except Exception as e:
            print("Error:", e)
