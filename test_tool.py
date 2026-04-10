from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
search_runner = DuckDuckGoSearchResults()

@tool("DuckDuckGoSearch")
def search_tool(query: str) -> str:
    """Use this tool to search the internet for information and actual URLs. Input should be a search query."""
    return search_runner.run(query)

from crewai import Agent
scraper = Agent(
    role="Data Gathering Specialist",
    goal="Test",
    backstory="Test",
    tools=[search_tool],
)
print("Agent created successfully with tool")
