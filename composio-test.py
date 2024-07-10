from openai import OpenAI, AzureOpenAI
import os

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain_core.tools import Tool

openai_api_key = "0e54ece989454fb7bce8b01f864c6085"
openai_ep = "https://lunartree-gpt-35-turbo-2.openai.azure.com/"
openai_ver = "2024-02-01"
serper_api_key = "c3f350324a1a0494d4122f3b58a6261e3bb1bd15116aefe933ff83aff41b5cd4"

if not openai_api_key:
    raise ValueError("Please enter your OpenAI API key.")
else:
    llm = AzureOpenAI(
        azure_endpoint=openai_ep,
        api_key=openai_api_key,
        api_version=openai_ver,
    )

google_search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
tools = [
    Tool(
        name="Intermediate Answer",
        func=google_search.run,
        description="useful for when you need to ask with search"
    )
]

agent = initialize_agent(tools = tools, 
                         llm = llm, 
                         agent=AgentType.SELF_ASK_WITH_SEARCH, 
                         verbose=True)
agent.run("What is the hometown of the 2001 US PGA champion?")