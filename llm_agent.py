from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.agent import Agent
from config import llm_model, llm_base_url, table_name
import asyncio
from pydantic import BaseModel,Field
from typing import Literal, Optional

class Graph(BaseModel):
    type: Literal["bar", "line", "scatter", "pie"] = Field(..., description="Type of graph to be plotted.")
    x: Optional[str] = Field(None, description="Column name to be used for the x-axis.")
    y: Optional[str] = Field(None, description="Column name to be used for the y-axis.")

class Answer(BaseModel):
    query: str = Field(..., description="Generated SQL query based on user input.")
    text: str = Field(..., description="One line answer to the question using the data")
    graph: Graph = Field(..., description="Graph metadata for visualization, including type and axes.")


#Initializes the AI agent with the system prompt.
def initialize_agent(columns: str, describe_data: str):
    system_prompt = f"""
    The dataset has the following columns: {columns}.
    Here is the statistical summary of the dataset:
    {describe_data}
    Generate a valid SQL that can run on the SQLite Table `{table_name}` and print all columns
    Specify the type of graph and the column(s) to be plotted.
    """

    ollama_model = OpenAIModel(
        model_name=llm_model,
        provider=OpenAIProvider(base_url=llm_base_url),
    )

    return Agent(ollama_model, system_prompt=system_prompt,result_type=Answer)

def ask_ai(agent: Agent, user_prompt: str):
    try:
        response = asyncio.run(agent.run(user_prompt))
        print("Raw Response:", response)
        return response.data
    except Exception as e:
        print("Error in ask_ai:", str(e))
        return Answer(
            query="",
            text="Failed to generate a response due to an error.",
            graph=Graph(type="bar", x=None, y=None)
        )


