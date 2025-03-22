from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.agent import Agent
from config import llm_model, llm_base_url, table_name
import asyncio
from pydantic import BaseModel, Field
from typing import  Optional,List,Union
import ollama

class Graph(BaseModel):
    x: Optional[str] = Field(None, description="Column name to be used for the x-axis.")
    y: Optional[str] = Field(None, description="Column name to be used for the y-axis.")
    x_data: Optional[List[Union[str, float, int]]] = Field(None, description="Data points for the x-axis.")
    y_data: Optional[List[Union[str, float, int]]] = Field(None, description="Data points for the y-axis.")

class Answer(BaseModel):
    query: str = Field(..., description="Generated SQL query based on user input.")
    graph: Graph = Field(..., description="Graph metadata for visualization, including type and axes.")
    
system_prompt=None

# Initializes the AI agent with the system prompt.
def initialize_agent(columns: str, describe_data: str):
    global system_prompt
    system_prompt = f"""
    The dataset has the following columns: {columns}.
    Statistical summary for one line answer:
    {describe_data}
    Generate a valid SQL query that can run on the SQLite Table `{table_name}` and return all columns.
    """

    ollama_model = OpenAIModel(
        model_name=llm_model,
        provider=OpenAIProvider(base_url=llm_base_url),
    )

    agent = Agent(
        ollama_model,
        system_prompt=system_prompt,
        result_type=Answer,
    )
    
    return agent



async def ask_ai_async(agent: Agent, user_prompt: str):
    global system_prompt
    try:
        response =  ollama.chat(
            model=llm_model,
            messages=[        
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
            format=Answer.model_json_schema(),
        )
        print("Validated Response:", response.message.content)
        return Answer.model_validate_json(response.message.content)
    except Exception as e:
        print("General Error in ask_ai:", str(e))
        return Answer(
            query="",
            graph=Graph( x=None, y=None,x_data=None,y_data=None)
        )

def ask_ai(agent: Agent, user_prompt: str):
    return asyncio.run(ask_ai_async(agent, user_prompt))
