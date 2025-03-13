from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.agent import Agent
from config import llm_model, llm_base_url, table_name
import asyncio
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional


class Graph(BaseModel):
    type: Literal["bar", "line", "scatter", "pie"] = Field(..., description="Type of graph to be plotted.")
    x: Optional[str] = Field(None, description="Column name to be used for the x-axis.")
    y: Optional[str] = Field(None, description="Column name to be used for the y-axis.")


class Answer(BaseModel):
    query: str = Field(..., description="Generated SQL query based on user input.")
    text: str = Field(..., description="Textual explanation or summary of the SQL query.")
    graph: Optional[Graph] = Field(None, description="Graph metadata for visualization, including type and axes.")


# Initializes the AI agent with the system prompt.
def initialize_agent(columns: str, describe_data: str):
    system_prompt = f"""
    You are an AI assistant that processes CSV data as an SQL database.
    The dataset has the following columns: {columns}.
    Here is the statistical summary of the dataset:
    {describe_data}
    When asked a question, generate a valid SQL query that can run on the SQLite Table `{table_name}`.
    Specify the type of graph and the column(s) to be plotted.
    """

    ollama_model = OpenAIModel(
        model_name=llm_model,
        provider=OpenAIProvider(base_url=llm_base_url),
    )

    return Agent(ollama_model, system_prompt=system_prompt, result_type=Answer)


# Asks the AI a question and validates the response
def ask_ai(agent: Agent, user_prompt: str):
    try:
        response = asyncio.run(agent.run(user_prompt))
        print("Raw AI Response:", response)

        validated_response = Answer.model_validate(response.data)

        return validated_response

    except ValidationError as e:
        print("Validation Error:", e)
        return {"error": "AI response validation failed. Please try again."}

    except Exception as e:
        print("Error in ask_ai:", e)
        return {"error": "An unexpected error occurred. Please try again."}
