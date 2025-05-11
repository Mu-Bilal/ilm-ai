from dataclasses import dataclass
import asyncio

from pydantic import Field

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from tools.rag import search_course_content

import logfire

logfire.configure(token='pylf_v1_eu_W7WPNNgs18K4HcXghtmnzsZ5HdQY1f2ZZkT7pBflsZdt')

@dataclass
class Deps:
    question: str = Field(..., description="The question to answer")


ollama_model = OpenAIModel(
    model_name='qwen3:4b', 
    provider=OpenAIProvider(base_url='http://localhost:11434/v1'),
)
agent = Agent(ollama_model, output_type=str, tools=[search_course_content], deps_type=Deps)

Agent.instrument_all()

async def getResponse(question: str):
    """
    Generate personalized notes for a user based on their mastery level and chapter content.
    
    Args:
        question: The question to answer
        
    Returns:
        str: Personalized notes for the chapter
    """
    try:
        # Run the agent to generate personalized notes
        result = await agent.run(
            "The student is asking a question about" + question + " Search the course material and answer the question based on the course material.",
        )
        print(result.output)
        return result.output
    except Exception as e:
        print(f"Error in getResponse: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(getResponse("Why is a sum of two gaussian random variables a gaussian random variable?"))