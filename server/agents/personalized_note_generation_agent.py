from dataclasses import dataclass

from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from agents.tools import get_user_mastery_for_chapter, search_course_content

# import logfire

# logfire.configure()

@dataclass
class Deps:
    user_id: str
    course_id: str
    chapter_id: str



ollama_model = OpenAIModel(
    model_name='qwen3:4b', 
    provider=OpenAIProvider(base_url='http://localhost:11434/v1'),
)
agent = Agent(ollama_model, output_type=str, tools=[get_user_mastery_for_chapter, search_course_content], deps_type=Deps)

Agent.instrument_all()

async def getNotes(user_id: str, course_id: str, chapter_id: str):
    """
    Generate personalized notes for a user based on their mastery level and chapter content.
    
    Args:
        user_id: The ID of the user
        course_id: The ID of the course
        chapter_id: The ID of the chapter
        
    Returns:
        str: Personalized notes for the chapter
    """
    try:
        # Run the agent to generate personalized notes
        result = await agent.run(
            "Give personalized notes. First get user mastery to"
            " understand what specific content the user would benefit from learning at this"
            " time (based on their mastery and shortcomings). Then search for relevant course content by using multiple related query terms."
            " Generate good summary of the notes that would be useful to this user in"
            " particular. Focus on making the notes cover just the things the user struggles with.",
            deps=Deps(user_id=user_id, course_id=course_id, chapter_id=chapter_id)
        )
        return result.output
    except Exception as e:
        print(f"Error in getNotes: {str(e)}")
        raise