from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from tools import get_user_mastery, get_chapter_notes

import logfire

logfire.configure()

@dataclass
class Deps:
    user_id: str
    course_id: str
    chapter_id: str



ollama_model = OpenAIModel(
    model_name='qwen3:4b', 
    provider=OpenAIProvider(base_url='http://localhost:11434/v1'),
)
agent = Agent(ollama_model, output_type=str, tools=[get_user_mastery, get_chapter_notes], deps_type=Deps)

Agent.instrument_all()

# Test with a simple query
result = agent.run_sync("""Give personalized notes. First get user mastery to
understand what specific content the user would benefit from learning at this
time (based on their mastery and shortcomings), and then get chapter notes.
Generate good summary of the notes that would be useful to this user in
particular. Focus on making the notes cover just the things the user struggles with.""", deps=Deps(user_id='1', course_id='pai', chapter_id='1 Fundamentals'))

print("\n=== Structured Output ===")
print(f"Notes: {result.output}")
print("=======================\n")

print("\n=== Usage Statistics ===")
print(result.usage())
print("=======================\n")

print("\n=== All Messages ===")
print(result.all_messages())
print("=======================\n")