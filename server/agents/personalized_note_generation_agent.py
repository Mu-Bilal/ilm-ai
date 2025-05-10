#!/usr/bin/env python3

from dataclasses import dataclass

import asyncpg
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from get_user_mastery import get_user_mastery
from get_chapter_notes import get_chapter_notes
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
result = agent.run_sync("""Give personalized notes. First get chapter notes, then
get user mastery to understand which section of these notes to focus on.
Generate good summary of the notes that would be useful to this user in
particular.""", deps=Deps(user_id='1', course_id='1', chapter_id='chapter_1'))

print("\n=== Structured Output ===")
print(f"Notes: {result.output}")
print("=======================\n")

print("\n=== Usage Statistics ===")
print(result.usage())
print("=======================\n")

print("\n=== All Messages ===")
print(result.all_messages())
print("=======================\n")