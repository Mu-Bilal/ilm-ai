from dataclasses import dataclass
from typing import Annotated, List, Union, Optional

from annotated_types import MinLen, MaxLen
from devtools import debug
from pydantic import BaseModel, Field
from typing_extensions import TypeAlias

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from tools import get_user_mastery, get_data_from_topic

import logfire

"""
    This agent is responsible for generating questions from the course data.
"""

logfire.configure()


@dataclass
class Deps:  
    course_id: str
    user_id: str
    total_questions: int = Field(default=10)

@dataclass
class MCQ(BaseModel):
    """
    A multiple choice question where the question key is the question for the user to solve, the choices key is a list answer choices, and the answer key is the index of the correct answer.
    """
    question: str
    choices: Annotated[List[str], MinLen(2), MaxLen(5)]
    answer: int


@dataclass
class TextQuestion(BaseModel):
    """
    A text question where the question key is the question for the user to solve, and the answer key is the model answer to the question.
    """
    question: str
    answer: str


Question = Union[MCQ, TextQuestion]
Result = Annotated[List[Question], MinLen(1)]


ollama_model = OpenAIModel(
    model_name='qwen3:4b', 
    provider=OpenAIProvider(base_url='http://localhost:11434/v1'),
)


agent = Agent(  
    ollama_model,
    deps_type=Deps,
    output_type=Result,
    tools=[
        get_user_mastery,
        get_data_from_topic,
    ],
    instrument=True
)


@agent.system_prompt
async def add_system_prompt() -> str:
    prompt = f"""
    You are a teacher who is responsible for generating questions for courses.
    Using this course name, fetch the chapters with their notes, as well as your student's mastery notes.
    These notes will give you scores on how well the student has mastered each chapter, along with notes on what the student needs to work on 
    in each chapter.
    Based on these notes, you will need to plan out how many questions to generate for each chapter.
    Create a mix of multiple choice and text questions, and allocate more questions to the chapters that the student is weaker at.
    You will then need to fetch the data for the most important topics in each chapter from a vector database.
    Finally, using the mastery notes, chapter notes, and the topic data, you will generate questions.
    """
    return prompt


def generate_questions(user_id: str, course_id: str, total_questions: int):
    deps = Deps(user_id=user_id, course_id=course_id, total_questions=total_questions)
    result = agent.run_sync(
        f"Generate {deps.total_questions} questions for the {deps.course_id} course.", 
        deps=deps, 
        output_type=Result)

    debug(result.output)

    return result.output


if __name__ == "__main__":
    print(generate_questions('1', 'Deep Learning', 10))
