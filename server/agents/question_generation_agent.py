from dataclasses import dataclass
from typing import Annotated, List, Union, Optional

from annotated_types import MinLen, MaxLen
from devtools import debug
from pydantic import BaseModel, Field
from typing_extensions import TypeAlias

from pydantic_ai import Agent, ModelRetry, RunContext, format_as_xml

"""
    This agent is responsible for generating questions from the course data.

    The inputs to this agent are the entire course data.
    The input should include a table of contents, the contents of each chapter, 
    and sample questions and/or answers for each chapter.

    The output should be a list of questions for each chapter. 
    The number of questions should be adjustable based on input, but will be set to 10 by default.
"""

@dataclass
class QGenDeps:  
    course_id: str

    
@dataclass
class MCQ(BaseModel):
    """
    A multiple choice question.
    """
    question: str
    choices: Annotated[List[str], MinLen(2), MaxLen(5)]
    answer: int

@dataclass
class TextQuestion(BaseModel):
    """
    A text question.
    """
    question: str
    answer: str


Question = Union[MCQ, TextQuestion]


class ChapterData(BaseModel):
    """
    The chapter data for the given chapter id.
    """
    chapter_id: str
    content: str
    sample_questions: Optional[List[Question]]
    

class CourseData(BaseModel):
    """
    The course data for the given course id.
    """
    course_id: str
    chapters: List[ChapterData]


class QGenOutput(BaseModel):
    """
    The output of the question generation agent for each chapter
    """
    course_id: str
    chapter_id: str
    questions: List[Question]


qgen_agent = Agent(  
    'google-gla:gemini-1.5-flash',
    system_prompt='You are a question generation agent.',
    deps_type=QGenDeps,
    output_type=QGenOutput
)


@qgen_agent.system_prompt  
async def add_course_name(ctx: RunContext[QGenDeps]) -> str:
    return f"The course's name is {ctx.deps.course_id!r}"


@qgen_agent.tool
async def fetch_course_chapters(ctx: RunContext[QGenDeps],) -> CourseData:
    """
    Fetch the course data for the given course id.
    """
    return CourseData.from_course_id(ctx.deps.course_id)


@qgen_agent.tool
async def fetch_chapter_data(ctx: RunContext[QGenDeps], chapter_id: str) -> ChapterData:
    """
    Fetch the chapter data for the given chapter id.
    """
    return ChapterData.from_chapter_id(ctx.deps.course_id, chapter_id)


result = qgen_agent.run_sync('Generate 10 total questions on simple mathematic arithmetics.', deps=QGenDeps(course_id='math-101'))  
debug(result.output)
