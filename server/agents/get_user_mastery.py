from dataclasses import dataclass

from pydantic_ai import RunContext
@dataclass
class Deps:
    user_id: str
    course_id: str
    chapter_id: str

async def get_user_mastery(ctx: RunContext[Deps]) -> str:
    """
    Call this function to get, for each chapter, the user's mastery score and a note on what the user needs to work on.
    """
    user_id = ctx.deps.user_id
    course_id = ctx.deps.course_id
    chapter_id = ctx.deps.chapter_id
    return {
        "chapter_1": {
            "mastery": 0.7,
            "notes": "The user does not understand normalization layers well.",
        },
        "chapter_2": {
            "mastery": 0.5,
            "notes": "The user does not understand the difference between a Pydantic model and a Pydantic BaseModel.",
        },
    }[chapter_id]