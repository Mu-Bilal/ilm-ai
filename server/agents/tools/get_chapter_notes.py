from dataclasses import dataclass
from collections import defaultdict

from pydantic_ai import RunContext

import os

# Get directory of current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate to the main.md file
main_md_path = os.path.join(current_dir, "..", "main.md")

@dataclass
class Deps:
    course_id: str
    chapter_id: str


CHAPTER_NOTES = {}


def load_chapter_notes():
    chapter_id = None
    with open(main_md_path, "r") as f:
        for line in f:
            if line.startswith("### *"):
                chapter_id = line.strip().replace("### ", "").replace("*", "")
                if chapter_id[0] not in "0123456789":
                    chapter_id = None
            else:
                line = line.strip()
                if chapter_id:
                    if chapter_id not in CHAPTER_NOTES:
                        CHAPTER_NOTES[chapter_id] = []
                    CHAPTER_NOTES[chapter_id].append(line)
    
    for chapter_id, notes in CHAPTER_NOTES.items():
        CHAPTER_NOTES[chapter_id] = "\n".join(notes)
    

load_chapter_notes()

for chapter_id, notes in CHAPTER_NOTES.items():
    print(f"Chapter {chapter_id}:")
    print(len(notes))
    print(notes[:100])
    print("\n")


def get_chapter_ids():
    return list(CHAPTER_NOTES.keys())


def get_chapter_notes_sync(course_id: str, chapter_id: str) -> str:
    return CHAPTER_NOTES[chapter_id]


async def get_chapter_notes(ctx: RunContext[Deps]) -> str:
    """
    Call this function to get the notes for a chapter.
    """
    course_id = ctx.deps.course_id
    chapter_id = ctx.deps.chapter_id
    to_return = get_chapter_notes_sync(course_id, chapter_id)
    print(course_id, chapter_id)
    print("Returning string of length", len(to_return))
    return to_return


if __name__ == "__main__":
    print(get_chapter_ids())
    print(get_chapter_notes_sync(get_chapter_ids()[0], get_chapter_ids()[0]))