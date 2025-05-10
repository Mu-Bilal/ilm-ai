from agents.personalized_note_generation_agent import getNotes as agent_getNotes

async def getNotes(user_id: str, course_id: str, chapter_id: str) -> str:
    """
    Get personalized notes for a specific chapter based on user's mastery level.
    
    Args:
        user_id: The ID of the user
        course_id: The ID of the course
        chapter_id: The ID of the chapter
        
    Returns:
        str: Personalized notes for the chapter
    """
    try:
        # Get personalized notes from the agent
        notes = await agent_getNotes(user_id, course_id, chapter_id)
        return notes
    except Exception as e:
        print(f"Error generating personalized notes: {str(e)}")
        raise 