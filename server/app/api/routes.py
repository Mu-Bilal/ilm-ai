from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.course_generator import CourseGenerator
from app.services.personalized_notes import getNotes
from agents.evaluate_answer import evaluate_answer

router = APIRouter()
course_generator = CourseGenerator()

class CourseGenerationRequest(BaseModel):
    title: str
    description: str
    urls: List[str]
    topics: Optional[List[str]] = None

class Request(BaseModel):
    user_id: str
    course_id: str
    chapter_id: str

class CheckAnswerRequest(BaseModel):
    userAnswer: str
    question: str
    explanation: str

@router.post("/generate-course")
async def generate_course(request: CourseGenerationRequest):
    try:
        course = await course_generator.generate_course(
            title=request.title,
            description=request.description,
            urls=request.urls,
            topics=request.topics
        )
        return course
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/personalize-notes")
async def personalize_notes(request: Request):
    try:
        
        notes = await getNotes(
            user_id=request.user_id,
            course_id=request.course_id,
            chapter_id=request.chapter_id
        )
        return {"notes": notes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-answer")
async def check_answer(request: CheckAnswerRequest):
    try:
        grade = await evaluate_answer(request.userAnswer, request.question, request.explanation)
        assessment = grade.assessment
        score = grade.score
        justification = grade.justification
        feedback_for_student = grade.feedback_for_student
        return {
            "assessment": assessment,
            "score": score,
            "justification": justification,
            "feedback_for_student": feedback_for_student
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/health")
async def health_check():
    return {"status": "healthy"} 