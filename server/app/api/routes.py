from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.course_generator import CourseGenerator

router = APIRouter()
course_generator = CourseGenerator()

class CourseGenerationRequest(BaseModel):
    title: str
    description: str
    urls: List[str]
    topics: Optional[List[str]] = None

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

@router.get("/health")
async def health_check():
    return {"status": "healthy"} 