import os
import asyncio
from typing import Optional, Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from agents.tools.rag import search_course_content

# import logfire
# logfire.configure(token='pylf_v1_eu_W7WPNNgs18K4HcXghtmnzsZ5HdQY1f2ZZkT7pBflsZdt')
# --- 1. Define Pydantic Models ---

class GradingInput(BaseModel):
    question: str = Field(..., description="The question asked to the student.")
    correct_answer: str = Field(..., description="The ideal or model answer to the question.")
    student_answer: str = Field(..., description="The student's submitted answer.")

class GradeOutput(BaseModel):
    assessment: Literal[
        "Correct",
        "Partially Correct",
        "Incorrect",
        "Conceptually Correct but Minor Error"
    ] = Field(..., description="The assessment of the student's answer.")
    score: float = Field(..., ge=0.0, le=1.0, description="A numerical score between 0.0 and 1.0 representing correctness.")
    justification: str = Field(..., description="Explanation for the assessment and score, citing specifics.")
    feedback_for_student: Optional[str] = Field(None, description="Constructive feedback for the student.")

# --- 2. Initialize LLM and Agent ---

# Define the model name as a constant
OLLAMA_MODEL_NAME = 'qwen3:4b'
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Initialize the Ollama LLM client
try:
    # Create provider
    provider = OpenAIProvider(base_url=OLLAMA_BASE_URL)
    
    # Create the LLM Model instance (e.g., OpenAIModel for Ollama's OpenAI-compatible API)
    ollama_llm_model = OpenAIModel(
        model_name=OLLAMA_MODEL_NAME,
        provider=provider
    )
    
    # Define the single grading agent
    # Pass the ollama_llm_model instance as the first argument
    grading_agent = Agent[GradeOutput]( 
        ollama_llm_model,               
        output_type=GradeOutput,        
        deps_type=GradingInput,
        tools=[search_course_content]        
    )

    # grading_agent.instrument_all()
    
except Exception as e:
    print(f"Error initializing Ollama LLM or Agent: {e}")
    print(f"Please ensure Ollama is running and the model '{OLLAMA_MODEL_NAME}' is available.")
    print("You can pull models using 'ollama pull <model_name>' (e.g., 'ollama pull qwen3:4b').")
    exit()

# --- 3. Define the Prompt Template ---
GRADING_PROMPT_TEMPLATE = """
You are an expert AI teaching assistant. Your task is to grade a student's answer based on the provided materials.
Output ONLY the JSON object matching the required schema, with no other text before or after it.

Question:
{question}

Correct Answer:
{correct_answer}

Student's Answer:
{student_answer}

To find the correct answer you can search the course notes/lecture notes to get the relevant information to answer the question.

Instructions for grading:
1.  Carefully compare the Student's Answer to the Correct Answer.
2.  Consult the Lecture Notes to understand the key concepts, expected depth, and terminology.
3.  Determine an assessment category: "Correct", "Partially Correct", "Incorrect", or "Conceptually Correct but Minor Error".
4.  Assign a numerical score between 0.0 (completely incorrect) and 1.0 (perfectly correct).
    -   "Correct": 1.0
    -   "Conceptually Correct but Minor Error": Typically 0.8-0.95
    -   "Partially Correct": Typically 0.3-0.7
    -   "Incorrect": 0.0-0.2
5.  Provide a clear justification for your assessment and score. Be specific:
    -   Point out what the student got right.
    -   Point out what the student got wrong or missed.
    -   Reference lecture notes or correct answer if it helps.
6.  (Optional) Provide brief, constructive feedback for the student.

Now, provide your assessment as a JSON object conforming to the structure:
{{
  "assessment": "...",
  "score": ...,
  "justification": "...",
  "feedback_for_student": "..."
}}
""" 

# --- 4. Define the Grading Workflow Function ---

async def get_answer_grade(grading_input: GradingInput) -> Optional[GradeOutput]:
    print(f"--- Grading Student Answer for Question: '{grading_input.question[:50]}...' ---")

    prompt = GRADING_PROMPT_TEMPLATE.format(
        question=grading_input.question,
        correct_answer=grading_input.correct_answer,
        student_answer=grading_input.student_answer,
    )

    try:
        print("Sending prompt to Ollama...")

        grade_result_container = await grading_agent.run( # Renamed to avoid confusion
            prompt,
        )
        
        # Access the parsed Pydantic model from the .output attribute
        parsed_grade_output: GradeOutput = grade_result_container.output

        # Now access attributes from the parsed_grade_output (which is of type GradeOutput)
        print(f"Assessment: {parsed_grade_output.assessment}")
        print(f"Score: {parsed_grade_output.score:.2f}")
        print(f"Justification: {parsed_grade_output.justification}")
        if parsed_grade_output.feedback_for_student:
            print(f"Feedback for Student: {parsed_grade_output.feedback_for_student}")

        print("--- Grading Complete ---\n")
        return parsed_grade_output # Return the actual GradeOutput instance

    except Exception as e:
        print(f"Error during grading with Ollama: {e}")
        print(f"Error type: {type(e)}")

        if hasattr(e, 'raw_llm_response'): # pydantic-ai often stores it here
            print(f"Raw LLM Response that failed parsing: {e.raw_llm_response}")
        elif hasattr(e, 'response'): # Fallback for other potential error structures
            print(f"Raw LLM Response that failed parsing: {e.response}")

        print("This might be due to the model not producing valid JSON, Ollama not running, or the model not being available.")
        print("Ensure the LLM is outputting ONLY the JSON structure as requested in the prompt.")
        print("Consider checking the Ollama server logs for more details.")
        return None

# --- 5. Example Usage ---

async def evaluate_answer(question: str, correct_answer: str, student_answer: str):

    input = GradingInput(
        question=question,
        correct_answer=correct_answer,
        student_answer=student_answer,
    )
    grade = await get_answer_grade(input)
    return grade


if __name__ == "__main__":
    asyncio.run(evaluate_answer("Does a gaussian random variable added to a gaussian random variable yield a gaussian random variable?", "Yes", "No"))