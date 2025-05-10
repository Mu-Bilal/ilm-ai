from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from typing import List
from PyPDF2 import PdfReader
import random
import asyncio


class Answer(BaseModel):
    id: int
    name: str
    notes: str
    progress: int

async def get_topics(pdf_path: str):
    pdf_text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            pdf_text += page.extract_text()
    # print(pdf_text)
    ollama_model = OpenAIModel(
        model_name='qwen3:4b', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
    )
    agent = Agent(ollama_model, output_type=Answer, system_prompt="You are an expert topic classifier. Given a paragraph of text, your job is to extract the main topic in 1–2 words. Focus on the core subject of the text. Respond with only the topic, no explanations or extra words.", retries=3)

    # take the middle of the pdf text
    middle_index = len(pdf_text) // 2
    random_chunk = pdf_text[middle_index - 500:middle_index + 500]

    result = await agent.run(random_chunk)
    ret = result.output
    ret.progress = 0
    return [ret]

if __name__ == "__main__":
    asyncio.run(get_topics("/Users/ahmadmahmood/Documents/Col_hack/ilm-ai/server/file.pdf"))