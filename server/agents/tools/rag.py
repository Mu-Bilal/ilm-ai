import os
import dotenv
from pymilvus import MilvusClient

from pydantic_ai import RunContext
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

from devtools import debug


dotenv.load_dotenv()


CLUSTER_ENDPOINT = os.getenv("MILVUS_CLUSTER_ENDPOINT")
TOKEN = os.getenv("MILVUS_CLUSTER_TOKEN")


client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN 
)


@dataclass
class Deps:
    topic: str
    mastery_notes: str

async def search_course_content(ctx: RunContext[Deps], query: str) -> str:
    """
    Call this function to get course content related to the query
    """
    query_vectors = model.encode(f"search_query: {query}")

    debug(query_vectors)

    # Get the cha data from the Milvus collection
    res = client.search(
        collection_name="PAI",
        data=[query_vectors.tolist()],
        limit=3,
        output_fields=["text"]
    )

    debug(res)

    output = [{'distance': hit['distance'], 'text': hit['entity']['text']} for hit in res[0]]

    return output


if __name__ == "__main__":
    import asyncio
    from typing import Generic, TypeVar
    
    # Create a simplified version of RunContext for testing
    T = TypeVar('T')
    class DummyRunContext(Generic[T]):
        def __init__(self, deps: T):
            self.deps = deps
    
    async def main():
        # Wrap Deps in the DummyRunContext
        dummy_ctx = DummyRunContext(Deps(topic="normalization", mastery_notes=""))
        res = await search_course_content(dummy_ctx)
        print(res)
    
    asyncio.run(main())
