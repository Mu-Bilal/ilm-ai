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

async def get_data_from_topic(ctx: RunContext[Deps]) -> str:
    """
    Call this function to get the notes for a chapter.
    """
    topic = ctx.deps.topic
    mastery_notes = ctx.deps.mastery_notes

    query_vectors = model.encode(f"search_query: {topic} taking into account the following notes: {mastery_notes}")

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
        res = await get_data_from_topic(dummy_ctx)
        print(res)
    
    asyncio.run(main())
