import os
import dotenv
from pymilvus import MilvusClient

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


def _clean_text(text: str) -> str:
    return text.replace("search_document: ", "").strip()


async def get_data_from_topic(topic: str, mastery_notes: str) -> str:
    """
    Call this function to get the notes for a topic.
    """
    query_vectors = model.encode(f"search_query: {topic} taking into account the following notes: {mastery_notes}")

    debug(query_vectors)

    # Get the cha data from the Milvus collection
    res = client.search(
        collection_name="PAI",
        data=[query_vectors.tolist()],
        limit=3,
        output_fields=["text"]
    )

    output = [{'distance': hit['distance'], 'text': _clean_text(hit['entity']['text'])} for hit in res[0]]

    debug(output)

    return output


if __name__ == "__main__":
    import asyncio

    async def main():
        res = await get_data_from_topic("normalization", "")
        print(res)
    
    asyncio.run(main())
