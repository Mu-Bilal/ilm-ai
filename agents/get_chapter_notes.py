from dataclasses import dataclass

from pydantic_ai import RunContext

@dataclass
class Deps:
    course_id: str
    chapter_id: str

async def get_chapter_notes(ctx: RunContext[Deps]) -> str:
    """
    Call this function to get the notes for a chapter.
    """
    course_id = ctx.deps.course_id
    chapter_id = ctx.deps.chapter_id
    return {
        "chapter_1": {
            "notes": """Deep learning is a subset of machine learning that uses artificial neural networks 
            with multiple layers to progressively extract higher-level features from raw input. These neural 
            networks are inspired by the biological neural networks in human brains. Deep learning has achieved 
            remarkable success in various tasks like computer vision, natural language processing, and speech recognition.

            Normalization layers play a crucial role in deep neural networks by standardizing the inputs to each layer. 
            The most common types are:
            - Batch Normalization: Normalizes the output of a previous activation layer by subtracting the batch mean 
              and dividing by the batch standard deviation
            - Layer Normalization: Similar to batch norm but normalizes across the features instead of the batch
            - Instance Normalization: Normalizes across the spatial dimensions only
            
            These layers help combat internal covariate shift and allow for faster training of deep networks.""",
        },
        "chapter_2": {
            "notes": """Data classes are a feature in Python that automatically adds generated special methods 
            such as __init__() and __repr__() to user-defined classes. They help reduce boilerplate code when 
            creating classes that primarily store data. Key features include:
            - Automatic generation of __init__, __repr__, __eq__ methods
            - Support for default values
            - Type annotations
            - Immutable instances using @dataclass(frozen=True)

            Pydantic models are a specific implementation of data validation using Python type annotations. They:
            - Enforce type hints at runtime
            - Provide automatic data validation
            - Support complex data structures
            - Handle data parsing and serialization
            - Integrate well with FastAPI and other modern Python frameworks""",
        },
    }[chapter_id]