from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

# re-ranking for better result
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# metadata filtering
from qdrant_client.models import Filter, FieldCondition, MatchValue

# metadata extraction from LLM
from schema import ChunkMetadata

from langchain_core.tools import tool

# Configuration
COLLECTION_NAME = ""
EMBEDDING_MODEL = ""
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
LLM_MODEL = ""

RERANKER_MODEL = ""

# Initialize LLM
llm = ChatOllama(
    model=LLM_MODEL,
    base_url="http://localhost:11434",
)

# Embeddings
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url="http://localhost:11434",
)

# Sparse embeddings
sparse_embeddings = FastEmbedSparse(
    model=SPARSE_EMBEDDING_MODEL,
)

# Connect to existing collection
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    sparse_embeddings=sparse_embeddings,
    collection_name=COLLECTION_NAME,
    url="http://localhost:6333",
    retrieval_mode=RetrievalMode.HYBRID,
)

# Filter Extraction with LLM


def extract_filters(user_query: str):
    prompt = f"""
            Extract metada filers from the query. Return None for fields not mentionned.
            
                <USER QUERY STARTS>
                {user_query}
                <USER QUERY ENDS>
                
                Extract metadata base on the user query only:        
            """

    structured_llm = llm.with_structured_output(ChunkMetadata)

    metadata = structured_llm.invoke(prompt)

    if metadata:
        filters = metadata.model_dump(exclude_none=True)
    else:
        filters = {}

    return filters


@tool
def hybrid_search(query: str, k: int = 5):
    """
    Perform hybrid search (dense + sparse vector).

    Args:
        query: Search query
        k: Number of results
        filters: Optional filters like {}

    Returns:
        List of Document objects
    """

    filters = extract_filters(query)

    qdrant_filter = None

    if filters:
        condition = [
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            for key, value in filters.items()
        ]

        qdrant_filter = Filter(must=condition)

    results = vector_store.similarity_search(
        query=query, k=k, filters=qdrant_filter
    )

    return results
