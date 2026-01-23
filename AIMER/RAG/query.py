from dotenv import load_dotenv

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from qdrant_client.models import FieldCondition, Filter, MatchValue

from scripts.schema import ChunkMetadata

load_dotenv()

COLLECTION_NAME = "rag_docs"

LLM_MODEL = "qwen3:8b"
EMBEDDING_MODEL = "qwen3-embedding:8b"
SPARCE_EMBEDDING_MODEL = "Qdrant/bm25"
RERANKER_MODEL = "Krakekai/qwen3-reranker-8b"

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
spare_embeddings = FastEmbedSparse(
    model=SPARCE_EMBEDDING_MODEL,
)

# Connection to existing collection
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    sparse_embedding=spare_embeddings,
    collection_name=COLLECTION_NAME,
    url="http://localhost:6333",
    retrieval_mode=RetrievalMode.HYBRID,
)


def extract_filters(user_query: str):
    prompt = """
            Extract metadata filters from the query. Return None for fields mentioned.

            <USER QUERY STARTS>
            {user_query}
            </USER QUERY ENDS>
            
            Extract metadata based on the user query only:           
        """

    structured_llm = llm.with_structured_output(ChunkMetadata)

    metadata = structured_llm.invoke(prompt)

    if metadata:
        filters = metadata.model_dump(exclude_nome=True)
    else:
        filters = {}

    return filters

    return filters


def hybrid_search(query: str, k: int = 10, filters: dict = None):
    qdrant_filter = None

    if filters:
        condition = [
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            for key, value in filters.items()
        ]

        qdrant_filter = Filter(must=condition)

    results = vector_store.similarity_search(
        query=query, k=k, filter=qdrant_filter
    )

    return results


def rerank_results(query: str, documents=list, top_k: int = 5):
    reranker = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL, model_kwargs={"device": "xpu"}
    )

    query_doc_pairs = [(query, doc.page_content) for doc in documents]

    scores = reranker.score(query_doc_pairs)

    reranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)[
        :top_k
    ]

    return [rank[1] for rank in reranked]


query = ""
filters = extract_filters(query)
print(filters)
results = hybrid_search(query, k=10, filters=filters)

response = rerank_results(query, results)
