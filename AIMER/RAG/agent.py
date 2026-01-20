
from langchain_ollama import ChatOllama






LLM_MODEL = ""

model = ChatOllama(
    model = LLM_MODEL,
    base_url="http://localhost:11434/",
)

