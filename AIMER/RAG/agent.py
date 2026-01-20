import subprocess
import sys

from langchain_ollama import ChatOllama

from langchain.agents import create_agent

from langchain.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from scripts.rag_tools import hybrid_search

from scripts.prompts import MULTIMODEL_AGENT_PROMPT

LLM_MODEL = ""

model = ChatOllama(
    model = LLM_MODEL,
    base_url="http://localhost:11434/",
)

