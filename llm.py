"""
llm.py

This module defines the shared language model (LLM) instance used across all agents.
This allows for consistent configuration and easy reuse.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.2,  # better for factual tasks
)
