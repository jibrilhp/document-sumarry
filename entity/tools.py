from typing import Optional, Set
from pydantic import BaseModel, Field

class VectorSearchInput(BaseModel):
    """Input for vector database search"""
    query: str = Field(description="Search query to find similar documents")
    k: int = Field(default=5, description="Number of results to return")
    score_threshold: Optional[float] = Field(default=None, description="Minimum similarity score")

class VectorSearchOutput(BaseModel):
    """Output for vector database search"""
    query: str = Field(description="Search query that was used")
    documents: list[str] = Field(description="List of document that match the search query")
    scores: Optional[list[float]] = Field(default=None, description="List of similarity scores for each document")
    total_results: int = Field(description="Total number of results found in the vector database")
    sources: Optional[Set[str]] = Field(default=None, description="Set of unique source identifiers for the documents")