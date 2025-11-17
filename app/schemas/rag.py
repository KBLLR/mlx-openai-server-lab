"""RAG (Retrieval-Augmented Generation) schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Document chunk representation."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    score: Optional[float] = Field(None, description="Similarity score (for retrieval results)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


class RAGIngestRequest(BaseModel):
    """Request to ingest documents into RAG collection."""

    collection: str = Field(..., description="Collection name")
    chunk_size: int = Field(default=512, description="Chunk size in tokens")
    overlap: int = Field(default=64, description="Overlap between chunks")
    embedding_model: Optional[str] = Field(None, description="Model for embeddings")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Collection metadata")


class RAGIngestResponse(BaseModel):
    """Response for document ingestion."""

    collection: str = Field(..., description="Collection name")
    documents_ingested: int = Field(..., description="Number of documents ingested")
    chunks_created: int = Field(..., description="Number of chunks created")
    embedding_time_ms: int = Field(..., description="Embedding generation time")
    document_ids: List[str] = Field(..., description="Document identifiers")
    status: str = Field(..., description="Ingestion status")


class RAGQueryRequest(BaseModel):
    """Request for RAG query."""

    query: str = Field(..., description="User query")
    collection: str = Field(default="default", description="Collection name")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    model_id: Optional[str] = Field(None, description="Generation model ID")
    embedding_model: Optional[str] = Field(None, description="Embedding model ID")
    include_sources: bool = Field(default=True, description="Include source chunks in response")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score")
    rerank: bool = Field(default=False, description="Enable reranking")
    generation_config: Optional[Dict[str, Any]] = Field(None, description="Generation parameters")


class RAGResult(BaseModel):
    """Result of a RAG query."""

    answer: str = Field(..., description="Generated answer")
    sources: List[DocumentChunk] = Field(..., description="Source chunks")
    retrieval_latency_ms: int = Field(..., description="Retrieval latency")
    generation_latency_ms: int = Field(..., description="Generation latency")
    total_latency_ms: int = Field(..., description="Total latency")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Query metadata")


class RAGCollection(BaseModel):
    """RAG collection representation."""

    name: str = Field(..., description="Collection name")
    document_count: int = Field(..., description="Number of documents")
    chunk_count: int = Field(..., description="Number of chunks")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    chunk_size: int = Field(..., description="Chunk size in tokens")
    overlap: int = Field(..., description="Chunk overlap")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Collection metadata")


class RAGDocument(BaseModel):
    """Document in a RAG collection."""

    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    chunk_count: int = Field(..., description="Number of chunks")
    ingested_at: datetime = Field(..., description="Ingestion timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")


class RAGCollectionInfo(BaseModel):
    """Detailed collection information."""

    name: str = Field(..., description="Collection name")
    document_count: int = Field(..., description="Number of documents")
    chunk_count: int = Field(..., description="Number of chunks")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    chunk_size: int = Field(..., description="Chunk size in tokens")
    overlap: int = Field(..., description="Chunk overlap")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Collection metadata")
    documents: List[RAGDocument] = Field(..., description="Documents in collection")
