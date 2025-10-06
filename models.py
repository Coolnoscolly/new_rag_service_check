from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class QueryRequest(BaseModel):
    query: str = Field(..., description="Вопрос для поиска ответа")
    collection_name: str = Field(..., description="Название коллекции в Qdrant")
    top_k: int = Field(
        default=5, description="Количество документов для поиска", ge=1, le=20
    )
    score_threshold: float = Field(
        default=0.5, description="Минимальный порог релевантности", ge=0.0, le=1.0
    )
    temperature: float = Field(
        default=0.1, description="Температура для генерации ответа", ge=0.0, le=1.0
    )
    use_hybrid: bool = Field(
        default=True, description="Использовать гибридный поиск"
    )
    fusion_alpha: float = Field(
        default=0.7, description="Коэффициент слияния для гибридного поиска", ge=0.0, le=1.0
    )
    bm25_weight: float = Field(
        default=1.2, description="Вес для BM25 скоринга", ge=0.1, le=3.0
    )
    use_rerank: bool = Field(
        default=True, description="Использовать переранжирование результатов"
    )

class SourceDocument(BaseModel):
    file_name: str
    chunk_text: str
    relevance_score: float
    chunk_id: Optional[str] = None
    search_type: Optional[str] = Field(default="semantic", description="Тип поиска: semantic, bm25, hybrid")

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query: str
    collection_name: str
    processing_time: float
    status: str
    search_type: str = Field(default="semantic", description="Тип использованного поиска")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, bool]

class BencmarkResponse(BaseModel):
    scores: list = Field(...)