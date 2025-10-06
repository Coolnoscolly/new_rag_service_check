from abc import ABC, abstractmethod
from typing import List, Any

class RagInterface(ABC):
    
    @abstractmethod
    def vectorize_query(self, *, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def search_knn_by_query(self, *, query: str, top_k: int = 5) -> Any:
        pass
    
    @abstractmethod
    def retrieve_context(self) -> str:
        pass