import uvicorn
from rag_service.config.settings import API_HOST, API_PORT
from rag_service.logs.logger import setup_logger

logger = setup_logger()

if __name__ == "__main__":
    logger.info(f"Запуск RAG Service API на {API_HOST}:{API_PORT}")
    uvicorn.run(
        "rag_service.app.main:app",
        host=API_HOST,
        port=int(API_PORT),
        reload=True,
        log_level="info",
    )
