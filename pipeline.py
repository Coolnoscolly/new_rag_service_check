from typing import List, Dict, Any, Tuple
import logging
from .base_pipeline import RagPipelineOllamaQdrant

logger = logging.getLogger(__name__)


class EnhancedRagPipeline:
    def __init__(self, qdrant_host: str, qdrant_port: int, collection_name: str):
        self._base_pipeline = RagPipelineOllamaQdrant(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection_name=collection_name,
        )
        self._collection_name = collection_name
        self._qdrant_host = qdrant_host
        self._qdrant_port = qdrant_port

        self._prompt = """
Ты - экспертная система для ответов на вопросы на основе предоставленной документации.

ВОПРОС: 
{query}

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context}

ИНСТРУКЦИИ:
1. Внимательно проанализируй контекст и найди ТОЧНО релевантную информацию для ответа на вопрос
2. Если в контексте есть прямой ответ - предоставь его полностью и точно
3. Если информация частичная - укажи, что именно известно, а что отсутствует
4. Если контекст не содержит ответа - честно скажи об этом
5. НЕ придумывай и не домысливай информацию
6. Структурируй ответ логично и последовательно
7. При наличии нескольких аспектов вопроса - освети каждый
8. Используй точные формулировки из документов

ОТВЕТ (на русском языке):
"""

    def vectorize_query(self, *, text: str) -> List[float]:
        """Векторизация запроса"""
        return self._base_pipeline.vectorize_query(text=text)

    def search_knn_by_query(self, *, query: str, top_k: int = 5) -> Any:
        """Базовый семантический поиск (для обратной совместимости)"""
        return self.search_knn_by_query_enhanced(
            query=query, top_k=top_k, use_hybrid=False
        )

    def search_knn_by_query_enhanced(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.4,
        use_hybrid: bool = True,
        fusion_alpha: float = 0.7,
        bm25_weight: float = 1.2,
        use_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Улучшенный поиск с поддержкой гибридного подхода с BM25

        Args:
            query: Поисковый запрос
            top_k: Количество возвращаемых результатов
            score_threshold: Порог релевантности
            use_hybrid: Использовать гибридный поиск
            fusion_alpha: Коэффициент слияния для гибридного поиска
            bm25_weight: Вес для BM25 скоринга
            use_rerank: Использовать переранжирование результатов
        """
        try:
            logger.info(f" Поиск по запросу: '{query}' (гибридный: {use_hybrid}, BM25)")

            if use_hybrid:
                search_result = self._base_pipeline.search_hybrid(
                    query=query,
                    top_k=min(top_k * 3, 30),  # Берем больше для лучшего слияния
                    score_threshold=score_threshold,
                    fusion_alpha=fusion_alpha,
                    bm25_weight=bm25_weight,
                    use_rerank=use_rerank,
                )
            else:
                search_result = self._base_pipeline.search_knn_by_query(
                    query=query,
                    top_k=min(top_k * 2, 20),
                    score_threshold=score_threshold,
                )

            results = [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                    "search_type": getattr(hit, "search_type", "semantic"),
                }
                for hit in search_result
                if hit.score >= score_threshold
            ]

            # Сортируем по релевантности и берем top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]

            # Логируем статистику по типам поиска
            search_types = [r.get("search_type", "semantic") for r in results]
            type_counts = {}
            for st in search_types:
                type_counts[st] = type_counts.get(st, 0) + 1

            logger.info(
                f" Найдено {len(results)} документов. Типы поиска: {type_counts}"
            )

            if results:
                scores_info = [f"{r['score']:.3f}" for r in results]
                logger.info(f" Скоринги: {scores_info}")

            return results

        except Exception as error:
            logger.error(f" Ошибка при поиске в Qdrant: {error}")
            raise Exception(f"Error in searching in Qdrant's DB: {error}")

    def search_bm25_only(
        self, query: str, top_k: int = 5, bm25_weight: float = 1.2
    ) -> List[Dict[str, Any]]:
        """
        Только BM25 поиск (для тестирования и сравнения)

        Args:
            query: Поисковый запрос
            top_k: Количество возвращаемых результатов
            bm25_weight: Вес для BM25 скоринга
        """
        try:
            logger.info(f" BM25 поиск по запросу: '{query}'")

            bm25_results = self._base_pipeline.search_bm25(
                query=query, top_k=top_k, bm25_weight=bm25_weight
            )

            results = [
                {
                    "id": result["id"],
                    "score": result["score"],
                    "payload": result["payload"],
                    "search_type": "bm25",
                    "original_bm25_score": result.get("original_score", 0),
                }
                for result in bm25_results
            ]

            logger.info(f" BM25 найдено {len(results)} документов")

            if results:
                bm25_scores = [f"{r['original_bm25_score']:.3f}" for r in results]
                logger.info(f"BM25 скоринги: {bm25_scores}")

            return results

        except Exception as error:
            logger.error(f" Ошибка при BM25 поиске: {error}")
            raise Exception(f"Error in BM25 search: {error}")

    def retrieve_context(self) -> str:
        """Заглушка для обратной совместимости"""
        raise NotImplementedError("Use retrieve_context_enhanced instead")

    def retrieve_context_enhanced(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.4,
        temperature: float = 0.1,
        use_hybrid: bool = True,
        fusion_alpha: float = 0.7,
        bm25_weight: float = 1.2,
        use_rerank: bool = True,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Улучшенное извлечение контекста с гибридным поиском и BM25

        Args:
            query: Поисковый запрос
            top_k: Количество возвращаемых результатов
            score_threshold: Порог релевантности
            temperature: Температура для генерации
            use_hybrid: Использовать гибридный поиск
            fusion_alpha: Коэффициент слияния для гибридного поиска
            bm25_weight: Вес для BM25 скоринга
            use_rerank: Использовать переранжирование результатов
        """
        try:
            self._base_pipeline.temperature = temperature

            logger.info(
                f" Начало обработки запроса: '{query}' "
                f"(гибридный+BM25: {use_hybrid}, alpha: {fusion_alpha})"
            )

            search_results = self.search_knn_by_query_enhanced(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                use_hybrid=use_hybrid,
                fusion_alpha=fusion_alpha,
                bm25_weight=bm25_weight,
                use_rerank=use_rerank,
            )

            if not search_results:
                logger.warning(" По запросу не найдено релевантных документов")
                return (
                    "По вашему запросу не найдено релевантной информации в базе знаний.",
                    [],
                )

            # Анализируем релевантность результатов
            highly_relevant = [r for r in search_results if r["score"] > 0.7]
            moderately_relevant = [
                r for r in search_results if 0.4 <= r["score"] <= 0.7
            ]
            low_relevant = [r for r in search_results if r["score"] < 0.4]

            logger.info(
                f" Релевантность: высоких={len(highly_relevant)}, "
                f"умеренных={len(moderately_relevant)}, низких={len(low_relevant)}"
            )

            # Форматируем контекст с учетом релевантности
            context = self._format_context_by_relevance(
                highly_relevant, moderately_relevant
            )

            if not context:
                logger.warning("Контекст пуст после фильтрации")
                return (
                    "Найденная информация недостаточно релевантна для точного ответа на ваш вопрос.",
                    [],
                )

            # Анализируем эффективность поиска
            self._log_search_effectiveness(search_results, query)

            formatted_prompt = self._prompt.format(query=query, context=context)

            logger.info(" Генерация ответа с помощью LLM...")
            response = self._base_pipeline.invoke(formatted_prompt)

            logger.info(
                f" Ответ успешно сгенерирован, длина: {len(response.content)} символов"
            )
            return response.content, search_results

        except Exception as error:
            logger.error(f" Ошибка при извлечении контекста: {error}")
            raise Exception(f"Error in retrieving context: {error}")

    def _log_search_effectiveness(
        self, search_results: List[Dict[str, Any]], query: str
    ) -> None:
        """Логирование эффективности поиска"""
        if not search_results:
            return

        # Анализ типов поиска
        search_types = [r.get("search_type", "semantic") for r in search_results]
        type_stats = {}
        for st in search_types:
            type_stats[st] = type_stats.get(st, 0) + 1

        # Средний скор
        avg_score = sum(r["score"] for r in search_results) / len(search_results)
        max_score = max(r["score"] for r in search_results)

        logger.info(f" Эффективность поиска для '{query}':")
        logger.info(f"   Типы поиска: {type_stats}")
        logger.info(f"   Средний скор: {avg_score:.3f}")
        logger.info(f"   Максимальный скор: {max_score:.3f}")
        logger.info(f"   Всего результатов: {len(search_results)}")

    def _format_context_by_relevance(
        self, highly_relevant: List[Dict], moderately_relevant: List[Dict]
    ) -> str:
        """Форматирование контекста с указанием типа поиска и релевантности"""
        context_parts = []

        if highly_relevant:
            context_parts.append("=== ВЫСОКОРЕЛЕВАНТНЫЕ ДОКУМЕНТЫ ===")
            for i, result in enumerate(highly_relevant, 1):
                filename = result["payload"].get("file_name", "Unknown")
                chunk = result["payload"].get("chunk_text", "")
                score = result["score"]
                search_type = result.get("search_type", "semantic")

                # Обрезаем слишком длинные чанки
                if len(chunk) > 1000:
                    chunk = chunk[:1000] + "... [обрезано]"

                search_type_emoji = {
                    "semantic": "🔍",
                    "bm25": "📝",
                    "hybrid": "🔄",
                }.get(search_type, "📄")

                context_parts.append(
                    f"\n{search_type_emoji} Документ {i}: {filename} "
                    f"(релевантность: {score:.3f}, тип: {search_type})\n"
                    f"Содержание:\n{chunk}\n"
                    f"{'-'*50}"
                )

        if moderately_relevant:
            context_parts.append("\n=== ДОПОЛНИТЕЛЬНЫЕ ДОКУМЕНТЫ ===")
            for i, result in enumerate(moderately_relevant, 1):
                filename = result["payload"].get("file_name", "Unknown")
                chunk = result["payload"].get("chunk_text", "")
                score = result["score"]
                search_type = result.get("search_type", "semantic")

                # Обрезаем слишком длинные чанки
                if len(chunk) > 800:
                    chunk = chunk[:800] + "... [обрезано]"

                search_type_emoji = {
                    "semantic": "🔍",
                    "bm25": "📝",
                    "hybrid": "🔄",
                }.get(search_type, "📄")

                context_parts.append(
                    f"\n{search_type_emoji} Документ {i}: {filename} "
                    f"(релевантность: {score:.3f}, тип: {search_type})\n"
                    f"Содержание:\n{chunk}\n"
                    f"{'-'*50}"
                )

        return "\n".join(context_parts) if context_parts else ""

    def compare_search_methods(
        self, query: str, top_k: int = 5, score_threshold: float = 0.4
    ) -> Dict[str, Any]:
        """
        Сравнение разных методов поиска для анализа эффективности

        Returns:
            Словарь с результатами всех методов поиска
        """
        try:
            logger.info(f" Сравнение методов поиска для: '{query}'")

            # Семантический поиск
            semantic_results = self.search_knn_by_query_enhanced(
                query=query, top_k=top_k, use_hybrid=False
            )

            # BM25 поиск
            bm25_results = self.search_bm25_only(query=query, top_k=top_k)

            # Гибридный поиск
            hybrid_results = self.search_knn_by_query_enhanced(
                query=query, top_k=top_k, use_hybrid=True
            )

            # Анализ пересечений
            semantic_ids = {r["id"] for r in semantic_results}
            bm25_ids = {r["id"] for r in bm25_results}
            hybrid_ids = {r["id"] for r in hybrid_results}

            semantic_bm25_intersection = semantic_ids & bm25_ids
            all_intersection = semantic_ids & bm25_ids & hybrid_ids

            comparison_result = {
                "query": query,
                "semantic": {
                    "count": len(semantic_results),
                    "avg_score": self._calculate_avg_score(semantic_results),
                    "max_score": self._calculate_max_score(semantic_results),
                    "results": semantic_results,
                },
                "bm25": {
                    "count": len(bm25_results),
                    "avg_score": self._calculate_avg_score(bm25_results),
                    "max_score": self._calculate_max_score(bm25_results),
                    "results": bm25_results,
                },
                "hybrid": {
                    "count": len(hybrid_results),
                    "avg_score": self._calculate_avg_score(hybrid_results),
                    "max_score": self._calculate_max_score(hybrid_results),
                    "results": hybrid_results,
                },
                "analysis": {
                    "semantic_bm25_overlap": len(semantic_bm25_intersection),
                    "all_methods_overlap": len(all_intersection),
                    "unique_semantic": len(semantic_ids - bm25_ids),
                    "unique_bm25": len(bm25_ids - semantic_ids),
                },
            }

            logger.info(f" Сравнение методов завершено:")
            logger.info(f"   Семантических: {comparison_result['semantic']['count']}")
            logger.info(f"   BM25: {comparison_result['bm25']['count']}")
            logger.info(f"   Гибридных: {comparison_result['hybrid']['count']}")
            logger.info(
                f"   Пересечение семантический+BM25: {comparison_result['analysis']['semantic_bm25_overlap']}"
            )

            return comparison_result

        except Exception as error:
            logger.error(f" Ошибка при сравнении методов поиска: {error}")
            return {"error": str(error)}

    def _calculate_avg_score(self, results: List[Dict[str, Any]]) -> float:
        """Вычисление среднего скора"""
        if not results:
            return 0.0
        return sum(r["score"] for r in results) / len(results)

    def _calculate_max_score(self, results: List[Dict[str, Any]]) -> float:
        """Вычисление максимального скора"""
        if not results:
            return 0.0
        return max(r["score"] for r in results)

    def clear_bm25_cache(self) -> Dict[str, Any]:
        """
        Очистка кэша BM25 (полезно при обновлении коллекции)

        Returns:
            Статистика очистки кэша
        """
        try:
            cache_keys = list(self._base_pipeline._bm25_cache.keys())
            documents_cache_keys = list(self._base_pipeline._documents_cache.keys())

            self._base_pipeline._bm25_cache.clear()
            self._base_pipeline._documents_cache.clear()

            result = {
                "cleared_bm25_cache": len(cache_keys),
                "cleared_documents_cache": len(documents_cache_keys),
                "remaining_memory": "Кэш BM25 очищен",
            }

            logger.info(
                f" Очищен кэш BM25: {len(cache_keys)} индексов, "
                f"{len(documents_cache_keys)} документов"
            )

            return result

        except Exception as error:
            logger.error(f" Ошибка при очистке кэша BM25: {error}")
            return {"error": str(error)}

    def get_search_stats(self) -> Dict[str, Any]:
        """
        Получение статистики поиска

        Returns:
            Статистика по кэшу и методам поиска
        """
        try:
            bm25_cache_size = len(self._base_pipeline._bm25_cache)
            documents_cache_size = len(self._base_pipeline._documents_cache)

            stats = {
                "bm25_cache_size": bm25_cache_size,
                "documents_cache_size": documents_cache_size,
                "active_collections": list(self._base_pipeline._bm25_cache.keys()),
                "status": "active",
            }

            logger.info(
                f" Статистика поиска: BM25 кэш={bm25_cache_size}, "
                f"документы={documents_cache_size}"
            )

            return stats

        except Exception as error:
            logger.error(f"❌ Ошибка при получении статистики: {error}")
            return {"error": str(error)}
