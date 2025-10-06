from typing import List, Any, Tuple, Dict
import requests
from qdrant_client import QdrantClient
import numpy as np
from sklearn.preprocessing import normalize
import logging
import re
from collections import Counter
from rank_bm25 import BM25Okapi
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os


logger = logging.getLogger(__name__)

# Указываем путь для данных NLTK
nltk_data_path = "/app/nltk_data"  # или другой путь в вашем контейнере
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Скачиваем необходимые данные если их нет
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    print(" Скачивание stopwords для NLTK...")
    nltk.download("stopwords", download_dir=nltk_data_path)
    nltk.download("punkt", download_dir=nltk_data_path)
    print("NLTK данные успешно скачаны")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)


class RagPipelineOllamaQdrant:
    def __init__(
        self,
        qdrant_host: str,
        qdrant_port: int,
        collection_name: str,
        qdrant_api_key: str = None,
        embed_url: str = "http://10.100.102.70:11434",
        generate_url: str = "http://10.100.102.70:11434",
        embed_model: str = "bge-m3:latest",
        generate_model: str = "qwen2.5:7b",
    ):
        self._qdrant_client = QdrantClient(
            url=f"{qdrant_host}:{qdrant_port}", api_key=qdrant_api_key, timeout=6000
        )
        self._collection_name = collection_name

        self._embed_url = embed_url
        self._generate_url = generate_url
        self._embed_model = embed_model
        self._generate_model = generate_model
        self._ollama_generator = self
        self._ollama_embedder = self

        # Кэш для BM25
        self._bm25_cache = {}
        self._documents_cache = {}

        self._check_connections()

    def _russian_tokenize(self, text: str) -> List[str]:
        """Токенизация для русского языка с расширенным списком стоп-слов"""
        # Расширенный список русских стоп-слов
        russian_stopwords = {
            "и",
            "в",
            "во",
            "не",
            "что",
            "он",
            "на",
            "я",
            "с",
            "со",
            "как",
            "а",
            "то",
            "все",
            "она",
            "так",
            "его",
            "но",
            "да",
            "ты",
            "к",
            "у",
            "же",
            "вы",
            "за",
            "бы",
            "по",
            "только",
            "ее",
            "мне",
            "было",
            "вот",
            "от",
            "меня",
            "еще",
            "нет",
            "о",
            "из",
            "ему",
            "теперь",
            "когда",
            "даже",
            "ну",
            "вдруг",
            "ли",
            "если",
            "уже",
            "или",
            "ни",
            "быть",
            "был",
            "него",
            "до",
            "вас",
            "нибудь",
            "опять",
            "уж",
            "вам",
            "ведь",
            "там",
            "потом",
            "себя",
            "ничего",
            "ей",
            "может",
            "они",
            "тут",
            "где",
            "есть",
            "надо",
            "ней",
            "для",
            "мы",
            "тебя",
            "их",
            "чем",
            "была",
            "сам",
            "чтоб",
            "без",
            "будто",
            "чего",
            "раз",
            "тоже",
            "себе",
            "под",
            "будет",
            "ж",
            "тогда",
            "кто",
            "этот",
            "того",
            "потому",
            "этого",
            "какой",
            "совсем",
            "ним",
            "здесь",
            "этом",
            "один",
            "почти",
            "мой",
            "тем",
            "чтобы",
            "нее",
            "сейчас",
            "были",
            "куда",
            "зачем",
            "всех",
            "никогда",
            "можно",
            "при",
            "наконец",
            "два",
            "об",
            "другой",
            "хоть",
            "после",
            "над",
            "больше",
            "тот",
            "через",
            "эти",
            "нас",
            "про",
            "всего",
            "них",
            "какая",
            "много",
            "разве",
            "три",
            "эту",
            "моя",
            "впрочем",
            "хорошо",
            "свою",
            "этой",
            "перед",
            "иногда",
            "лучше",
            "чуть",
            "том",
            "нельзя",
            "такой",
            "им",
            "более",
            "всегда",
            "конечно",
            "всю",
            "между",
            "это",
            "как",
            "так",
            "и",
            "в",
            "над",
            "к",
            "до",
            "не",
            "на",
            "но",
            "за",
            "то",
            "с",
            "ли",
            "а",
            "во",
            "от",
            "со",
            "для",
            "о",
            "же",
            "ну",
            "вы",
            "бы",
            "что",
            "кто",
            "он",
            "она",
            "что",
            "где",
            "когда",
            "какой",
            "какая",
            "какое",
            "какие",
            "почему",
            "зачем",
            "сколько",
            "как",
            "на",
            "по",
            "из",
            "от",
            "до",
            "у",
            "без",
            "под",
            "над",
            "при",
            "после",
            "в",
            "течение",
            "или",
            "либо",
            "ни",
            "нет",
            "да",
            "но",
            "однако",
            "хотя",
            "пусть",
            "если",
            "то",
            "так",
            "как",
            "будто",
            "точно",
            "словно",
            "чем",
            "нежели",
            "чтобы",
            "кабы",
            "дабы",
            "пока",
            "едва",
            "лишь",
            "только",
            "ли",
            "же",
            "ведь",
            "вот",
            "мол",
            "дескать",
            "типа",
            "например",
            "так",
            "итак",
            "следовательно",
            "поэтому",
            "затем",
            "потом",
            "вдобавок",
            "кроме",
            "сверх",
            "вместо",
            "около",
            "возле",
            "вокруг",
            "перед",
            "за",
            "из-за",
            "из-под",
            "через",
            "сквозь",
            "внутри",
            "снаружи",
            "среди",
            "между",
            "наверху",
            "внизу",
            "спереди",
            "сзади",
            "сбоку",
            "вперед",
            "назад",
            "вверх",
            "вниз",
            "вправо",
            "влево",
            "далеко",
            "близко",
            "высоко",
            "низко",
            "глубоко",
            "мелко",
            "рано",
            "поздно",
            "долго",
            "скоро",
            "сразу",
            "сейчас",
            "теперь",
            "тогда",
            "иногда",
            "часто",
            "редко",
            "всегда",
            "никогда",
            "уже",
            "еще",
            "тоже",
            "также",
            "причем",
            "притом",
            "впрочем",
            "однако",
            "зато",
            "только",
            "лишь",
            "исключительно",
            "особенно",
            "даже",
            "уж",
            "вовсе",
            "отнюдь",
            "совсем",
            "абсолютно",
            "полностью",
            "целиком",
            "почти",
            "примерно",
            "приблизительно",
            "ровно",
            "точно",
            "прямо",
            "как",
            "так",
            "столько",
            "сколько",
            "настолько",
            "до",
            "после",
            "перед",
            "за",
            "из-за",
            "из-под",
            "через",
            "сквозь",
            "внутри",
            "снаружи",
            "среди",
            "между",
            "наверху",
            "внизу",
            "спереди",
            "сзади",
            "сбоку",
            "вперед",
            "назад",
            "вверх",
            "вниз",
            "вправо",
            "влево",
            "далеко",
            "близко",
            "высоко",
            "низко",
            "глубоко",
            "мелко",
            "рано",
            "поздно",
            "долго",
            "скоро",
            "сразу",
            "сейчас",
            "теперь",
            "тогда",
            "иногда",
            "часто",
            "редко",
            "всегда",
            "никогда",
            "уже",
            "еще",
            "тоже",
            "также",
            "причем",
            "притом",
            "впрочем",
            "однако",
            "зато",
            "только",
            "лишь",
            "исключительно",
            "особенно",
            "даже",
            "уж",
            "вовсе",
            "отнюдь",
            "совсем",
            "абсолютно",
            "полностью",
            "целиком",
            "почти",
            "примерно",
            "приблизительно",
            "ровно",
            "точно",
            "прямо",
            "как",
            "так",
            "столько",
            "сколько",
            "настолько",
            "этом",
            "всем",
            "своих",
            "ними",
            "вами",
            "нами",
            "ими",
            "вами",
            "нами",
            "ими",
            "вами",
            "нами",
        }

        # Простая токенизация
        words = re.findall(r"\b[а-яёa-z]{3,}\b", text.lower())

        return [word for word in words if word not in russian_stopwords]

    def _preprocess_text(self, text: str) -> List[str]:
        """Предобработка текста для BM25"""
        # Удаляем специальные символы, оставляем буквы и цифры
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        # Токенизация
        return self._russian_tokenize(clean_text)

    def _build_bm25_index(self, documents: List[str]) -> BM25Okapi:
        """Построение BM25 индекса для коллекции документов"""
        tokenized_docs = [self._preprocess_text(doc) for doc in documents]
        return BM25Okapi(tokenized_docs)

    def _get_collection_documents(
        self, limit: int = 1000
    ) -> List[Tuple[str, str, Any]]:
        """Получение документов из коллекции для построения BM25 индекса"""
        try:
            logger.info(
                f" Загрузка документов из коллекции {self._collection_name} для BM25..."
            )

            scroll_results = self._qdrant_client.scroll(
                collection_name=self._collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            documents = []
            for hit in scroll_results[0]:
                doc_id = str(hit.id)
                chunk_text = hit.payload.get("chunk_text", "")
                documents.append((doc_id, chunk_text, hit))

            logger.info(f" Загружено {len(documents)} документов для BM25 индекса")
            return documents

        except Exception as e:
            logger.error(f" Ошибка при загрузке документов для BM25: {e}")
            return []

    def search_bm25(
        self, query: str, top_k: int = 10, bm25_weight: float = 1.2
    ) -> List[Dict[str, Any]]:
        """Поиск с использованием BM25"""
        try:
            cache_key = f"{self._collection_name}_bm25"

            # Проверяем кэш или строим индекс
            if cache_key not in self._bm25_cache:
                documents_data = self._get_collection_documents()
                if not documents_data:
                    return []

                # Сохраняем документы в кэше
                self._documents_cache[cache_key] = documents_data

                # Строим BM25 индекс
                document_texts = [doc[1] for doc in documents_data]
                self._bm25_cache[cache_key] = self._build_bm25_index(document_texts)

                logger.info(
                    f" Построен BM25 индекс для {len(documents_data)} документов"
                )

            # Получаем данные из кэша
            bm25 = self._bm25_cache[cache_key]
            documents_data = self._documents_cache[cache_key]

            # Токенизируем запрос
            tokenized_query = self._preprocess_text(query)

            if not tokenized_query:
                logger.warning(" Запрос не содержит значимых терминов для BM25")
                return []

            # Получаем BM25 скоринги
            doc_scores = bm25.get_scores(tokenized_query)

            # Сортируем документы по релевантности
            scored_docs = []
            for i, (doc_id, doc_text, hit) in enumerate(documents_data):
                if doc_scores[i] > 0:  # Только документы с ненулевым скором
                    scored_docs.append(
                        {
                            "id": doc_id,
                            "score": doc_scores[i] * bm25_weight,  # Применяем вес BM25
                            "payload": hit.payload,
                            "search_type": "bm25",
                            "original_score": doc_scores[i],
                        }
                    )

            # Сортируем и возвращаем top_k
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            results = scored_docs[:top_k]

            logger.info(f" BM25 поиск: найдено {len(results)} документов")
            if results:
                logger.info(
                    f" BM25 скоринги: {[r['original_score'] for r in results[:3]]}"
                )

            return results

        except Exception as e:
            logger.error(f" Ошибка в BM25 поиске: {e}")
            return []

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.4,
        fusion_alpha: float = 0.7,
        bm25_weight: float = 1.2,
        use_rerank: bool = True,
    ) -> List[Any]:
        """
        Гибридный поиск: комбинация семантического и BM25

        Args:
            query: Поисковый запрос
            top_k: Количество возвращаемых результатов
            score_threshold: Порог релевантности
            fusion_alpha: Коэффициент слияния (0-1)
            bm25_weight: Вес для BM25 скоринга
            use_rerank: Использовать переранжирование
        """
        try:
            # Параллельный поиск
            semantic_results = self.search_knn_by_query(
                query=query,
                top_k=top_k * 3,  # Берем больше для последующего слияния
                score_threshold=score_threshold,
            )

            bm25_results = self.search_bm25(
                query=query, top_k=top_k * 3, bm25_weight=bm25_weight
            )

            logger.info(
                f" Гибридный поиск: семантических={len(semantic_results)}, BM25={len(bm25_results)}"
            )

            # Слияние результатов
            if use_rerank:
                fused_results = self._fuse_with_rerank(
                    semantic_results, bm25_results, alpha=fusion_alpha, top_k=top_k
                )
            else:
                fused_results = self._fuse_simple(
                    semantic_results, bm25_results, alpha=fusion_alpha, top_k=top_k
                )

            logger.info(f" Объединенных результатов: {len(fused_results)}")

            return fused_results

        except Exception as e:
            logger.error(f" Ошибка в гибридном поиске: {e}")
            # Fallback to semantic search
            return self.search_knn_by_query(
                query=query, top_k=top_k, score_threshold=score_threshold
            )

    def _fuse_with_rerank(
        self,
        semantic_results: List,
        bm25_results: List,
        alpha: float = 0.7,
        top_k: int = 5,
    ) -> List[Any]:
        """Слияние с переранжированием на основе комбинированного скоринга"""
        try:
            # Создаем словарь для всех результатов
            all_results = {}

            # Обрабатываем семантические результаты
            for i, hit in enumerate(semantic_results):
                result_id = hit.id
                semantic_score = hit.score
                position_boost = 1.0 / (i + 1)  # Буст за позицию

                combined_score = (alpha * semantic_score) + (
                    (1 - alpha) * position_boost
                )

                all_results[result_id] = {
                    "score": combined_score,
                    "semantic_score": semantic_score,
                    "bm25_score": 0,
                    "item": {"id": hit.id, "score": hit.score, "payload": hit.payload},
                    "search_type": "semantic",
                    "position": i,
                }

            # Обрабатываем BM25 результаты
            for i, result in enumerate(bm25_results):
                result_id = result["id"]
                bm25_score = result["score"]
                position_boost = 1.0 / (i + 1)

                if result_id in all_results:
                    # Уже есть семантический результат - комбинируем скоринг
                    existing = all_results[result_id]
                    combined_score = (alpha * existing["semantic_score"]) + (
                        (1 - alpha) * bm25_score
                    )

                    all_results[result_id].update(
                        {
                            "score": combined_score,
                            "bm25_score": bm25_score,
                            "search_type": "hybrid",
                        }
                    )
                else:
                    # Только BM25 результат
                    combined_score = (1 - alpha) * bm25_score
                    all_results[result_id] = {
                        "score": combined_score,
                        "semantic_score": 0,
                        "bm25_score": bm25_score,
                        "item": {
                            "id": result["id"],
                            "score": result["score"],
                            "payload": result["payload"],
                        },
                        "search_type": "bm25",
                        "position": i,
                    }

            # Переранжируем по комбинированному скору
            ranked_results = sorted(
                all_results.values(), key=lambda x: x["score"], reverse=True
            )[:top_k]

            # Конвертируем в формат результата поиска
            final_results = []
            for result in ranked_results:
                hit = result["item"]

                class MockHit:
                    def __init__(self, id, score, payload, search_type):
                        self.id = id
                        self.score = score
                        self.payload = payload
                        self.search_type = search_type

                final_results.append(
                    MockHit(
                        hit["id"],
                        result["score"],
                        hit["payload"],
                        result["search_type"],
                    )
                )

            return final_results

        except Exception as e:
            logger.error(f" Ошибка при переранжировании: {e}")
            return self._fuse_simple(semantic_results, bm25_results, alpha, top_k)

    def _fuse_simple(
        self,
        semantic_results: List,
        bm25_results: List,
        alpha: float = 0.7,
        top_k: int = 5,
    ) -> List[Any]:
        """Простое слияние результатов"""
        try:
            # Объединяем и убираем дубликаты
            all_hits = {}

            # Добавляем семантические результаты
            for hit in semantic_results:
                all_hits[hit.id] = {
                    "hit": hit,
                    "semantic_score": hit.score,
                    "bm25_score": 0,
                    "type": "semantic",
                }

            # Добавляем BM25 результаты
            for result in bm25_results:
                result_id = result["id"]
                if result_id in all_hits:
                    # Обновляем существующий
                    all_hits[result_id]["bm25_score"] = result["score"]
                    all_hits[result_id]["type"] = "hybrid"
                else:
                    # Создаем новый
                    class MockHit:
                        def __init__(self, id, score, payload):
                            self.id = id
                            self.score = score
                            self.payload = payload

                    all_hits[result_id] = {
                        "hit": MockHit(
                            result["id"], result["score"], result["payload"]
                        ),
                        "semantic_score": 0,
                        "bm25_score": result["score"],
                        "type": "bm25",
                    }

            # Вычисляем комбинированные скоринги и сортируем
            combined_results = []
            for result_id, data in all_hits.items():
                combined_score = (alpha * data["semantic_score"]) + (
                    (1 - alpha) * data["bm25_score"]
                )

                class CombinedHit:
                    def __init__(self, hit, score, search_type):
                        self.id = hit.id
                        self.score = score
                        self.payload = hit.payload
                        self.search_type = search_type

                combined_results.append(
                    CombinedHit(data["hit"], combined_score, data["type"])
                )

            combined_results.sort(key=lambda x: x.score, reverse=True)
            return combined_results[:top_k]

        except Exception as e:
            logger.error(f" Ошибка при простом слиянии: {e}")
            return semantic_results[:top_k]

    def _check_connections(self):
        """Улучшенная проверка соединений"""
        # Проверка Qdrant
        try:
            collections = self._qdrant_client.get_collections()
            logger.info(f" Qdrant: {len(collections.collections)} коллекций")

            if self._collection_name in [col.name for col in collections.collections]:
                collection_info = self._qdrant_client.get_collection(
                    self._collection_name
                )
                logger.info(
                    f" Коллекция '{self._collection_name}' (размерность: {collection_info.config.params.vectors.size})"
                )
            else:
                logger.warning(f" Коллекция '{self._collection_name}' не найдена")
        except Exception as e:
            logger.error(f" Qdrant: {e}")

        # Универсальная проверка Ollama
        self._check_ollama_models()

    def _check_ollama_models(self):
        """Проверка доступности моделей в Ollama"""
        try:
            # Проверяем оба endpoint'а
            endpoints = [(self._generate_url, "Generate"), (self._embed_url, "Embed")]

            all_models = []

            for url, endpoint_type in endpoints:
                try:
                    response = requests.get(f"{url}/api/tags", timeout=10)
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        model_names = [m["name"] for m in models]
                        all_models.extend(model_names)
                        logger.info(f" {endpoint_type} Ollama: {len(models)} моделей")

                        # Проверяем нужные модели для этого endpoint'а
                        if endpoint_type == "Generate":
                            if self._generate_model in model_names:
                                logger.info(
                                    f"   Модель генерации '{self._generate_model}' доступна"
                                )
                            else:
                                logger.error(
                                    f"   Модель генерации '{self._generate_model}' не найдена"
                                )

                        if endpoint_type == "Embed" or url == self._generate_url:
                            if self._embed_model in model_names:
                                logger.info(
                                    f"   Модель эмбеддингов '{self._embed_model}' доступна"
                                )
                            else:
                                logger.error(
                                    f"   Модель эмбеддингов '{self._embed_model}' не найдена"
                                )

                    else:
                        logger.error(
                            f" {endpoint_type} Ollama: HTTP {response.status_code}"
                        )

                except Exception as e:
                    logger.error(f" {endpoint_type} Ollama: {e}")

            # Итоговая проверка
            if self._embed_model in all_models:
                logger.info(f" Модель эмбеддингов '{self._embed_model}' подтверждена")
            else:
                logger.error(
                    f" Критическая ошибка: модель эмбеддингов '{self._embed_model}' недоступна"
                )
                logger.info(f" Все доступные модели: {list(set(all_models))}")

        except Exception as e:
            logger.error(f" Общая ошибка проверки Ollama: {e}")

    def _format_text_for_embedding(self, text: str, is_query: bool = True) -> str:
        """Форматирование текста для BGE-M3"""
        if "bge-m3" in self._embed_model:
            if is_query:
                return f"В этом задании представлен запрос к поисковой системе. Представьте запрос для извлечения соответствующих документов: {text}"
            else:
                return f"В этом задании представлен документ. Представьте документ для извлечения соответствующих запросов: {text}"
        return text

    def vectorize_query(self, *, text: str) -> List[float]:
        """Получение эмбеддинга из Ollama с улучшенной диагностикой"""
        formatted_text = self._format_text_for_embedding(text, is_query=True)
        payload = {"model": self._embed_model, "prompt": formatted_text}

        try:
            logger.info(f" Запрос эмбеддинга для модели: {self._embed_model}")

            response = requests.post(
                f"{self._embed_url}/api/embeddings", json=payload, timeout=300
            )

            if response.status_code == 200:
                embedding = response.json().get("embedding", [])
                logger.info(f" Получен эмбеддинг размерности: {len(embedding)}")

                # Нормализация для BGE-M3
                if embedding and "bge-m3" in self._embed_model:
                    embedding_array = np.array(embedding).reshape(1, -1)
                    normalized_embedding = normalize(embedding_array, norm="l2")[0]
                    embedding = normalized_embedding.tolist()
                    logger.info(" Эмбеддинг нормализован для BGE-M3")

                return embedding
            else:
                logger.error(f" Ошибка получения эмбеддинга: {response.status_code}")
                logger.error(f" Ответ: {response.text[:200]}...")
                return []

        except Exception as e:
            logger.error(f" Ошибка подключения к Ollama для эмбеддингов: {e}")
            return []

    def search_knn_by_query(
        self, *, query: str, top_k: int = 5, score_threshold: float = 0.3
    ) -> Any:
        """Поиск похожих документов в Qdrant"""
        query_embedding = self.vectorize_query(text=query)
        if not query_embedding:
            logger.error("Не удалось получить эмбеддинг для запроса")
            return []

        try:
            search_result = self._qdrant_client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )

            logger.info(f" Найдено документов: {len(search_result)}")
            if search_result:
                scores = [hit.score for hit in search_result]
                logger.info(
                    f" Диапазон скорингов: min={min(scores):.3f}, max={max(scores):.3f}"
                )

            return search_result
        except Exception as e:
            logger.error(f" Ошибка поиска в Qdrant: {e}")
            return []

    def retrieve_context(self) -> str:
        """Заглушка для retrieve_context (используется retrieve_context_enhanced)"""
        raise NotImplementedError(
            "Use retrieve_context_enhanced in EnhancedRagPipeline"
        )

    def invoke(self, prompt: str) -> Any:
        """Генерация ответа с помощью Ollama"""
        payload = {
            "model": self._generate_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_ctx": 4096},
        }
        try:
            response = requests.post(
                f"{self._generate_url}/api/generate", json=payload, timeout=6000
            )
            if response.status_code == 200:

                class Response:
                    def __init__(self, content):
                        self.content = content

                return Response(
                    response.json().get("response", "Ошибка генерации ответа")
                )
            logger.error(f" Ошибка Ollama при генерации: {response.status_code}")
            return Response(f"Ошибка Ollama: {response.status_code}")
        except Exception as e:
            logger.error(f" Ошибка подключения к Ollama: {e}")
            return Response(f"Ошибка подключения к Ollama: {e}")

    @property
    def temperature(self):
        """Mock temperature property for compatibility"""
        return 0.1

    @temperature.setter
    def temperature(self, value: float):
        """Mock temperature setter"""
        pass
