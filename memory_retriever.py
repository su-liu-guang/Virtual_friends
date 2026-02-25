import asyncio
import hashlib
import json
from typing import List, Optional, Dict, Any

import aiohttp
from nonebot import logger

from .clients import EmbeddingClient
from .config import ConfigManager
from .database import MemoryVector, Summary, ImportantEvent


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class RerankerClient:
    """调用 BAAI/bge-reranker-v2-m3 的通用 HTTP 客户端（网络 API）。"""

    def __init__(self, config: ConfigManager):
        self.api_url = config.get_env("reranker_api_url")
        self.api_key = config.get_env("reranker_api_key")
        self.model = config.get_env("reranker_model")
        self.timeout = int(config.get_env("reranker_timeout", "5") or 5)
        self.enabled = bool(self.api_url and self.api_key and self.model)

    async def rerank(self, query: str, texts: List[str]) -> Optional[List[float]]:
        if not self.enabled or not texts:
            return None

        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, data=json.dumps(payload), headers=headers, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        logger.warning(f"[Reranker] HTTP {resp.status}: {await resp.text()}")
                        return None
                    data = await resp.json()
                    # siliconflow 返回 {"results":[{"index":0,"relevance_score":...}, ...]}
                    if isinstance(data, dict):
                        if "results" in data and isinstance(data["results"], list):
                            scores = []
                            for item in data["results"]:
                                if isinstance(item, dict):
                                    if "relevance_score" in item:
                                        scores.append(float(item["relevance_score"]))
                                    elif "score" in item:
                                        scores.append(float(item["score"]))
                            if scores:
                                return scores
                        if "scores" in data and isinstance(data["scores"], list):
                            return [float(s) for s in data["scores"]]
                    logger.warning(f"[Reranker] 无法解析响应: {data}")
        except Exception as exc:
            logger.warning(f"[Reranker] 调用失败: {exc}")
        return None


class MemoryRetriever:
    """记忆向量存储与检索，支持向量召回 + rerank。"""

    def __init__(self, embedding_client: EmbeddingClient, config: ConfigManager):
        self.embedding_client = embedding_client
        self.reranker = RerankerClient(config)

    async def _embed(self, text: str) -> List[float]:
        return await self.embedding_client.get_embedding(text)

    async def upsert_summary(self, summary: Summary):
        content = summary.content.strip()
        if not content:
            return

        logger.info(
            f"[Memory] 回填摘要 id={summary.id} group={summary.group_id} level=L{summary.level} len={len(content)}"
        )
        content_hash = _md5(content)
        existing = await MemoryVector.get_or_none(ref_type="summary", ref_id=summary.id)
        if existing and existing.content_hash == content_hash:
            logger.debug(f"[Memory] 摘要 id={summary.id} 内容未变，跳过")
            return

        embedding = await self._embed(content)
        if not embedding:
            logger.warning(f"[Memory] 摘要 id={summary.id} 向量化失败，已跳过")
            return

        if existing:
            existing.content = content
            existing.content_hash = content_hash
            existing.embedding = embedding
            existing.level = summary.level
            await existing.save()
        else:
            await MemoryVector.create(
                group_id=summary.group_id,
                ref_type="summary",
                ref_id=summary.id,
                level=summary.level,
                content=content,
                content_hash=content_hash,
                embedding=embedding,
            )
        logger.info(f"[Memory] 摘要 id={summary.id} 已写入/更新向量")

    async def upsert_fact(self, fact: ImportantEvent):
        content = fact.event_content.strip()
        if not content:
            return

        logger.info(
            f"[Memory] 回填事实 id={fact.id} group={fact.group_id} len={len(content)}"
        )
        content_hash = _md5(content)
        existing = await MemoryVector.get_or_none(ref_type="fact", ref_id=fact.id)
        if existing and existing.content_hash == content_hash:
            logger.debug(f"[Memory] 事实 id={fact.id} 内容未变，跳过")
            return

        embedding = await self._embed(content)
        if not embedding:
            logger.warning(f"[Memory] 事实 id={fact.id} 向量化失败，已跳过")
            return

        if existing:
            existing.content = content
            existing.content_hash = content_hash
            existing.embedding = embedding
            existing.level = None
            await existing.save()
        else:
            await MemoryVector.create(
                group_id=fact.group_id,
                ref_type="fact",
                ref_id=fact.id,
                level=None,
                content=content,
                content_hash=content_hash,
                embedding=embedding,
            )
            logger.info(f"[Memory] 事实 id={fact.id} 已写入/更新向量")

    @staticmethod
    def _cosine(v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    async def retrieve(self, group_id: str, query: str, top_k: int = 12, final_k: int = 5) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        query_emb = await self._embed(query)
        if not query_emb:
            return []

        # 向量召回
        candidates = await MemoryVector.filter(group_id=group_id).all()
        scored = []
        for c in candidates:
            if not c.embedding:
                continue
            score = self._cosine(query_emb, c.embedding)
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored[:top_k]

        if not top_candidates:
            return []

        contents = [c.content for _, c in top_candidates]
        if self.reranker.enabled:
            logger.debug(
                f"[Reranker] 开始重排序，候选={len(contents)} query_len={len(query)}"
            )
        else:
            logger.debug("[Reranker] 未配置 reranker，直接用向量得分")

        rerank_scores = await self.reranker.rerank(query, contents)

        if rerank_scores and len(rerank_scores) == len(contents):
            logger.info(f"[Reranker] 重排序成功，使用 rerank 结果，候选={len(contents)}")
            reranked = sorted(
                zip(rerank_scores, top_candidates), key=lambda x: x[0], reverse=True
            )
            ordered = [item[1] for item in reranked]
        else:
            if self.reranker.enabled:
                logger.warning("[Reranker] 重排序失败或返回数量不匹配，回退向量相似度")
            ordered = top_candidates

        results = []
        for score, obj in ordered[:final_k]:
            tag = "事实" if obj.ref_type == "fact" else f"L{obj.level} 摘要"
            results.append({
                "tag": tag,
                "content": obj.content,
                "score": score,
            })

        return results