"""SemanticCache —— 语义缓存模块。

缓存 用户prompt → LLM response，支持基于语义相似度的缓存命中，
避免对语义相似的提问重复调用大模型。
"""

import hashlib
import json

import redis

from backend.core.redis_client import get_redis_client
from backend.core.embeddings_cache import EmbeddingsCache
from backend.utils.vectorizer import cosine_similarity, get_embedding


class SemanticCache:
    """语义缓存。

    将 prompt 向量化后与历史缓存做余弦相似度比较，
    距离小于 distance_threshold 时返回缓存结果。
    """

    def __init__(
        self,
        name: str = "default",
        ttl: int = 3600,
        distance_threshold: float = 0.1,
        redis_url: str | None = None,
    ):
        """初始化语义缓存。

        Args:
            name: 缓存命名空间，区分不同业务场景。
            ttl: 默认过期时间（秒）。
            distance_threshold: 语义距离阈值（0~1），越小匹配越严格。
            redis_url: Redis 连接地址。
        """
        self.name = name
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self._redis: redis.Redis = get_redis_client(redis_url)
        # 内嵌 EmbeddingsCache 缓存 prompt 的 embedding
        self._embed_cache = EmbeddingsCache(
            name=f"{name}_sem_emb", ttl=ttl, redis_url=redis_url
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_text(text: str) -> str:
        """SHA-256 哈希。"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _entry_key(self, h: str) -> str:
        return f"rvl:semantic_cache:{self.name}:entry:{h}"

    def _emb_key(self, h: str) -> str:
        return f"rvl:semantic_cache:{self.name}:emb:{h}"

    def _hashes_key(self) -> str:
        return f"rvl:semantic_cache:{self.name}:hashes"

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def store(
        self,
        prompt: str,
        response: str,
        metadata: dict | None = None,
    ) -> None:
        """缓存一对 prompt 与 LLM response。

        Args:
            prompt: 用户提问原文。
            response: LLM 返回结果。
            metadata: 附加元数据（如模型名、调用时间等）。
        """
        h = self._hash_text(prompt)
        # 向量化 prompt 并缓存 embedding
        emb = get_embedding(prompt, self._embed_cache)
        self._redis.set(self._emb_key(h), json.dumps(emb), ex=self.ttl)
        # 存储条目
        entry = json.dumps({
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {},
        })
        self._redis.set(self._entry_key(h), entry, ex=self.ttl)
        # 记录 hash 到集合
        self._redis.sadd(self._hashes_key(), h)
        self._redis.expire(self._hashes_key(), self.ttl)

    def check(self, prompt: str) -> dict | None:
        """查询语义缓存。

        将 prompt 向量化后与所有历史缓存做余弦相似度比较，
        返回距离最小且在阈值内的条目。

        Args:
            prompt: 当前用户提问。

        Returns:
            命中返回 {"prompt", "response", "metadata"}，未命中返回 None。
        """
        input_emb = get_embedding(prompt, self._embed_cache)

        hashes = self._redis.smembers(self._hashes_key())
        if not hashes:
            return None

        best_distance = float("inf")
        best_entry: dict | None = None

        for h in hashes:
            emb_json = self._redis.get(self._emb_key(h))
            if emb_json is None:
                continue
            stored_emb = json.loads(emb_json)
            distance = 1.0 - cosine_similarity(input_emb, stored_emb)

            if distance < best_distance:
                entry_json = self._redis.get(self._entry_key(h))
                if entry_json is None:
                    continue
                best_distance = distance
                best_entry = json.loads(entry_json)

        if best_distance < self.distance_threshold and best_entry is not None:
            return best_entry
        return None

    def clear(self) -> None:
        """清空当前 name 下的所有语义缓存。"""
        hashes = self._redis.smembers(self._hashes_key())
        keys_to_delete: list[str] = []
        for h in hashes:
            keys_to_delete.extend([self._entry_key(h), self._emb_key(h)])
        if keys_to_delete:
            self._redis.delete(*keys_to_delete)
        self._redis.delete(self._hashes_key())
