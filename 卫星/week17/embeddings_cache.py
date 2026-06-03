"""EmbeddingsCache —— 文本 Embedding 缓存模块。

缓存 文本 → 向量 的映射，避免对相同文本重复调用 embedding 模型，节约计算资源。
"""

import hashlib
import json
from typing import Optional

import redis

from backend.core.redis_client import get_redis_client


class EmbeddingsCache:
    """文本 Embedding 缓存。

    以 hash(text) 作为 key 存储向量，避免文本过长导致 Redis key 超限。
    """

    def __init__(self, name: str = "default", ttl: int = 3600,
                 redis_url: str | None = None):
        """初始化缓存实例。

        Args:
            name: 缓存命名空间，区分不同业务场景。
            ttl: 默认过期时间（秒）。
            redis_url: Redis 连接地址，为 None 时使用全局配置。
        """
        self.name = name
        self.ttl = ttl
        self._redis: redis.Redis = get_redis_client(redis_url)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_text(text: str) -> str:
        """对文本做 SHA-256 哈希，生成定长 key。"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _make_key(self, text: str) -> str:
        """生成 Redis 存储 key。

        格式: rvl:embed_cache:{name}:{hash}
        """
        return f"rvl:embed_cache:{self.name}:{self._hash_text(text)}"

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def set_embedding(self, text: str, embedding: list[float],
                      ttl: int | None = None) -> None:
        """缓存文本的 embedding 向量。

        Args:
            text: 原始文本。
            embedding: 向量（float 列表）。
            ttl: 过期时间（秒），为 None 时使用实例默认 TTL。
        """
        key = self._make_key(text)
        value = json.dumps(embedding)
        expire = ttl if ttl is not None else self.ttl
        self._redis.set(key, value, ex=expire)

    def get_embedding(self, text: str) -> list[float] | None:
        """查询缓存的 embedding 向量。

        Args:
            text: 原始文本。

        Returns:
            命中返回向量列表，未命中返回 None。
        """
        key = self._make_key(text)
        value = self._redis.get(key)
        if value is None:
            return None
        return json.loads(value)

    def delete_embedding(self, text: str) -> None:
        """删除指定文本的 embedding 缓存。"""
        key = self._make_key(text)
        self._redis.delete(key)

    def exists(self, text: str) -> bool:
        """检查指定文本的 embedding 是否已缓存。"""
        key = self._make_key(text)
        return bool(self._redis.exists(key))
