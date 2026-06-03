"""SemanticRouter —— 语义路由模块。

根据用户输入的语义将请求路由到不同的处理器（Agent），
例如情感分析、知识问答、日常问候等。支持路由结果缓存。
"""

import hashlib
import json
from dataclasses import dataclass, field

import redis

from backend.core.redis_client import get_redis_client
from backend.core.embeddings_cache import EmbeddingsCache
from backend.utils.vectorizer import cosine_similarity, get_embedding


# ------------------------------------------------------------------
# Route 数据类
# ------------------------------------------------------------------

@dataclass
class Route:
    """路由规则定义。

    Attributes:
        name: 路由名称（唯一标识）。
        references: 参考示例文本列表，用于计算语义相似度。
        metadata: 附加元数据。
        distance_threshold: 语义距离阈值，越小匹配越严格。
    """
    name: str
    references: list[str]
    metadata: dict | None = None
    distance_threshold: float = 0.3

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ------------------------------------------------------------------
# 默认路由
# ------------------------------------------------------------------

DEFAULT_ROUTES: list[Route] = [
    Route(
        name="sentiment",
        references=["我今天很开心", "心情不好很郁闷", "太让人生气了",
                     "这件事让我好伤心", "感到非常幸福"],
        metadata={"type": "sentiment_analysis", "description": "情感分析"},
        distance_threshold=0.3,
    ),
    Route(
        name="qa",
        references=["什么是人工智能？", "如何学习Python编程？",
                     "请解释一下量子力学的基本原理", "为什么会下雨？",
                     "深度学习与机器学习的区别是什么？"],
        metadata={"type": "knowledge_qa", "description": "知识问答"},
        distance_threshold=0.3,
    ),
    Route(
        name="greeting",
        references=["你好", "早上好", "嗨，最近怎么样？",
                     "好久不见，你还好吗？", "下午好"],
        metadata={"type": "greeting", "description": "日常问候"},
        distance_threshold=0.3,
    ),
    Route(
        name="fallback",
        references=["请帮我处理一下", "我有一个问题",
                     "你能帮我吗？", "其他"],
        metadata={"type": "fallback", "description": "兜底处理"},
        distance_threshold=0.6,  # 较高阈值以便兜底
    ),
]


# ------------------------------------------------------------------
# SemanticRouter
# ------------------------------------------------------------------

class SemanticRouter:
    """语义路由器。

    将用户 query 与各路由的参考示例做语义相似度比较，
    返回最匹配的 Route。支持路由结果缓存。
    """

    def __init__(
        self,
        name: str = "default",
        routes: list[Route] | None = None,
        redis_url: str | None = None,
        _redis_client: "redis.Redis | None" = None,
    ):
        """初始化语义路由器。

        Args:
            name: 路由器命名空间。
            routes: 初始路由列表，为 None 时使用 DEFAULT_ROUTES。
            redis_url: Redis 连接地址。
            _redis_client: 测试用——直接注入 Redis 客户端，绕过连接。
        """
        self.name = name
        if _redis_client is not None:
            self._redis: redis.Redis = _redis_client
        else:
            self._redis = get_redis_client(redis_url)
        self._embed_cache = EmbeddingsCache(
            name=f"{name}_router_emb", ttl=86400, redis_url=redis_url
        )
        if _redis_client is not None:
            self._embed_cache._redis = _redis_client
        self._routes: dict[str, Route] = {}
        # 每个 route 的 reference embeddings 缓存: {route_name: [emb1, emb2, ...]}
        self._ref_embs: dict[str, list[list[float]]] = {}

        # 初始化路由
        for route in (routes or DEFAULT_ROUTES):
            self.add_route(route)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_key(self, query: str) -> str:
        """路由结果缓存 key。"""
        return f"rvl:semantic_router:{self.name}:cache:{self._hash_text(query)}"

    def _route_key(self) -> str:
        """路由定义存储 key。"""
        return f"rvl:semantic_router:{self.name}:routes"

    def _compute_ref_embeddings(self, route: Route) -> list[list[float]]:
        """计算并缓存 route 所有 reference 的 embedding。"""
        embs = []
        for ref in route.references:
            emb = get_embedding(ref, self._embed_cache)
            embs.append(emb)
        return embs

    def _persist_routes(self) -> None:
        """将路由定义持久化到 Redis。"""
        data = {}
        for r in self._routes.values():
            data[r.name] = {
                "name": r.name,
                "references": r.references,
                "metadata": r.metadata,
                "distance_threshold": r.distance_threshold,
            }
        self._redis.set(self._route_key(), json.dumps(data, ensure_ascii=False))

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def add_route(self, route: Route) -> None:
        """动态添加路由规则。

        Args:
            route: Route 实例。
        """
        self._routes[route.name] = route
        self._ref_embs[route.name] = self._compute_ref_embeddings(route)
        self._persist_routes()

    def route(self, query: str) -> Route:
        """根据输入匹配最佳路由。

        优先查缓存，未命中则计算语义相似度。

        Args:
            query: 用户输入文本。

        Returns:
            最匹配的 Route；若所有 route 均未达阈值，返回 fallback 路由。
        """
        # 1. 查缓存
        cached = self._redis.get(self._cache_key(query))
        if cached:
            route_name = cached.decode("utf-8") if isinstance(cached, bytes) else cached
            if route_name in self._routes:
                return self._routes[route_name]

        # 2. 计算 query embedding
        query_emb = get_embedding(query, self._embed_cache)

        # 3. 与每个 route 比较，取最小距离
        best_route: Route | None = None
        best_distance = float("inf")

        for route_name, route in self._routes.items():
            if route_name == "fallback":
                continue  # fallback 不参与常规匹配
            ref_embs = self._ref_embs.get(route_name, [])
            for ref_emb in ref_embs:
                distance = 1.0 - cosine_similarity(query_emb, ref_emb)
                if distance < best_distance:
                    best_distance = distance
                    best_route = route

        # 4. 判断是否命中
        if best_route is not None and best_distance < best_route.distance_threshold:
            result = best_route
        else:
            result = self._routes.get("fallback", best_route)

        # 5. 缓存结果（1 小时）
        if result:
            self._redis.set(self._cache_key(query), result.name, ex=3600)

        return result

    def __call__(self, query: str) -> Route:
        """__call__ 别名，语义同 route()。"""
        return self.route(query)

    def clear_cache(self) -> None:
        """清空路由结果缓存（路由定义保留）。"""
        pattern = f"rvl:semantic_router:{self.name}:cache:*"
        cursor = 0
        while True:
            cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
            if keys:
                self._redis.delete(*keys)
            if cursor == 0:
                break
