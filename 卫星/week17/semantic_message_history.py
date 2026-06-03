"""SemanticMessageHistory —— 会话消息历史管理模块。

按 session_id 隔离不同会话，使用 Redis List 存储多轮对话历史，
支持最近消息查询和语义检索。
"""

import json
from datetime import datetime, timezone

import redis

from backend.core.redis_client import get_redis_client
from backend.core.embeddings_cache import EmbeddingsCache
from backend.utils.vectorizer import cosine_similarity, get_embedding


class SemanticMessageHistory:
    """会话消息历史管理。

    按 session_id 隔离存储，每条消息包含 role、content、timestamp 和
    可选的 metadata。底层使用 Redis List（LPUSH 追加，LRANGE 查询）。
    """

    VALID_ROLES = {"system", "user", "llm", "tool"}

    def __init__(
        self,
        name: str = "default",
        ttl: int = 86400,
        redis_url: str | None = None,
    ):
        """初始化消息历史管理器。

        Args:
            name: 命名空间，区分不同业务场景。
            ttl: 消息默认过期时间（秒），默认 24 小时。
            redis_url: Redis 连接地址。
        """
        self.name = name
        self.ttl = ttl
        self._redis: redis.Redis = get_redis_client(redis_url)
        # 用于缓存消息 embedding
        self._embed_cache = EmbeddingsCache(
            name=f"{name}_msg_emb", ttl=ttl, redis_url=redis_url
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _msg_key(self, session_id: str) -> str:
        """消息列表 key: rvl:message_history:{name}:{session_id}:messages"""
        return f"rvl:message_history:{self.name}:{session_id}:messages"

    def _emb_list_key(self, session_id: str) -> str:
        """embedding 列表 key: rvl:message_history:{name}:{session_id}:embeddings"""
        return f"rvl:message_history:{self.name}:{session_id}:embeddings"

    @staticmethod
    def _now_iso() -> str:
        """返回当前 UTC 时间 ISO 字符串。"""
        return datetime.now(timezone.utc).isoformat()

    def _validate_role(self, role: str) -> None:
        """校验 role 是否合法。"""
        if role not in self.VALID_ROLES:
            raise ValueError(
                f"无效的 role: '{role}'，允许值: {self.VALID_ROLES}"
            )

    def _build_message(
        self, role: str, content: str, metadata: dict | None = None
    ) -> str:
        """构建一条消息的 JSON 字符串。"""
        msg = {
            "role": role,
            "content": content,
            "timestamp": self._now_iso(),
            "metadata": metadata or {},
        }
        return json.dumps(msg, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> None:
        """添加单条消息到指定 session。

        Args:
            session_id: 会话标识。
            role: 角色（system / user / llm / tool）。
            content: 消息正文。
            metadata: 附加元数据。
        """
        self._validate_role(role)
        msg_json = self._build_message(role, content, metadata)
        msg_key = self._msg_key(session_id)
        self._redis.lpush(msg_key, msg_json)
        self._redis.expire(msg_key, self.ttl)

    def add_messages(self, session_id: str, messages: list[dict]) -> None:
        """批量添加消息。

        Args:
            session_id: 会话标识。
            messages: 消息列表，每条为 {"role": ..., "content": ...,
                       "metadata"(可选): ...}。
        """
        for msg in messages:
            self.add_message(
                session_id=session_id,
                role=msg["role"],
                content=msg["content"],
                metadata=msg.get("metadata"),
            )

    def get_last_messages(
        self, session_id: str, limit: int = 10
    ) -> list[dict]:
        """获取指定 session 最近 n 条消息（按时间倒序）。

        Args:
            session_id: 会话标识。
            limit: 返回数量上限。

        Returns:
            消息列表（每条为 dict），最新的在前。
        """
        msg_key = self._msg_key(session_id)
        raw_list = self._redis.lrange(msg_key, 0, limit - 1)
        return [json.loads(raw) for raw in raw_list]

    def clear_session(self, session_id: str) -> None:
        """清空指定 session 的全部消息。"""
        self._redis.delete(self._msg_key(session_id))
        self._redis.delete(self._emb_list_key(session_id))

    def search_similar_messages(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
        distance_threshold: float = 0.7,
    ) -> list[dict]:
        """语义检索与 query 最相似的历史消息。

        将 query 向量化后与消息 embedding 做余弦相似度比较，
        返回距离在 threshold 内的 top_k 条消息。

        Args:
            query: 查询文本。
            session_id: 限定搜索的 session，为 None 时不限定。
            top_k: 返回条数。
            distance_threshold: 语义距离阈值。

        Returns:
            匹配的消息列表，按相似度降序排列。
        """
        query_emb = get_embedding(query, self._embed_cache)

        # 收集待检索的 session
        if session_id:
            session_ids = [session_id]
        else:
            # 扫描所有匹配的 key
            pattern = f"rvl:message_history:{self.name}:*:messages"
            session_ids = []
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(
                    cursor, match=pattern, count=100
                )
                for key in keys:
                    # 从 key 中提取 session_id
                    # key 格式: rvl:message_history:{name}:{session_id}:messages
                    parts = key.decode("utf-8").split(":")
                    # 取倒数第二个部分作为 session_id
                    # 但 session_id 可能包含冒号... 实际上不太可能
                    # 更安全的方式: 去掉前缀和后缀
                    prefix = f"rvl:message_history:{self.name}:"
                    suffix = ":messages"
                    sid = key.decode("utf-8")[len(prefix):-len(suffix)]
                    session_ids.append(sid)
                if cursor == 0:
                    break

        if not session_ids:
            return []

        # 收集所有候选消息及其 embedding
        candidates: list[tuple[float, dict]] = []  # (distance, message)

        for sid in session_ids:
            msg_key = self._msg_key(sid)
            raw_msgs = self._redis.lrange(msg_key, 0, -1)
            for raw in raw_msgs:
                msg = json.loads(raw)
                msg_emb = get_embedding(msg["content"], self._embed_cache)
                distance = 1.0 - cosine_similarity(query_emb, msg_emb)
                if distance < distance_threshold:
                    candidates.append((distance, msg))

        # 按距离升序排序（越小越相似），取 top_k
        candidates.sort(key=lambda x: x[0])
        return [
            {**msg, "_distance": round(dist, 4)}
            for dist, msg in candidates[:top_k]
        ]
