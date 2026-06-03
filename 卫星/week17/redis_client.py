"""Redis 连接管理——提供统一的 Redis 客户端获取接口。"""

import redis
from config import Config


def get_redis_client(redis_url: str | None = None) -> redis.Redis:
    """获取 Redis 客户端实例。

    Args:
        redis_url: Redis 连接地址，为 None 时使用全局配置。
    """
    url = redis_url or Config.REDIS_URL
    return redis.from_url(url)
