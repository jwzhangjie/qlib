import aioredis
from typing import Optional
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)
redis_client: Optional[aioredis.Redis] = None


async def init_redis():
    """初始化Redis连接"""
    global redis_client
    try:
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50
        )
        
        # 测试连接
        await redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise


async def get_redis() -> aioredis.Redis:
    """获取Redis客户端"""
    if redis_client is None:
        await init_redis()
    return redis_client


async def close_redis():
    """关闭Redis连接"""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")


class RedisKeys:
    """Redis键名常量"""
    
    @staticmethod
    def user_session(user_id: str) -> str:
        return f"session:{user_id}"
    
    @staticmethod
    def stock_data(symbol: str) -> str:
        return f"stock:{symbol}"
    
    @staticmethod
    def model_cache(model_id: str) -> str:
        return f"model:{model_id}"
    
    @staticmethod
    def backtest_result(backtest_id: str) -> str:
        return f"backtest:{backtest_id}"
    
    @staticmethod
    def data_update_task(task_id: str) -> str:
        return f"data_update:{task_id}"
    
    @staticmethod
    def rate_limit(ip: str) -> str:
        return f"rate_limit:{ip}"
    
    @staticmethod
    def model_training(model_id: str) -> str:
        return f"training:{model_id}"
    
    @staticmethod
    def cache_key(prefix: str, key: str) -> str:
        return f"cache:{prefix}:{key}"