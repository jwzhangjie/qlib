import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用配置"""
    
    # 基础配置
    APP_NAME: str = "Qlib Trading Platform"
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # 安全配置
    SECRET_KEY: str = Field(env="SECRET_KEY")
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # 数据库配置
    DATABASE_URL: str = Field(env="DATABASE_URL")
    
    # Redis配置
    REDIS_URL: str = Field(env="REDIS_URL")
    
    # CORS配置
    ALLOWED_HOSTS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="ALLOWED_HOSTS"
    )
    
    # Qlib配置
    QLIB_DATA_PATH: Optional[str] = Field(default=None, env="QLIB_DATA_PATH")
    
    # Celery配置
    CELERY_BROKER_URL: str = Field(env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(env="CELERY_RESULT_BACKEND")
    
    # 日志配置
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # 文件上传配置
    UPLOAD_DIR: str = Field(default="uploads", env="UPLOAD_DIR")
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    
    # 数据配置
    DEFAULT_DATA_SOURCE: str = Field(default="yahoo", env="DEFAULT_DATA_SOURCE")
    DATA_UPDATE_INTERVAL: int = Field(default=3600, env="DATA_UPDATE_INTERVAL")  # 1小时
    
    # 模型配置
    MODEL_CACHE_DIR: str = Field(default="models", env="MODEL_CACHE_DIR")
    DEFAULT_MODEL_TYPE: str = Field(default="lightgbm", env="DEFAULT_MODEL_TYPE")
    
    # 回测配置
    BACKTEST_START_DATE: str = Field(default="2020-01-01", env="BACKTEST_START_DATE")
    BACKTEST_END_DATE: str = Field(default="2023-12-31", env="BACKTEST_END_DATE")
    BACKTEST_INITIAL_CASH: float = Field(default=1000000.0, env="BACKTEST_INITIAL_CASH")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# 创建配置实例
settings = Settings()