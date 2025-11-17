from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from app.core.config import settings
from app.core.database import init_db
from app.core.redis import init_redis
from app.api.v1.router import api_router
from app.core.logger import setup_logging

# 加载环境变量
load_dotenv()

# 设置日志
setup_logging()

security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    await init_db()
    await init_redis()
    
    # 初始化Qlib
    try:
        import qlib
        from qlib.constant import REG_CN
        
        # 设置Qlib数据路径
        qlib_data_path = settings.QLIB_DATA_PATH or "~/.qlib/qlib_data/cn_data"
        qlib.init(provider_uri=qlib_data_path, region=REG_CN)
        print(f"Qlib initialized with data path: {qlib_data_path}")
    except Exception as e:
        print(f"Warning: Qlib initialization failed: {e}")
        print("Continuing without Qlib integration...")
    
    yield
    
    # 关闭时清理
    print("Shutting down application...")


# 创建FastAPI应用
app = FastAPI(
    title="Qlib Trading Platform API",
    description="基于Qlib的量化交易平台后端API",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router, prefix="/api/v1")

# 静态文件服务
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Qlib Trading Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "qlib-trading-platform"}


@app.get("/favicon.ico")
async def favicon():
    """Favicon"""
    return FileResponse("static/favicon.ico") if os.path.exists("static/favicon.ico") else None


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )