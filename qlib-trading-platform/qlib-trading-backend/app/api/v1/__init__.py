from fastapi import APIRouter
from app.api.v1 import auth, users, stocks, models, backtest, data, portfolio

api_router = APIRouter()

# 认证相关
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])

# 股票相关
api_router.include_router(stocks.router, prefix="/stocks", tags=["stocks"])

# 模型相关
api_router.include_router(models.router, prefix="/models", tags=["models"])

# 回测相关
api_router.include_router(backtest.router, prefix="/backtest", tags=["backtest"])

# 数据管理
api_router.include_router(data.router, prefix="/data", tags=["data"])

# 投资组合
api_router.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])