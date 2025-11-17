from fastapi import APIRouter, Depends, HTTPException, Query, HTTPBearer
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from typing import List, Optional, Dict, Any
import logging
import json
import uuid
import os

from app.core.database import get_db
from app.models import Backtest, Trade, Model, User
from app.schemas import (
    BacktestCreate, BacktestInDB, BacktestListResponse, BacktestResponse,
    BacktestRequest, ResponseBase
)
from app.core.security import verify_token
from app.services.backtest_service import BacktestService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=BacktestListResponse)
async def get_backtests(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    status: Optional[str] = Query(None, description="回测状态"),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """获取回测列表"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        # 构建查询条件
        conditions = [Backtest.user_id == user_id]
        if status:
            conditions.append(Backtest.status == status)
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 查询总数
        count_stmt = select(Backtest).where(and_(*conditions))
        total_result = await db.execute(select(Backtest).where(and_(*conditions)))
        total = len(total_result.scalars().all())
        
        # 查询数据
        stmt = select(Backtest).where(and_(*conditions)).offset(offset).limit(page_size)
        result = await db.execute(stmt)
        backtests = result.scalars().all()
        
        return BacktestListResponse(
            success=True,
            message="获取回测列表成功",
            data=backtests,
            total=total,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        logger.error(f"获取回测列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取回测列表失败")


@router.get("/{backtest_id}", response_model=BacktestResponse)
async def get_backtest_detail(
    backtest_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """获取回测详情"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        result = await db.execute(
            select(Backtest).where(and_(Backtest.id == backtest_id, Backtest.user_id == user_id))
        )
        backtest = result.scalar_one_or_none()
        
        if not backtest:
            raise HTTPException(status_code=404, detail="回测不存在")
        
        return BacktestResponse(success=True, message="获取回测详情成功", data=backtest)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取回测详情失败: {e}")
        raise HTTPException(status_code=500, detail="获取回测详情失败")


@router.post("/", response_model=BacktestResponse)
async def create_backtest(
    backtest: BacktestCreate,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """创建回测"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        # 创建新回测
        db_backtest = Backtest(
            **backtest.dict(),
            user_id=user_id,
            status="pending"
        )
        
        db.add(db_backtest)
        await db.commit()
        await db.refresh(db_backtest)
        
        return BacktestResponse(success=True, message="创建回测成功", data=db_backtest)
    except Exception as e:
        logger.error(f"创建回测失败: {e}")
        raise HTTPException(status_code=500, detail="创建回测失败")


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(
    backtest_request: BacktestRequest,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """运行回测"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        # 创建回测记录
        backtest_name = f"{backtest_request.strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db_backtest = Backtest(
            name=backtest_name,
            description=f"策略类型: {backtest_request.strategy_type}",
            strategy_config=backtest_request.dict(),
            user_id=user_id,
            start_date=backtest_request.start_date,
            end_date=backtest_request.end_date,
            initial_cash=backtest_request.initial_cash,
            status="running",
            started_at=datetime.now()
        )
        
        db.add(db_backtest)
        await db.commit()
        await db.refresh(db_backtest)
        
        # 运行回测
        backtest_service = BacktestService(db)
        
        # 异步运行回测（这里简化处理，实际应该使用Celery）
        try:
            result = await backtest_service.run_backtest(
                backtest=db_backtest,
                strategy_type=backtest_request.strategy_type,
                symbols=backtest_request.symbols,
                start_date=backtest_request.start_date,
                end_date=backtest_request.end_date,
                initial_cash=backtest_request.initial_cash,
                parameters=backtest_request.parameters
            )
            
            # 更新回测结果
            db_backtest.status = "completed"
            db_backtest.completed_at = datetime.now()
            
            # 保存回测结果
            if result:
                db_backtest.total_return = result.get('total_return')
                db_backtest.annual_return = result.get('annual_return')
                db_backtest.max_drawdown = result.get('max_drawdown')
                db_backtest.sharpe_ratio = result.get('sharpe_ratio')
                db_backtest.volatility = result.get('volatility')
                db_backtest.win_rate = result.get('win_rate')
                db_backtest.profit_factor = result.get('profit_factor')
                db_backtest.total_trades = result.get('total_trades')
                db_backtest.winning_trades = result.get('winning_trades')
                db_backtest.losing_trades = result.get('losing_trades')
            
            await db.commit()
            
        except Exception as e:
            # 更新错误状态
            db_backtest.status = "failed"
            db_backtest.error_message = str(e)
            db_backtest.completed_at = datetime.now()
            await db.commit()
            
            raise HTTPException(status_code=500, detail=f"回测运行失败: {str(e)}")
        
        return BacktestResponse(success=True, message="回测运行成功", data=db_backtest)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"运行回测失败: {e}")
        raise HTTPException(status_code=500, detail="运行回测失败")


@router.get("/{backtest_id}/trades", response_model=ResponseBase)
async def get_backtest_trades(
    backtest_id: int,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(50, ge=1, le=200, description="每页数量"),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """获取回测交易记录"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        # 检查回测是否存在
        result = await db.execute(
            select(Backtest).where(and_(Backtest.id == backtest_id, Backtest.user_id == user_id))
        )
        backtest = result.scalar_one_or_none()
        
        if not backtest:
            raise HTTPException(status_code=404, detail="回测不存在")
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 查询交易记录
        stmt = select(Trade).where(Trade.backtest_id == backtest_id).offset(offset).limit(page_size)
        result = await db.execute(stmt)
        trades = result.scalars().all()
        
        # 查询总数
        count_stmt = select(Trade).where(Trade.backtest_id == backtest_id)
        total_result = await db.execute(count_stmt)
        total = len(total_result.scalars().all())
        
        return ResponseBase(
            success=True,
            message="获取交易记录成功",
            data={
                "trades": trades,
                "total": total,
                "page": page,
                "page_size": page_size
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取交易记录失败: {e}")
        raise HTTPException(status_code=500, detail="获取交易记录失败")


@router.get("/{backtest_id}/results", response_model=ResponseBase)
async def get_backtest_results(
    backtest_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """获取回测结果详情"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        # 获取回测详情
        result = await db.execute(
            select(Backtest).where(and_(Backtest.id == backtest_id, Backtest.user_id == user_id))
        )
        backtest = result.scalar_one_or_none()
        
        if not backtest:
            raise HTTPException(status_code=404, detail="回测不存在")
        
        # 获取交易记录
        trades_result = await db.execute(
            select(Trade).where(Trade.backtest_id == backtest_id).order_by(Trade.trade_date)
        )
        trades = trades_result.scalars().all()
        
        # 获取回测服务
        backtest_service = BacktestService(db)
        
        # 生成详细的回测结果
        detailed_results = await backtest_service.generate_detailed_results(backtest, trades)
        
        return ResponseBase(
            success=True,
            message="获取回测结果成功",
            data=detailed_results
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取回测结果失败: {e}")
        raise HTTPException(status_code=500, detail="获取回测结果失败")


@router.delete("/{backtest_id}", response_model=ResponseBase)
async def delete_backtest(
    backtest_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """删除回测"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        result = await db.execute(
            select(Backtest).where(and_(Backtest.id == backtest_id, Backtest.user_id == user_id))
        )
        backtest = result.scalar_one_or_none()
        
        if not backtest:
            raise HTTPException(status_code=404, detail="回测不存在")
        
        # 删除相关的交易记录
        await db.execute(
            select(Trade).where(Trade.backtest_id == backtest_id).delete()
        )
        
        # 删除回测记录
        await db.delete(backtest)
        await db.commit()
        
        return ResponseBase(success=True, message="删除回测成功")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除回测失败: {e}")
        raise HTTPException(status_code=500, detail="删除回测失败")