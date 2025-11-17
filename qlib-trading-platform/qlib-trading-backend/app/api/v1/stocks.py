from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from typing import List, Optional
import logging

from app.core.database import get_db
from app.models import Stock, StockPrice
from app.schemas import (
    StockCreate, StockInDB, StockListResponse, StockResponse,
    StockDataRequest, StockDataResponse, ResponseBase
)
from app.services.stock_service import StockService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/search", response_model=StockListResponse)
async def search_stocks(
    query: str = Query(..., description="搜索关键词"),
    market: Optional[str] = Query(None, description="市场代码"),
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    db: AsyncSession = Depends(get_db)
):
    """搜索股票"""
    try:
        # 构建查询条件
        conditions = []
        if query:
            conditions.append(
                or_(
                    Stock.symbol.contains(query),
                    Stock.name.contains(query)
                )
            )
        if market:
            conditions.append(Stock.market == market)
        
        # 执行查询
        stmt = select(Stock).where(and_(*conditions)).limit(limit)
        result = await db.execute(stmt)
        stocks = result.scalars().all()
        
        return StockListResponse(
            success=True,
            message="搜索成功",
            data=stocks,
            total=len(stocks),
            page=1,
            page_size=limit
        )
    except Exception as e:
        logger.error(f"搜索股票失败: {e}")
        raise HTTPException(status_code=500, detail="搜索股票失败")


@router.get("/", response_model=StockListResponse)
async def get_stocks(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    market: Optional[str] = Query(None, description="市场代码"),
    is_active: Optional[bool] = Query(None, description="是否活跃"),
    db: AsyncSession = Depends(get_db)
):
    """获取股票列表"""
    try:
        # 构建查询条件
        conditions = []
        if market:
            conditions.append(Stock.market == market)
        if is_active is not None:
            conditions.append(Stock.is_active == is_active)
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 查询总数
        count_stmt = select(Stock).where(and_(*conditions))
        total_result = await db.execute(select(Stock).where(and_(*conditions)))
        total = len(total_result.scalars().all())
        
        # 查询数据
        stmt = select(Stock).where(and_(*conditions)).offset(offset).limit(page_size)
        result = await db.execute(stmt)
        stocks = result.scalars().all()
        
        return StockListResponse(
            success=True,
            message="获取股票列表成功",
            data=stocks,
            total=total,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取股票列表失败")


@router.get("/{symbol}", response_model=StockResponse)
async def get_stock_detail(
    symbol: str,
    db: AsyncSession = Depends(get_db)
):
    """获取股票详情"""
    try:
        result = await db.execute(select(Stock).where(Stock.symbol == symbol))
        stock = result.scalar_one_or_none()
        
        if not stock:
            raise HTTPException(status_code=404, detail="股票不存在")
        
        return StockResponse(success=True, message="获取股票详情成功", data=stock)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取股票详情失败: {e}")
        raise HTTPException(status_code=500, detail="获取股票详情失败")


@router.post("/", response_model=StockResponse)
async def create_stock(
    stock: StockCreate,
    db: AsyncSession = Depends(get_db)
):
    """创建股票"""
    try:
        # 检查股票是否已存在
        result = await db.execute(select(Stock).where(Stock.symbol == stock.symbol))
        if result.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="股票已存在")
        
        # 创建新股票
        db_stock = Stock(**stock.dict())
        db.add(db_stock)
        await db.commit()
        await db.refresh(db_stock)
        
        return StockResponse(success=True, message="创建股票成功", data=db_stock)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建股票失败: {e}")
        raise HTTPException(status_code=500, detail="创建股票失败")


@router.put("/{symbol}", response_model=StockResponse)
async def update_stock(
    symbol: str,
    stock_update: StockCreate,
    db: AsyncSession = Depends(get_db)
):
    """更新股票信息"""
    try:
        result = await db.execute(select(Stock).where(Stock.symbol == symbol))
        stock = result.scalar_one_or_none()
        
        if not stock:
            raise HTTPException(status_code=404, detail="股票不存在")
        
        # 更新字段
        for field, value in stock_update.dict(exclude_unset=True).items():
            setattr(stock, field, value)
        
        await db.commit()
        await db.refresh(stock)
        
        return StockResponse(success=True, message="更新股票成功", data=stock)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新股票失败: {e}")
        raise HTTPException(status_code=500, detail="更新股票失败")


@router.delete("/{symbol}", response_model=ResponseBase)
async def delete_stock(
    symbol: str,
    db: AsyncSession = Depends(get_db)
):
    """删除股票"""
    try:
        result = await db.execute(select(Stock).where(Stock.symbol == symbol))
        stock = result.scalar_one_or_none()
        
        if not stock:
            raise HTTPException(status_code=404, detail="股票不存在")
        
        await db.delete(stock)
        await db.commit()
        
        return ResponseBase(success=True, message="删除股票成功")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除股票失败: {e}")
        raise HTTPException(status_code=500, detail="删除股票失败")


@router.get("/{symbol}/data", response_model=StockDataResponse)
async def get_stock_data(
    symbol: str,
    start_date: Optional[str] = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="结束日期 (YYYY-MM-DD)"),
    period: str = Query("1d", description="数据周期: 1d, 1w, 1m, 1y"),
    indicators: Optional[List[str]] = Query(None, description="技术指标: ma5, ma20, rsi, macd"),
    db: AsyncSession = Depends(get_db)
):
    """获取股票历史数据"""
    try:
        # 获取股票信息
        result = await db.execute(select(Stock).where(Stock.symbol == symbol))
        stock = result.scalar_one_or_none()
        
        if not stock:
            raise HTTPException(status_code=404, detail="股票不存在")
        
        # 获取股票数据
        stock_service = StockService(db)
        data = await stock_service.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            period=period,
            indicators=indicators
        )
        
        return StockDataResponse(success=True, message="获取股票数据成功", data=data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取股票数据失败: {e}")
        raise HTTPException(status_code=500, detail="获取股票数据失败")


@router.get("/{symbol}/indicators")
async def get_stock_indicators(
    symbol: str,
    indicators: List[str] = Query(..., description="技术指标列表"),
    period: str = Query("1d", description="数据周期"),
    db: AsyncSession = Depends(get_db)
):
    """获取股票技术指标"""
    try:
        # 检查股票是否存在
        result = await db.execute(select(Stock).where(Stock.symbol == symbol))
        stock = result.scalar_one_or_none()
        
        if not stock:
            raise HTTPException(status_code=404, detail="股票不存在")
        
        # 计算技术指标
        stock_service = StockService(db)
        indicator_data = await stock_service.calculate_indicators(
            symbol=symbol,
            indicators=indicators,
            period=period
        )
        
        return ResponseBase(success=True, message="获取技术指标成功", data=indicator_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取技术指标失败: {e}")
        raise HTTPException(status_code=500, detail="获取技术指标失败")


@router.post("/{symbol}/sync")
async def sync_stock_data(
    symbol: str,
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    db: AsyncSession = Depends(get_db)
):
    """同步股票数据"""
    try:
        # 检查股票是否存在
        result = await db.execute(select(Stock).where(Stock.symbol == symbol))
        stock = result.scalar_one_or_none()
        
        if not stock:
            raise HTTPException(status_code=404, detail="股票不存在")
        
        # 同步数据
        stock_service = StockService(db)
        sync_result = await stock_service.sync_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        return ResponseBase(success=True, message="数据同步成功", data=sync_result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"同步股票数据失败: {e}")
        raise HTTPException(status_code=500, detail="同步股票数据失败")