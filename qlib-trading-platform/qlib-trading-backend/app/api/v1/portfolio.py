from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from pydantic import BaseModel
import logging

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import User, Portfolio, PortfolioStock, Stock
from app.services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"])


# Pydantic模型
class PortfolioCreate(BaseModel):
    name: str
    description: Optional[str] = None
    initial_cash: float = 100000.0


class PortfolioUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class PortfolioStockAdd(BaseModel):
    symbol: str
    quantity: int
    average_cost: float


class PortfolioStockUpdate(BaseModel):
    quantity: Optional[int] = None
    average_cost: Optional[float] = None
    current_price: Optional[float] = None


class PortfolioResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    total_value: float
    cash: float
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class PortfolioStockResponse(BaseModel):
    id: int
    symbol: str
    name: str
    quantity: int
    average_cost: float
    current_price: Optional[float]
    market_value: Optional[float]
    profit_loss: Optional[float]
    profit_loss_percent: Optional[float]
    
    class Config:
        from_attributes = True


class PortfolioPerformance(BaseModel):
    total_value: float
    total_return: float
    total_return_percent: float
    cash: float
    stock_value: float
    positions_count: int
    
    class Config:
        from_attributes = True


# API端点
@router.post("/", response_model=PortfolioResponse)
async def create_portfolio(
    portfolio_data: PortfolioCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建投资组合"""
    try:
        portfolio_service = PortfolioService(db)
        portfolio = await portfolio_service.create_portfolio(
            user_id=current_user.id,
            name=portfolio_data.name,
            description=portfolio_data.description,
            initial_cash=portfolio_data.initial_cash
        )
        
        return portfolio
        
    except Exception as e:
        logger.error(f"创建投资组合失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建投资组合失败: {str(e)}"
        )


@router.get("/", response_model=List[PortfolioResponse])
async def get_portfolios(
    skip: int = 0,
    limit: int = 100,
    is_active: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取用户的投资组合列表"""
    try:
        query = select(Portfolio).where(Portfolio.user_id == current_user.id)
        
        if is_active is not None:
            query = query.where(Portfolio.is_active == is_active)
        
        query = query.offset(skip).limit(limit).order_by(Portfolio.created_at.desc())
        
        result = await db.execute(query)
        portfolios = result.scalars().all()
        
        return portfolios
        
    except Exception as e:
        logger.error(f"获取投资组合列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取投资组合列表失败: {str(e)}"
        )


@router.get("/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取投资组合详情"""
    try:
        result = await db.execute(
            select(Portfolio).where(
                and_(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id)
            )
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="投资组合不存在"
            )
        
        return portfolio
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取投资组合详情失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取投资组合详情失败: {str(e)}"
        )


@router.put("/{portfolio_id}", response_model=PortfolioResponse)
async def update_portfolio(
    portfolio_id: int,
    portfolio_data: PortfolioUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新投资组合"""
    try:
        result = await db.execute(
            select(Portfolio).where(
                and_(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id)
            )
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="投资组合不存在"
            )
        
        # 更新字段
        update_data = portfolio_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(portfolio, field, value)
        
        await db.commit()
        await db.refresh(portfolio)
        
        return portfolio
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新投资组合失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新投资组合失败: {str(e)}"
        )


@router.delete("/{portfolio_id}")
async def delete_portfolio(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除投资组合"""
    try:
        result = await db.execute(
            select(Portfolio).where(
                and_(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id)
            )
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="投资组合不存在"
            )
        
        await db.delete(portfolio)
        await db.commit()
        
        return {"message": "投资组合删除成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除投资组合失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除投资组合失败: {str(e)}"
        )


@router.get("/{portfolio_id}/performance", response_model=PortfolioPerformance)
async def get_portfolio_performance(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取投资组合表现"""
    try:
        portfolio_service = PortfolioService(db)
        performance = await portfolio_service.calculate_performance(portfolio_id, current_user.id)
        
        return performance
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"获取投资组合表现失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取投资组合表现失败: {str(e)}"
        )


@router.get("/{portfolio_id}/stocks", response_model=List[PortfolioStockResponse])
async def get_portfolio_stocks(
    portfolio_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取投资组合中的股票"""
    try:
        # 验证投资组合所有权
        result = await db.execute(
            select(Portfolio).where(
                and_(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id)
            )
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="投资组合不存在"
            )
        
        # 获取投资组合中的股票
        result = await db.execute(
            select(PortfolioStock, Stock).join(Stock).where(
                PortfolioStock.portfolio_id == portfolio_id
            ).order_by(PortfolioStock.created_at.desc())
        )
        
        portfolio_stocks = []
        for portfolio_stock, stock in result:
            portfolio_stocks.append({
                "id": portfolio_stock.id,
                "symbol": stock.symbol,
                "name": stock.name,
                "quantity": portfolio_stock.quantity,
                "average_cost": portfolio_stock.average_cost,
                "current_price": portfolio_stock.current_price,
                "market_value": portfolio_stock.market_value,
                "profit_loss": portfolio_stock.profit_loss,
                "profit_loss_percent": portfolio_stock.profit_loss_percent
            })
        
        return portfolio_stocks
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取投资组合股票失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取投资组合股票失败: {str(e)}"
        )


@router.post("/{portfolio_id}/stocks", response_model=dict)
async def add_portfolio_stock(
    portfolio_id: int,
    stock_data: PortfolioStockAdd,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """向投资组合添加股票"""
    try:
        portfolio_service = PortfolioService(db)
        result = await portfolio_service.add_stock_to_portfolio(
            portfolio_id=portfolio_id,
            user_id=current_user.id,
            symbol=stock_data.symbol,
            quantity=stock_data.quantity,
            average_cost=stock_data.average_cost
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"添加投资组合股票失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加投资组合股票失败: {str(e)}"
        )


@router.put("/{portfolio_id}/stocks/{stock_id}", response_model=dict)
async def update_portfolio_stock(
    portfolio_id: int,
    stock_id: int,
    stock_data: PortfolioStockUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新投资组合中的股票"""
    try:
        portfolio_service = PortfolioService(db)
        result = await portfolio_service.update_portfolio_stock(
            portfolio_id=portfolio_id,
            stock_id=stock_id,
            user_id=current_user.id,
            update_data=stock_data.dict(exclude_unset=True)
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"更新投资组合股票失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新投资组合股票失败: {str(e)}"
        )


@router.delete("/{portfolio_id}/stocks/{stock_id}")
async def remove_portfolio_stock(
    portfolio_id: int,
    stock_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """从投资组合中移除股票"""
    try:
        portfolio_service = PortfolioService(db)
        result = await portfolio_service.remove_stock_from_portfolio(
            portfolio_id=portfolio_id,
            stock_id=stock_id,
            user_id=current_user.id
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"移除投资组合股票失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"移除投资组合股票失败: {str(e)}"
        )


@router.post("/{portfolio_id}/rebalance")
async def rebalance_portfolio(
    portfolio_id: int,
    target_weights: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """重新平衡投资组合"""
    try:
        portfolio_service = PortfolioService(db)
        result = await portfolio_service.rebalance_portfolio(
            portfolio_id=portfolio_id,
            user_id=current_user.id,
            target_weights=target_weights
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"重新平衡投资组合失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重新平衡投资组合失败: {str(e)}"
        )