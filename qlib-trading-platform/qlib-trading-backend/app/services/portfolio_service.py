from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
import logging

from app.models import Portfolio, PortfolioStock, Stock, StockPrice
from app.services.stock_service import StockService

logger = logging.getLogger(__name__)


class PortfolioService:
    """投资组合服务"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.stock_service = StockService(db)
    
    async def create_portfolio(
        self,
        user_id: int,
        name: str,
        description: Optional[str] = None,
        initial_cash: float = 100000.0
    ) -> Portfolio:
        """创建投资组合"""
        try:
            portfolio = Portfolio(
                name=name,
                description=description,
                user_id=user_id,
                cash=initial_cash,
                total_value=initial_cash
            )
            
            self.db.add(portfolio)
            await self.db.commit()
            await self.db.refresh(portfolio)
            
            logger.info(f"创建投资组合成功: {name} (ID: {portfolio.id})")
            return portfolio
            
        except Exception as e:
            logger.error(f"创建投资组合失败: {e}")
            await self.db.rollback()
            raise
    
    async def calculate_performance(self, portfolio_id: int, user_id: int) -> Dict[str, Any]:
        """计算投资组合表现"""
        try:
            # 验证投资组合所有权
            result = await self.db.execute(
                select(Portfolio).where(
                    and_(Portfolio.id == portfolio_id, Portfolio.user_id == user_id)
                )
            )
            portfolio = result.scalar_one_or_none()
            
            if not portfolio:
                raise ValueError("投资组合不存在或无权限访问")
            
            # 获取投资组合中的股票
            result = await self.db.execute(
                select(PortfolioStock, Stock).join(Stock).where(
                    PortfolioStock.portfolio_id == portfolio_id
                )
            )
            
            stock_value = 0.0
            positions_count = 0
            
            # 更新股票当前价格
            for portfolio_stock, stock in result:
                try:
                    # 获取最新价格
                    price_result = await self.db.execute(
                        select(StockPrice).where(
                            StockPrice.stock_id == stock.id
                        ).order_by(StockPrice.date.desc()).limit(1)
                    )
                    latest_price = price_result.scalar_one_or_none()
                    
                    if latest_price:
                        current_price = latest_price.close
                        market_value = portfolio_stock.quantity * current_price
                        profit_loss = market_value - (portfolio_stock.quantity * portfolio_stock.average_cost)
                        profit_loss_percent = (profit_loss / (portfolio_stock.quantity * portfolio_stock.average_cost)) * 100 if portfolio_stock.average_cost > 0 else 0
                        
                        # 更新投资组合股票信息
                        portfolio_stock.current_price = current_price
                        portfolio_stock.market_value = market_value
                        portfolio_stock.profit_loss = profit_loss
                        portfolio_stock.profit_loss_percent = profit_loss_percent
                        
                        stock_value += market_value
                        positions_count += 1
                        
                        self.db.add(portfolio_stock)
                
                except Exception as e:
                    logger.error(f"更新股票 {stock.symbol} 价格失败: {e}")
                    continue
            
            # 更新投资组合总价值
            total_value = portfolio.cash + stock_value
            total_return = total_value - portfolio.cash  # 简化计算，假设初始只有现金
            total_return_percent = (total_return / portfolio.cash) * 100 if portfolio.cash > 0 else 0
            
            portfolio.total_value = total_value
            self.db.add(portfolio)
            
            await self.db.commit()
            
            return {
                "total_value": total_value,
                "total_return": total_return,
                "total_return_percent": total_return_percent,
                "cash": portfolio.cash,
                "stock_value": stock_value,
                "positions_count": positions_count
            }
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"计算投资组合表现失败: {e}")
            raise
    
    async def add_stock_to_portfolio(
        self,
        portfolio_id: int,
        user_id: int,
        symbol: str,
        quantity: int,
        average_cost: float
    ) -> Dict[str, Any]:
        """向投资组合添加股票"""
        try:
            # 验证投资组合所有权
            result = await self.db.execute(
                select(Portfolio).where(
                    and_(Portfolio.id == portfolio_id, Portfolio.user_id == user_id)
                )
            )
            portfolio = result.scalar_one_or_none()
            
            if not portfolio:
                raise ValueError("投资组合不存在或无权限访问")
            
            # 检查现金是否足够
            total_cost = quantity * average_cost
            if portfolio.cash < total_cost:
                raise ValueError("现金余额不足")
            
            # 获取股票信息
            result = await self.db.execute(
                select(Stock).where(Stock.symbol == symbol)
            )
            stock = result.scalar_one_or_none()
            
            if not stock:
                raise ValueError(f"股票 {symbol} 不存在")
            
            # 检查是否已存在该股票
            result = await self.db.execute(
                select(PortfolioStock).where(
                    and_(PortfolioStock.portfolio_id == portfolio_id, PortfolioStock.stock_id == stock.id)
                )
            )
            existing_stock = result.scalar_one_or_none()
            
            if existing_stock:
                # 更新现有持仓
                total_quantity = existing_stock.quantity + quantity
                total_cost = (existing_stock.quantity * existing_stock.average_cost) + (quantity * average_cost)
                existing_stock.quantity = total_quantity
                existing_stock.average_cost = total_cost / total_quantity
                
                self.db.add(existing_stock)
                portfolio_stock = existing_stock
            else:
                # 创建新持仓
                portfolio_stock = PortfolioStock(
                    portfolio_id=portfolio_id,
                    stock_id=stock.id,
                    quantity=quantity,
                    average_cost=average_cost
                )
                self.db.add(portfolio_stock)
            
            # 更新现金余额
            portfolio.cash -= total_cost
            self.db.add(portfolio)
            
            await self.db.commit()
            await self.db.refresh(portfolio_stock)
            
            return {
                "message": "股票添加成功",
                "portfolio_stock_id": portfolio_stock.id,
                "symbol": symbol,
                "quantity": portfolio_stock.quantity,
                "average_cost": portfolio_stock.average_cost,
                "remaining_cash": portfolio.cash
            }
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"添加股票到投资组合失败: {e}")
            await self.db.rollback()
            raise
    
    async def update_portfolio_stock(
        self,
        portfolio_id: int,
        stock_id: int,
        user_id: int,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新投资组合中的股票"""
        try:
            # 验证投资组合所有权
            result = await self.db.execute(
                select(Portfolio).where(
                    and_(Portfolio.id == portfolio_id, Portfolio.user_id == user_id)
                )
            )
            portfolio = result.scalar_one_or_none()
            
            if not portfolio:
                raise ValueError("投资组合不存在或无权限访问")
            
            # 获取投资组合股票
            result = await self.db.execute(
                select(PortfolioStock, Stock).join(Stock).where(
                    and_(PortfolioStock.id == stock_id, PortfolioStock.portfolio_id == portfolio_id)
                )
            )
            portfolio_stock_data = result.first()
            
            if not portfolio_stock_data:
                raise ValueError("投资组合中未找到该股票")
            
            portfolio_stock, stock = portfolio_stock_data
            
            # 更新字段
            for field, value in update_data.items():
                if hasattr(portfolio_stock, field) and value is not None:
                    setattr(portfolio_stock, field, value)
            
            # 重新计算市值和盈亏
            if portfolio_stock.current_price and portfolio_stock.quantity:
                portfolio_stock.market_value = portfolio_stock.quantity * portfolio_stock.current_price
                portfolio_stock.profit_loss = portfolio_stock.market_value - (portfolio_stock.quantity * portfolio_stock.average_cost)
                portfolio_stock.profit_loss_percent = (portfolio_stock.profit_loss / (portfolio_stock.quantity * portfolio_stock.average_cost)) * 100 if portfolio_stock.average_cost > 0 else 0
            
            self.db.add(portfolio_stock)
            await self.db.commit()
            await self.db.refresh(portfolio_stock)
            
            return {
                "message": "股票更新成功",
                "portfolio_stock_id": portfolio_stock.id,
                "symbol": stock.symbol,
                "quantity": portfolio_stock.quantity,
                "average_cost": portfolio_stock.average_cost,
                "current_price": portfolio_stock.current_price,
                "market_value": portfolio_stock.market_value,
                "profit_loss": portfolio_stock.profit_loss,
                "profit_loss_percent": portfolio_stock.profit_loss_percent
            }
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"更新投资组合股票失败: {e}")
            await self.db.rollback()
            raise
    
    async def remove_stock_from_portfolio(
        self,
        portfolio_id: int,
        stock_id: int,
        user_id: int
    ) -> Dict[str, Any]:
        """从投资组合中移除股票"""
        try:
            # 验证投资组合所有权
            result = await self.db.execute(
                select(Portfolio).where(
                    and_(Portfolio.id == portfolio_id, Portfolio.user_id == user_id)
                )
            )
            portfolio = result.scalar_one_or_none()
            
            if not portfolio:
                raise ValueError("投资组合不存在或无权限访问")
            
            # 获取投资组合股票
            result = await self.db.execute(
                select(PortfolioStock, Stock).join(Stock).where(
                    and_(PortfolioStock.id == stock_id, PortfolioStock.portfolio_id == portfolio_id)
                )
            )
            portfolio_stock_data = result.first()
            
            if not portfolio_stock_data:
                raise ValueError("投资组合中未找到该股票")
            
            portfolio_stock, stock = portfolio_stock_data
            
            # 计算卖出价值（使用当前价格或平均成本）
            sell_price = portfolio_stock.current_price or portfolio_stock.average_cost
            sell_value = portfolio_stock.quantity * sell_price
            
            # 更新现金余额
            portfolio.cash += sell_value
            self.db.add(portfolio)
            
            # 删除股票记录
            await self.db.delete(portfolio_stock)
            await self.db.commit()
            
            return {
                "message": "股票移除成功",
                "symbol": stock.symbol,
                "quantity": portfolio_stock.quantity,
                "sell_value": sell_value,
                "remaining_cash": portfolio.cash
            }
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"从投资组合移除股票失败: {e}")
            await self.db.rollback()
            raise
    
    async def rebalance_portfolio(
        self,
        portfolio_id: int,
        user_id: int,
        target_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """重新平衡投资组合"""
        try:
            # 验证投资组合所有权
            result = await self.db.execute(
                select(Portfolio).where(
                    and_(Portfolio.id == portfolio_id, Portfolio.user_id == user_id)
                )
            )
            portfolio = result.scalar_one_or_none()
            
            if not portfolio:
                raise ValueError("投资组合不存在或无权限访问")
            
            # 获取当前持仓
            result = await self.db.execute(
                select(PortfolioStock, Stock).join(Stock).where(
                    PortfolioStock.portfolio_id == portfolio_id
                )
            )
            current_positions = result.all()
            
            # 获取当前价格
            total_value = portfolio.cash
            current_weights = {}
            
            for portfolio_stock, stock in current_positions:
                # 获取最新价格
                price_result = await self.db.execute(
                    select(StockPrice).where(
                        StockPrice.stock_id == stock.id
                    ).order_by(StockPrice.date.desc()).limit(1)
                )
                latest_price = price_result.scalar_one_or_none()
                
                if latest_price:
                    current_price = latest_price.close
                    market_value = portfolio_stock.quantity * current_price
                    total_value += market_value
                    current_weights[stock.symbol] = market_value
                    
                    # 更新当前价格
                    portfolio_stock.current_price = current_price
                    self.db.add(portfolio_stock)
            
            # 计算当前权重
            for symbol in current_weights:
                current_weights[symbol] = current_weights[symbol] / total_value
            
            # 计算需要调整的股票
            trades = []
            
            # 处理需要卖出的股票
            for symbol, current_weight in current_weights.items():
                if symbol not in target_weights or target_weights[symbol] < current_weight:
                    # 需要卖出
                    target_weight = target_weights.get(symbol, 0)
                    sell_weight = current_weight - target_weight
                    sell_value = sell_weight * total_value
                    
                    # 找到对应的持仓
                    for portfolio_stock, stock in current_positions:
                        if stock.symbol == symbol:
                            current_price = portfolio_stock.current_price or portfolio_stock.average_cost
                            sell_quantity = int(sell_value / current_price)
                            
                            if sell_quantity > 0:
                                # 确保不会卖出超过持仓数量
                                sell_quantity = min(sell_quantity, portfolio_stock.quantity)
                                
                                # 更新持仓
                                portfolio_stock.quantity -= sell_quantity
                                sell_value_actual = sell_quantity * current_price
                                
                                if portfolio_stock.quantity == 0:
                                    # 删除持仓
                                    await self.db.delete(portfolio_stock)
                                else:
                                    self.db.add(portfolio_stock)
                                
                                # 更新现金
                                portfolio.cash += sell_value_actual
                                
                                trades.append({
                                    "symbol": symbol,
                                    "action": "sell",
                                    "quantity": sell_quantity,
                                    "price": current_price,
                                    "value": sell_value_actual
                                })
                                
                                logger.info(f"卖出 {symbol}: {sell_quantity} 股 @ {current_price}")
                                break
            
            # 处理需要买入的股票
            for symbol, target_weight in target_weights.items():
                current_weight = current_weights.get(symbol, 0)
                if target_weight > current_weight:
                    # 需要买入
                    buy_weight = target_weight - current_weight
                    buy_value = buy_weight * total_value
                    
                    # 检查现金是否足够
                    if portfolio.cash >= buy_value:
                        # 获取股票信息
                        result = await self.db.execute(
                            select(Stock).where(Stock.symbol == symbol)
                        )
                        stock = result.scalar_one_or_none()
                        
                        if stock:
                            # 获取当前价格
                            price_result = await self.db.execute(
                                select(StockPrice).where(
                                    StockPrice.stock_id == stock.id
                                ).order_by(StockPrice.date.desc()).limit(1)
                            )
                            latest_price = price_result.scalar_one_or_none()
                            
                            if latest_price:
                                current_price = latest_price.close
                                buy_quantity = int(buy_value / current_price)
                                
                                if buy_quantity > 0:
                                    # 检查是否已存在该股票
                                    existing_position = None
                                    for portfolio_stock, _ in current_positions:
                                        if portfolio_stock.stock_id == stock.id:
                                            existing_position = portfolio_stock
                                            break
                                    
                                    if existing_position:
                                        # 更新现有持仓
                                        total_quantity = existing_position.quantity + buy_quantity
                                        total_cost = (existing_position.quantity * existing_position.average_cost) + (buy_quantity * current_price)
                                        existing_position.quantity = total_quantity
                                        existing_position.average_cost = total_cost / total_quantity
                                        existing_position.current_price = current_price
                                        
                                        self.db.add(existing_position)
                                    else:
                                        # 创建新持仓
                                        new_position = PortfolioStock(
                                            portfolio_id=portfolio_id,
                                            stock_id=stock.id,
                                            quantity=buy_quantity,
                                            average_cost=current_price,
                                            current_price=current_price
                                        )
                                        self.db.add(new_position)
                                    
                                    # 更新现金
                                    portfolio.cash -= (buy_quantity * current_price)
                                    
                                    trades.append({
                                        "symbol": symbol,
                                        "action": "buy",
                                        "quantity": buy_quantity,
                                        "price": current_price,
                                        "value": buy_quantity * current_price
                                    })
                                    
                                    logger.info(f"买入 {symbol}: {buy_quantity} 股 @ {current_price}")
            
            # 更新投资组合总价值
            await self.calculate_performance(portfolio_id, user_id)
            
            await self.db.commit()
            
            return {
                "message": "投资组合重新平衡成功",
                "trades": trades,
                "remaining_cash": portfolio.cash,
                "target_weights": target_weights,
                "current_weights": current_weights
            }
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"重新平衡投资组合失败: {e}")
            await self.db.rollback()
            raise