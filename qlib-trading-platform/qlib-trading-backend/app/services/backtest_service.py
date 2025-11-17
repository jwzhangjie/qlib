import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import json
import os
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models import Backtest, Trade, Stock, StockPrice
from app.core.config import settings

logger = logging.getLogger(__name__)


class BacktestService:
    """回测服务"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def run_backtest(
        self,
        backtest: Backtest,
        strategy_type: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_cash: float,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """运行回测"""
        try:
            # 获取回测数据
            backtest_data = await self._get_backtest_data(symbols, start_date, end_date)
            
            if backtest_data.empty:
                raise ValueError("没有可用的回测数据")
            
            # 根据策略类型运行回测
            if strategy_type == 'buy_and_hold':
                result = await self._run_buy_and_hold_strategy(
                    backtest_data, initial_cash, parameters
                )
            elif strategy_type == 'momentum':
                result = await self._run_momentum_strategy(
                    backtest_data, initial_cash, parameters
                )
            elif strategy_type == 'mean_reversion':
                result = await self._run_mean_reversion_strategy(
                    backtest_data, initial_cash, parameters
                )
            elif strategy_type == 'ml_based':
                result = await self._run_ml_based_strategy(
                    backtest_data, initial_cash, parameters
                )
            else:
                raise ValueError(f"不支持的策略类型: {strategy_type}")
            
            # 保存交易记录
            if 'trades' in result:
                await self._save_trades(backtest.id, result['trades'])
            
            return result
            
        except Exception as e:
            logger.error(f"运行回测失败: {e}")
            raise
    
    async def _get_backtest_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """获取回测数据"""
        try:
            # 查询股票数据
            result = await self.db.execute(
                select(StockPrice).join(Stock).where(
                    and_(
                        Stock.symbol.in_(symbols),
                        StockPrice.date >= start_date,
                        StockPrice.date <= end_date
                    )
                ).order_by(StockPrice.date)
            )
            prices = result.scalars().all()
            
            if not prices:
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame([
                {
                    'symbol': price.stock.symbol,
                    'date': price.date,
                    'open': price.open,
                    'high': price.high,
                    'low': price.low,
                    'close': price.close,
                    'volume': price.volume,
                    'adj_close': price.adj_close
                }
                for price in prices
            ])
            
            # 设置索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"获取回测数据失败: {e}")
            raise
    
    async def _run_buy_and_hold_strategy(
        self,
        data: pd.DataFrame,
        initial_cash: float,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """运行买入持有策略"""
        try:
            # 获取唯一股票代码
            symbols = data['symbol'].unique()
            
            # 计算每只股票的权重
            weight = 1.0 / len(symbols)
            
            # 初始化投资组合
            portfolio = {
                'cash': initial_cash,
                'positions': {},
                'values': []
            }
            
            trades = []
            daily_values = []
            
            # 第一天买入所有股票
            first_date = data.index.min()
            first_day_data = data[data.index == first_date]
            
            for symbol in symbols:
                symbol_data = first_day_data[first_day_data['symbol'] == symbol]
                if not symbol_data.empty:
                    price = symbol_data['close'].iloc[0]
                    cash_per_stock = initial_cash * weight
                    shares = int(cash_per_stock / price)
                    
                    if shares > 0:
                        cost = shares * price
                        portfolio['cash'] -= cost
                        portfolio['positions'][symbol] = shares
                        
                        trades.append({
                            'date': first_date,
                            'symbol': symbol,
                            'action': 'buy',
                            'quantity': shares,
                            'price': price,
                            'value': cost
                        })
            
            # 计算每日组合价值
            for date in data.index.unique():
                day_data = data[data.index == date]
                portfolio_value = portfolio['cash']
                
                for symbol, shares in portfolio['positions'].items():
                    symbol_data = day_data[day_data['symbol'] == symbol]
                    if not symbol_data.empty:
                        price = symbol_data['close'].iloc[0]
                        portfolio_value += shares * price
                
                daily_values.append({
                    'date': date,
                    'portfolio_value': portfolio_value
                })
            
            # 计算回测指标
            daily_values_df = pd.DataFrame(daily_values)
            daily_values_df['returns'] = daily_values_df['portfolio_value'].pct_change()
            
            total_return = (daily_values_df['portfolio_value'].iloc[-1] - initial_cash) / initial_cash
            annual_return = (1 + total_return) ** (252 / len(daily_values_df)) - 1
            volatility = daily_values_df['returns'].std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # 计算最大回撤
            cumulative_returns = (1 + daily_values_df['returns'].fillna(0)).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                'strategy_type': 'buy_and_hold',
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(trades),
                'trades': trades,
                'daily_values': daily_values
            }
            
        except Exception as e:
            logger.error(f"运行买入持有策略失败: {e}")
            raise
    
    async def _run_momentum_strategy(
        self,
        data: pd.DataFrame,
        initial_cash: float,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """运行动量策略"""
        try:
            # 默认参数
            lookback_period = parameters.get('lookback_period', 20) if parameters else 20
            holding_period = parameters.get('holding_period', 5) if parameters else 5
            top_n = parameters.get('top_n', 5) if parameters else 5
            
            # 获取唯一股票代码
            symbols = data['symbol'].unique()
            
            # 初始化投资组合
            portfolio = {
                'cash': initial_cash,
                'positions': {},
                'last_rebalance': None
            }
            
            trades = []
            daily_values = []
            
            # 按日期循环
            unique_dates = sorted(data.index.unique())
            
            for i, date in enumerate(unique_dates):
                day_data = data[data.index == date]
                
                # 计算动量（过去lookback_period天的收益率）
                momentum_scores = {}
                
                for symbol in symbols:
                    symbol_data = data[data['symbol'] == symbol]
                    symbol_day_data = day_data[day_data['symbol'] == symbol]
                    
                    if not symbol_day_data.empty and i >= lookback_period:
                        # 获取过去lookback_period天的数据
                        start_idx = max(0, i - lookback_period)
                        past_data = symbol_data.iloc[start_idx:i]
                        
                        if len(past_data) >= lookback_period:
                            past_return = (past_data['close'].iloc[-1] / past_data['close'].iloc[0]) - 1
                            momentum_scores[symbol] = past_return
                
                # 选择动量最高的股票
                if momentum_scores:
                    top_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
                    selected_symbols = [item[0] for item in top_symbols]
                    
                    # 检查是否需要重新平衡
                    if (portfolio['last_rebalance'] is None or 
                        (date - portfolio['last_rebalance']).days >= holding_period):
                        
                        # 卖出所有持仓
                        for symbol, shares in list(portfolio['positions'].items()):
                            if symbol not in selected_symbols:
                                symbol_data = day_data[day_data['symbol'] == symbol]
                                if not symbol_data.empty:
                                    price = symbol_data['close'].iloc[0]
                                    value = shares * price
                                    portfolio['cash'] += value
                                    
                                    trades.append({
                                        'date': date,
                                        'symbol': symbol,
                                        'action': 'sell',
                                        'quantity': shares,
                                        'price': price,
                                        'value': value
                                    })
                                    
                                    del portfolio['positions'][symbol]
                        
                        # 买入选中的股票
                        cash_per_stock = portfolio['cash'] / len(selected_symbols) if selected_symbols else 0
                        
                        for symbol in selected_symbols:
                            symbol_data = day_data[day_data['symbol'] == symbol]
                            if not symbol_data.empty:
                                price = symbol_data['close'].iloc[0]
                                shares = int(cash_per_stock / price)
                                
                                if shares > 0:
                                    cost = shares * price
                                    portfolio['cash'] -= cost
                                    portfolio['positions'][symbol] = shares
                                    
                                    trades.append({
                                        'date': date,
                                        'symbol': symbol,
                                        'action': 'buy',
                                        'quantity': shares,
                                        'price': price,
                                        'value': cost
                                    })
                        
                        portfolio['last_rebalance'] = date
                
                # 计算组合价值
                portfolio_value = portfolio['cash']
                for symbol, shares in portfolio['positions'].items():
                    symbol_data = day_data[day_data['symbol'] == symbol]
                    if not symbol_data.empty:
                        price = symbol_data['close'].iloc[0]
                        portfolio_value += shares * price
                
                daily_values.append({
                    'date': date,
                    'portfolio_value': portfolio_value
                })
            
            # 计算回测指标
            daily_values_df = pd.DataFrame(daily_values)
            daily_values_df['returns'] = daily_values_df['portfolio_value'].pct_change()
            
            total_return = (daily_values_df['portfolio_value'].iloc[-1] - initial_cash) / initial_cash
            annual_return = (1 + total_return) ** (252 / len(daily_values_df)) - 1
            volatility = daily_values_df['returns'].std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # 计算最大回撤
            cumulative_returns = (1 + daily_values_df['returns'].fillna(0)).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 计算胜率
            winning_trades = sum(1 for trade in trades if trade['action'] == 'sell')
            total_trades = len(trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            return {
                'strategy_type': 'momentum',
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'trades': trades,
                'daily_values': daily_values
            }
            
        except Exception as e:
            logger.error(f"运行动量策略失败: {e}")
            raise
    
    async def _run_mean_reversion_strategy(
        self,
        data: pd.DataFrame,
        initial_cash: float,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """运行均值回归策略"""
        try:
            # 默认参数
            lookback_period = parameters.get('lookback_period', 20) if parameters else 20
            z_score_threshold = parameters.get('z_score_threshold', 2.0) if parameters else 2.0
            
            # 获取唯一股票代码
            symbols = data['symbol'].unique()
            
            # 初始化投资组合
            portfolio = {
                'cash': initial_cash,
                'positions': {}
            }
            
            trades = []
            daily_values = []
            
            # 按日期循环
            unique_dates = sorted(data.index.unique())
            
            for i, date in enumerate(unique_dates):
                day_data = data[data.index == date]
                
                for symbol in symbols:
                    symbol_data = data[data['symbol'] == symbol]
                    symbol_day_data = day_data[day_data['symbol'] == symbol]
                    
                    if not symbol_day_data.empty and i >= lookback_period:
                        # 获取过去lookback_period天的数据
                        start_idx = max(0, i - lookback_period)
                        past_data = symbol_data.iloc[start_idx:i]
                        
                        if len(past_data) >= lookback_period:
                            # 计算均值和标准差
                            mean_price = past_data['close'].mean()
                            std_price = past_data['close'].std()
                            
                            current_price = symbol_day_data['close'].iloc[0]
                            z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
                            
                            # 当前持仓
                            current_position = portfolio['positions'].get(symbol, 0)
                            
                            # 均值回归信号
                            if z_score < -z_score_threshold and current_position == 0:
                                # 价格低于均值，买入
                                cash_available = portfolio['cash']
                                shares = int(cash_available / current_price * 0.9)  # 使用90%的现金
                                
                                if shares > 0:
                                    cost = shares * current_price
                                    portfolio['cash'] -= cost
                                    portfolio['positions'][symbol] = shares
                                    
                                    trades.append({
                                        'date': date,
                                        'symbol': symbol,
                                        'action': 'buy',
                                        'quantity': shares,
                                        'price': current_price,
                                        'value': cost
                                    })
                            
                            elif z_score > z_score_threshold and current_position > 0:
                                # 价格高于均值，卖出
                                shares = current_position
                                value = shares * current_price
                                portfolio['cash'] += value
                                
                                trades.append({
                                    'date': date,
                                    'symbol': symbol,
                                    'action': 'sell',
                                    'quantity': shares,
                                    'price': current_price,
                                    'value': value
                                })
                                
                                del portfolio['positions'][symbol]
                
                # 计算组合价值
                portfolio_value = portfolio['cash']
                for symbol, shares in portfolio['positions'].items():
                    symbol_data = day_data[day_data['symbol'] == symbol]
                    if not symbol_data.empty:
                        price = symbol_data['close'].iloc[0]
                        portfolio_value += shares * price
                
                daily_values.append({
                    'date': date,
                    'portfolio_value': portfolio_value
                })
            
            # 计算回测指标
            daily_values_df = pd.DataFrame(daily_values)
            daily_values_df['returns'] = daily_values_df['portfolio_value'].pct_change()
            
            total_return = (daily_values_df['portfolio_value'].iloc[-1] - initial_cash) / initial_cash
            annual_return = (1 + total_return) ** (252 / len(daily_values_df)) - 1
            volatility = daily_values_df['returns'].std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # 计算最大回撤
            cumulative_returns = (1 + daily_values_df['returns'].fillna(0)).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 计算胜率
            winning_trades = sum(1 for trade in trades if trade['action'] == 'sell')
            total_trades = len(trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            return {
                'strategy_type': 'mean_reversion',
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'trades': trades,
                'daily_values': daily_values
            }
            
        except Exception as e:
            logger.error(f"运行均值回归策略失败: {e}")
            raise
    
    async def _run_ml_based_strategy(
        self,
        data: pd.DataFrame,
        initial_cash: float,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """运行基于机器学习的策略"""
        try:
            # 这里简化处理，实际应该使用训练好的模型
            # 使用简单的移动平均交叉策略作为示例
            
            short_window = 10
            long_window = 30
            
            # 获取唯一股票代码
            symbols = data['symbol'].unique()
            
            # 初始化投资组合
            portfolio = {
                'cash': initial_cash,
                'positions': {}
            }
            
            trades = []
            daily_values = []
            
            # 按日期循环
            unique_dates = sorted(data.index.unique())
            
            for i, date in enumerate(unique_dates):
                day_data = data[data.index == date]
                
                if i >= long_window:
                    for symbol in symbols:
                        symbol_data = data[data['symbol'] == symbol]
                        symbol_day_data = day_data[day_data['symbol'] == symbol]
                        
                        if not symbol_day_data.empty:
                            # 获取过去的数据
                            start_idx = max(0, i - long_window)
                            past_data = symbol_data.iloc[start_idx:i]
                            
                            if len(past_data) >= long_window:
                                # 计算移动平均线
                                short_ma = past_data['close'].tail(short_window).mean()
                                long_ma = past_data['close'].tail(long_window).mean()
                                
                                current_price = symbol_day_data['close'].iloc[0]
                                current_position = portfolio['positions'].get(symbol, 0)
                                
                                # 交易信号
                                if short_ma > long_ma and current_position == 0:
                                    # 金叉，买入
                                    cash_available = portfolio['cash']
                                    shares = int(cash_available / current_price * 0.9)
                                    
                                    if shares > 0:
                                        cost = shares * current_price
                                        portfolio['cash'] -= cost
                                        portfolio['positions'][symbol] = shares
                                        
                                        trades.append({
                                            'date': date,
                                            'symbol': symbol,
                                            'action': 'buy',
                                            'quantity': shares,
                                            'price': current_price,
                                            'value': cost
                                        })
                                
                                elif short_ma < long_ma and current_position > 0:
                                    # 死叉，卖出
                                    shares = current_position
                                    value = shares * current_price
                                    portfolio['cash'] += value
                                    
                                    trades.append({
                                        'date': date,
                                        'symbol': symbol,
                                        'action': 'sell',
                                        'quantity': shares,
                                        'price': current_price,
                                        'value': value
                                    })
                                    
                                    del portfolio['positions'][symbol]
                
                # 计算组合价值
                portfolio_value = portfolio['cash']
                for symbol, shares in portfolio['positions'].items():
                    symbol_data = day_data[day_data['symbol'] == symbol]
                    if not symbol_data.empty:
                        price = symbol_data['close'].iloc[0]
                        portfolio_value += shares * price
                
                daily_values.append({
                    'date': date,
                    'portfolio_value': portfolio_value
                })
            
            # 计算回测指标
            daily_values_df = pd.DataFrame(daily_values)
            daily_values_df['returns'] = daily_values_df['portfolio_value'].pct_change()
            
            total_return = (daily_values_df['portfolio_value'].iloc[-1] - initial_cash) / initial_cash
            annual_return = (1 + total_return) ** (252 / len(daily_values_df)) - 1
            volatility = daily_values_df['returns'].std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # 计算最大回撤
            cumulative_returns = (1 + daily_values_df['returns'].fillna(0)).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # 计算胜率
            winning_trades = sum(1 for trade in trades if trade['action'] == 'sell')
            total_trades = len(trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            return {
                'strategy_type': 'ml_based',
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'trades': trades,
                'daily_values': daily_values
            }
            
        except Exception as e:
            logger.error(f"运行机器学习策略失败: {e}")
            raise
    
    async def _save_trades(self, backtest_id: int, trades: List[Dict[str, Any]]):
        """保存交易记录"""
        try:
            for trade in trades:
                db_trade = Trade(
                    backtest_id=backtest_id,
                    stock_id=0,  # 将在外部设置
                    trade_date=trade['date'],
                    action=trade['action'],
                    quantity=trade['quantity'],
                    price=trade['price'],
                    value=trade['value']
                )
                self.db.add(db_trade)
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")
            raise
    
    async def generate_detailed_results(
        self,
        backtest: Backtest,
        trades: List[Trade]
    ) -> Dict[str, Any]:
        """生成详细的回测结果"""
        try:
            # 基础统计
            basic_stats = {
                'total_return': backtest.total_return,
                'annual_return': backtest.annual_return,
                'max_drawdown': backtest.max_drawdown,
                'sharpe_ratio': backtest.sharpe_ratio,
                'volatility': backtest.volatility,
                'win_rate': backtest.win_rate,
                'profit_factor': backtest.profit_factor,
                'total_trades': backtest.total_trades,
                'winning_trades': backtest.winning_trades,
                'losing_trades': backtest.losing_trades
            }
            
            # 交易分析
            trade_analysis = await self._analyze_trades(trades)
            
            # 时间序列分析
            time_series_analysis = await self._analyze_time_series(backtest)
            
            return {
                'basic_stats': basic_stats,
                'trade_analysis': trade_analysis,
                'time_series_analysis': time_series_analysis,
                'trades': trades
            }
            
        except Exception as e:
            logger.error(f"生成详细回测结果失败: {e}")
            raise
    
    async def _analyze_trades(self, trades: List[Trade]) -> Dict[str, Any]:
        """分析交易"""
        try:
            if not trades:
                return {}
            
            # 转换为DataFrame
            trades_df = pd.DataFrame([
                {
                    'date': trade.trade_date,
                    'symbol': trade.stock.symbol if trade.stock else 'Unknown',
                    'action': trade.action,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'value': trade.value
                }
                for trade in trades
            ])
            
            # 分析买入交易
            buy_trades = trades_df[trades_df['action'] == 'buy']
            sell_trades = trades_df[trades_df['action'] == 'sell']
            
            analysis = {
                'total_trades': len(trades_df),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'avg_buy_price': buy_trades['price'].mean() if len(buy_trades) > 0 else 0,
                'avg_sell_price': sell_trades['price'].mean() if len(sell_trades) > 0 else 0,
                'total_buy_value': buy_trades['value'].sum() if len(buy_trades) > 0 else 0,
                'total_sell_value': sell_trades['value'].sum() if len(sell_trades) > 0 else 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"分析交易失败: {e}")
            raise
    
    async def _analyze_time_series(self, backtest: Backtest) -> Dict[str, Any]:
        """分析时间序列"""
        try:
            # 这里可以添加更复杂的时间序列分析
            # 目前返回基本信息
            return {
                'start_date': backtest.start_date,
                'end_date': backtest.end_date,
                'duration_days': (backtest.end_date - backtest.start_date).days,
                'started_at': backtest.started_at,
                'completed_at': backtest.completed_at,
                'duration_seconds': (backtest.completed_at - backtest.started_at).total_seconds() if backtest.completed_at and backtest.started_at else 0
            }
            
        except Exception as e:
            logger.error(f"分析时间序列失败: {e}")
            raise