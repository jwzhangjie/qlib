import yfinance as yf
import akshare as ak
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import logging

from app.models import Stock, StockPrice
from app.core.config import settings

logger = logging.getLogger(__name__)


class StockService:
    """股票数据服务"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.data_sources = {
            'yahoo': self._fetch_yahoo_data,
            'tushare': self._fetch_tushare_data,
            'akshare': self._fetch_akshare_data
        }
    
    async def get_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1d",
        indicators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """获取股票数据"""
        try:
            # 获取股票信息
            result = await self.db.execute(select(Stock).where(Stock.symbol == symbol))
            stock = result.scalar_one_or_none()
            
            if not stock:
                raise ValueError(f"股票 {symbol} 不存在")
            
            # 转换日期格式
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            else:
                start_dt = datetime.now() - timedelta(days=365)
            
            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end_dt = datetime.now()
            
            # 从数据库获取数据
            price_result = await self.db.execute(
                select(StockPrice).where(
                    and_(
                        StockPrice.stock_id == stock.id,
                        StockPrice.date >= start_dt,
                        StockPrice.date <= end_dt
                    )
                ).order_by(StockPrice.date)
            )
            prices = price_result.scalars().all()
            
            if not prices:
                # 如果数据库中没有数据，尝试从外部数据源获取
                prices = await self._fetch_external_data(symbol, start_dt, end_dt)
            
            # 转换为DataFrame
            df = pd.DataFrame([
                {
                    'date': p.date,
                    'open': p.open,
                    'high': p.high,
                    'low': p.low,
                    'close': p.close,
                    'volume': p.volume,
                    'adj_close': p.adj_close
                }
                for p in prices
            ])
            
            # 计算技术指标
            if indicators:
                df = self._calculate_indicators(df, indicators)
            
            return {
                'symbol': symbol,
                'name': stock.name,
                'market': stock.market,
                'data': df.to_dict('records'),
                'start_date': start_date,
                'end_date': end_date,
                'total_records': len(df)
            }
            
        except Exception as e:
            logger.error(f"获取股票数据失败: {e}")
            raise
    
    async def calculate_indicators(
        self,
        symbol: str,
        indicators: List[str],
        period: str = "1d"
    ) -> Dict[str, Any]:
        """计算技术指标"""
        try:
            # 获取股票数据
            stock_data = await self.get_stock_data(symbol=symbol, period=period)
            df = pd.DataFrame(stock_data['data'])
            
            # 计算指标
            result_df = self._calculate_indicators(df, indicators)
            
            return {
                'symbol': symbol,
                'indicators': indicators,
                'data': result_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            raise
    
    async def sync_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """同步股票数据"""
        try:
            # 获取股票信息
            result = await self.db.execute(select(Stock).where(Stock.symbol == symbol))
            stock = result.scalar_one_or_none()
            
            if not stock:
                raise ValueError(f"股票 {symbol} 不存在")
            
            # 转换日期格式
            if start_date:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            else:
                start_dt = datetime.now() - timedelta(days=365)
            
            if end_date:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end_dt = datetime.now()
            
            # 从外部数据源获取数据
            prices = await self._fetch_external_data(symbol, start_dt, end_dt)
            
            # 保存到数据库
            synced_count = 0
            for price in prices:
                # 检查是否已存在
                existing = await self.db.execute(
                    select(StockPrice).where(
                        and_(
                            StockPrice.stock_id == stock.id,
                            StockPrice.date == price.date
                        )
                    )
                )
                
                if existing.scalar_one_or_none():
                    continue
                
                # 创建新价格记录
                new_price = StockPrice(
                    stock_id=stock.id,
                    date=price.date,
                    open=price.open,
                    high=price.high,
                    low=price.low,
                    close=price.close,
                    volume=price.volume,
                    adj_close=price.adj_close
                )
                self.db.add(new_price)
                synced_count += 1
            
            await self.db.commit()
            
            return {
                'symbol': symbol,
                'synced_records': synced_count,
                'start_date': start_date,
                'end_date': end_date
            }
            
        except Exception as e:
            logger.error(f"同步股票数据失败: {e}")
            raise
    
    def _calculate_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        # 移动平均线
        if 'ma5' in indicators:
            df['ma5'] = df['close'].rolling(window=5).mean()
        if 'ma10' in indicators:
            df['ma10'] = df['close'].rolling(window=10).mean()
        if 'ma20' in indicators:
            df['ma20'] = df['close'].rolling(window=20).mean()
        if 'ma30' in indicators:
            df['ma30'] = df['close'].rolling(window=30).mean()
        
        # RSI
        if 'rsi' in indicators:
            df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        if 'macd' in indicators:
            df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        
        # 布林带
        if 'bollinger' in indicators:
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # 成交量指标
        if 'volume_ma' in indicators:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    async def _fetch_external_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[StockPrice]:
        """从外部数据源获取数据"""
        try:
            # 根据股票代码选择数据源
            if symbol.endswith('.SZ') or symbol.endswith('.SH'):
                # A股数据
                return await self._fetch_akshare_data(symbol, start_date, end_date)
            else:
                # 美股或其他市场数据
                return await self._fetch_yahoo_data(symbol, start_date, end_date)
                
        except Exception as e:
            logger.error(f"从外部数据源获取数据失败: {e}")
            raise
    
    async def _fetch_yahoo_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[StockPrice]:
        """从Yahoo Finance获取数据"""
        try:
            # 下载数据
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                return []
            
            # 转换为StockPrice对象
            prices = []
            for index, row in df.iterrows():
                price = StockPrice(
                    stock_id=0,  # 将在外部设置
                    date=index.to_pydatetime(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume']),
                    adj_close=float(row['Close'])  # Yahoo数据已调整后
                )
                prices.append(price)
            
            return prices
            
        except Exception as e:
            logger.error(f"从Yahoo Finance获取数据失败: {e}")
            raise
    
    async def _fetch_akshare_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[StockPrice]:
        """从Akshare获取A股数据"""
        try:
            # 转换日期格式
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            
            # 获取数据
            if symbol.endswith('.SZ'):
                # 深交所
                df = ak.stock_zh_a_hist(symbol=symbol.replace('.SZ', ''), start_date=start_str, end_date=end_str)
            elif symbol.endswith('.SH'):
                # 上交所
                df = ak.stock_zh_a_hist(symbol=symbol.replace('.SH', ''), start_date=start_str, end_date=end_str)
            else:
                return []
            
            if df.empty:
                return []
            
            # 转换列名
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume'
            })
            
            # 转换为StockPrice对象
            prices = []
            for _, row in df.iterrows():
                price = StockPrice(
                    stock_id=0,  # 将在外部设置
                    date=pd.to_datetime(row['date']),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']),
                    adj_close=float(row['close'])
                )
                prices.append(price)
            
            return prices
            
        except Exception as e:
            logger.error(f"从Akshare获取数据失败: {e}")
            raise
    
    async def _fetch_tushare_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[StockPrice]:
        """从Tushare获取数据"""
        try:
            # 设置Tushare token
            ts.set_token(settings.TUSHARE_TOKEN)
            pro = ts.pro_api()
            
            # 转换日期格式
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            
            # 获取数据
            ts_symbol = symbol.replace('.SZ', '').replace('.SH', '')
            df = pro.daily(ts_code=ts_symbol, start_date=start_str, end_date=end_str)
            
            if df.empty:
                return []
            
            # 转换列名
            df = df.rename(columns={
                'trade_date': 'date',
                'open': 'open',
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'vol': 'volume'
            })
            
            # 转换为StockPrice对象
            prices = []
            for _, row in df.iterrows():
                price = StockPrice(
                    stock_id=0,  # 将在外部设置
                    date=pd.to_datetime(row['date']),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']),
                    adj_close=float(row['close'])
                )
                prices.append(price)
            
            return prices
            
        except Exception as e:
            logger.error(f"从Tushare获取数据失败: {e}")
            raise