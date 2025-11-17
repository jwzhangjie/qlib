from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime


class User(Base):
    """用户模型"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联
    portfolios = relationship("Portfolio", back_populates="user")
    backtests = relationship("Backtest", back_populates="user")
    models = relationship("Model", back_populates="user")


class Stock(Base):
    """股票模型"""
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    market = Column(String(20), nullable=False)  # 'SH', 'SZ', 'US', etc.
    industry = Column(String(50), nullable=True)
    sector = Column(String(50), nullable=True)
    exchange = Column(String(20), nullable=True)
    currency = Column(String(10), default="CNY")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联
    prices = relationship("StockPrice", back_populates="stock")
    portfolios = relationship("PortfolioStock", back_populates="stock")


class StockPrice(Base):
    """股票价格模型"""
    __tablename__ = "stock_prices"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    adj_close = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 关联
    stock = relationship("Stock", back_populates="prices")
    
    # 复合索引
    __table_args__ = (
        # 确保每个股票每天只有一条记录
        {'sqlite_autoincrement': True},
    )


class Model(Base):
    """机器学习模型"""
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    model_type = Column(String(50), nullable=False)  # 'lightgbm', 'xgboost', 'lstm', etc.
    config = Column(JSON, nullable=True)  # 模型配置参数
    features = Column(JSON, nullable=True)  # 特征列表
    target = Column(String(50), nullable=False)  # 预测目标
    performance_metrics = Column(JSON, nullable=True)  # 模型性能指标
    file_path = Column(String(500), nullable=True)  # 模型文件路径
    is_trained = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    trained_at = Column(DateTime(timezone=True), nullable=True)
    
    # 关联
    user = relationship("User", back_populates="models")
    predictions = relationship("Prediction", back_populates="model")


class Prediction(Base):
    """模型预测结果"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)
    prediction = Column(Float, nullable=False)
    probability = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 关联
    model = relationship("Model", back_populates="predictions")
    stock = relationship("Stock")


class Backtest(Base):
    """回测结果"""
    __tablename__ = "backtests"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    strategy_config = Column(JSON, nullable=False)  # 策略配置
    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    initial_cash = Column(Float, default=1000000.0)
    
    # 回测结果
    total_return = Column(Float, nullable=True)
    annual_return = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    
    # 统计信息
    total_trades = Column(Integer, nullable=True)
    winning_trades = Column(Integer, nullable=True)
    losing_trades = Column(Integer, nullable=True)
    
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    
    result_file = Column(String(500), nullable=True)  # 结果文件路径
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # 关联
    user = relationship("User", back_populates="backtests")
    model = relationship("Model")
    trades = relationship("Trade", back_populates="backtest")


class Trade(Base):
    """交易记录"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    backtest_id = Column(Integer, ForeignKey("backtests.id"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    
    trade_date = Column(DateTime(timezone=True), nullable=False)
    action = Column(String(10), nullable=False)  # 'buy', 'sell'
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    value = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    
    # 交易后的持仓信息
    position_after = Column(Integer, nullable=True)
    cash_after = Column(Float, nullable=True)
    portfolio_value = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 关联
    backtest = relationship("Backtest", back_populates="trades")
    stock = relationship("Stock")


class Portfolio(Base):
    """投资组合"""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    total_value = Column(Float, default=0.0)
    cash = Column(Float, default=0.0)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联
    user = relationship("User", back_populates="portfolios")
    stocks = relationship("PortfolioStock", back_populates="portfolio")


class PortfolioStock(Base):
    """投资组合中的股票"""
    __tablename__ = "portfolio_stocks"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    
    quantity = Column(Integer, nullable=False)
    average_cost = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    market_value = Column(Float, nullable=True)
    profit_loss = Column(Float, nullable=True)
    profit_loss_percent = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联
    portfolio = relationship("Portfolio", back_populates="stocks")
    stock = relationship("Stock", back_populates="portfolios")


class DataUpdateTask(Base):
    """数据更新任务"""
    __tablename__ = "data_update_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(100), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    data_type = Column(String(50), nullable=False)  # 'stock', 'market', 'financial'
    symbols = Column(JSON, nullable=True)  # 股票代码列表
    date_range_start = Column(DateTime(timezone=True), nullable=True)
    date_range_end = Column(DateTime(timezone=True), nullable=True)
    
    status = Column(String(20), default="pending")  # pending, running, completed, failed, cancelled
    progress = Column(Float, default=0.0)
    total_items = Column(Integer, default=0)
    processed_items = Column(Integer, default=0)
    
    error_message = Column(Text, nullable=True)
    result_summary = Column(JSON, nullable=True)
    
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class MarketData(Base):
    """市场数据"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime(timezone=True), nullable=False)
    market = Column(String(20), nullable=False)  # 'SH', 'SZ', 'US', etc.
    index_symbol = Column(String(20), nullable=False)  # 'SH000001', 'SZ399001', etc.
    index_name = Column(String(100), nullable=False)
    
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 复合索引
    __table_args__ = (
        # 确保每个指数每天只有一条记录
        {'sqlite_autoincrement': True},
    )