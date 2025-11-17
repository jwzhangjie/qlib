from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from decimal import Decimal


# 基础响应模型
class ResponseBase(BaseModel):
    success: bool = True
    message: str = "success"
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    detail: Optional[str] = None


# 用户相关模型
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, max_length=100)


class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserResponse(ResponseBase):
    data: Optional[UserInDB] = None


# 股票相关模型
class StockBase(BaseModel):
    symbol: str = Field(..., max_length=20)
    name: str = Field(..., max_length=100)
    market: str = Field(..., max_length=20)
    industry: Optional[str] = None
    sector: Optional[str] = None
    exchange: Optional[str] = None
    currency: str = "CNY"
    is_active: bool = True


class StockCreate(StockBase):
    pass


class StockUpdate(BaseModel):
    name: Optional[str] = None
    industry: Optional[str] = None
    sector: Optional[str] = None
    is_active: Optional[bool] = None


class StockInDB(StockBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class StockResponse(ResponseBase):
    data: Optional[StockInDB] = None


class StockListResponse(ResponseBase):
    data: List[StockInDB]
    total: int
    page: int
    page_size: int


# 股票价格模型
class StockPriceBase(BaseModel):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: Optional[float] = None


class StockPriceCreate(StockPriceBase):
    stock_id: int


class StockPriceInDB(StockPriceBase):
    id: int
    stock_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class StockPriceResponse(ResponseBase):
    data: List[StockPriceInDB]


# 模型相关模型
class ModelBase(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    model_type: str = Field(..., max_length=50)
    target: str = Field(..., max_length=50)
    features: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None


class ModelCreate(ModelBase):
    pass


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class ModelInDB(ModelBase):
    id: int
    user_id: int
    file_path: Optional[str] = None
    is_trained: bool
    is_active: bool
    performance_metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    trained_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ModelResponse(ResponseBase):
    data: Optional[ModelInDB] = None


class ModelListResponse(ResponseBase):
    data: List[ModelInDB]
    total: int
    page: int
    page_size: int


# 预测结果模型
class PredictionBase(BaseModel):
    model_id: int
    stock_id: int
    date: datetime
    prediction: float
    probability: Optional[float] = None
    confidence: Optional[float] = None


class PredictionInDB(PredictionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionResponse(ResponseBase):
    data: List[PredictionInDB]


# 回测相关模型
class BacktestBase(BaseModel):
    name: str = Field(..., max_length=200)
    description: Optional[str] = None
    strategy_config: Dict[str, Any]
    start_date: datetime
    end_date: datetime
    initial_cash: float = 1000000.0


class BacktestCreate(BacktestBase):
    model_id: Optional[int] = None


class BacktestUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class BacktestInDB(BacktestBase):
    id: int
    user_id: int
    model_id: Optional[int] = None
    
    # 回测结果
    total_return: Optional[float] = None
    annual_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    volatility: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    
    # 统计信息
    total_trades: Optional[int] = None
    winning_trades: Optional[int] = None
    losing_trades: Optional[int] = None
    
    status: str = "pending"
    error_message: Optional[str] = None
    result_file: Optional[str] = None
    
    created_at: datetime
    updated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class BacktestResponse(ResponseBase):
    data: Optional[BacktestInDB] = None


class BacktestListResponse(ResponseBase):
    data: List[BacktestInDB]
    total: int
    page: int
    page_size: int


# 交易记录模型
class TradeBase(BaseModel):
    stock_id: int
    trade_date: datetime
    action: str = Field(..., max_length=10)  # 'buy', 'sell'
    quantity: int
    price: float
    value: float
    commission: float = 0.0


class TradeInDB(TradeBase):
    id: int
    backtest_id: int
    
    # 交易后的持仓信息
    position_after: Optional[int] = None
    cash_after: Optional[float] = None
    portfolio_value: Optional[float] = None
    
    created_at: datetime

    class Config:
        from_attributes = True


class TradeResponse(ResponseBase):
    data: List[TradeInDB]


# 投资组合模型
class PortfolioBase(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    total_value: float = 0.0
    cash: float = 0.0


class PortfolioCreate(PortfolioBase):
    pass


class PortfolioUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class PortfolioInDB(PortfolioBase):
    id: int
    user_id: int
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PortfolioResponse(ResponseBase):
    data: Optional[PortfolioInDB] = None


# 数据更新任务模型
class DataUpdateTaskBase(BaseModel):
    name: str = Field(..., max_length=200)
    description: Optional[str] = None
    data_type: str = Field(..., max_length=50)
    symbols: Optional[List[str]] = None
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None


class DataUpdateTaskCreate(DataUpdateTaskBase):
    pass


class DataUpdateTaskInDB(DataUpdateTaskBase):
    id: int
    task_id: str
    
    status: str = "pending"
    progress: float = 0.0
    total_items: int = 0
    processed_items: int = 0
    error_message: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None
    
    created_by: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DataUpdateTaskResponse(ResponseBase):
    data: Optional[DataUpdateTaskInDB] = None


# 认证相关模型
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[int] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(ResponseBase):
    data: Optional[Token] = None


# 股票搜索模型
class StockSearchRequest(BaseModel):
    query: str
    market: Optional[str] = None
    limit: int = Field(default=20, le=100)


class StockSearchResponse(ResponseBase):
    data: List[StockInDB]


# 股票数据请求模型
class StockDataRequest(BaseModel):
    symbol: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    period: str = "1d"  # '1d', '1w', '1m', '1y'
    indicators: Optional[List[str]] = None  # ['ma5', 'ma20', 'rsi', 'macd']


class StockDataResponse(ResponseBase):
    data: Dict[str, Any]  # 包含价格和指标数据


# 模型训练请求模型
class ModelTrainRequest(BaseModel):
    model_id: int
    start_date: datetime
    end_date: datetime
    symbols: Optional[List[str]] = None
    hyperparameters: Optional[Dict[str, Any]] = None


class ModelTrainResponse(ResponseBase):
    data: Dict[str, Any]  # 包含训练任务信息


# 预测请求模型
class PredictionRequest(BaseModel):
    model_id: int
    symbol: str
    date: Optional[datetime] = None


class PredictionResponse(ResponseBase):
    data: Dict[str, Any]  # 包含预测结果


# 回测请求模型
class BacktestRequest(BaseModel):
    strategy_type: str  # 'buy_and_hold', 'momentum', 'mean_reversion', 'ml_based'
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    initial_cash: float = 1000000.0
    parameters: Optional[Dict[str, Any]] = None


class BacktestResponse(ResponseBase):
    data: Dict[str, Any]  # 包含回测结果