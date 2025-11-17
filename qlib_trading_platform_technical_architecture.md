# Qlib量化交易平台技术架构文档

## 1. 系统架构概述

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端层 (Vue.js)                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │  股票查询   │ │  模型训练   │ │   回测     │ │  数据管理   ││
│  │  组件       │ │  组件       │ │  组件      │ │  组件       ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                    API网关 (Nginx)                              │
├─────────────────────────────────────────────────────────────────┤
│                        后端层 (FastAPI)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │  股票服务   │ │  模型服务   │ │  回测服务   │ │  用户服务   ││
│  │  Stock API  │ │  Model API  │ │Backtest API│ │  User API   ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                    业务逻辑层 (Service Layer)                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │  Qlib集成   │ │  数据处理   │ │  异步任务   │ │  缓存管理   ││
│  │  Qlib Wrapper│ │  Data Proc  │ │  Celery    │ │   Redis     ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                        数据层                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │ PostgreSQL  │ │    Redis    │ │   MinIO     │ │   Qlib      ││
│  │  业务数据   │ │   缓存层    │ │  文件存储   │ │  数据引擎   ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 架构特点

- **前后端分离**：Vue.js 单页应用 + FastAPI RESTful API
- **微服务架构**：各功能模块独立部署，便于扩展和维护
- **异步处理**：使用 Celery 处理长时间运行的任务（模型训练、回测）
- **多层缓存**：Redis 缓存 + 应用级缓存 + 数据库查询优化
- **高可用性**：支持水平扩展和负载均衡

## 2. 前端架构 (Vue.js)

### 2.1 技术栈

- **框架**：Vue 3 + TypeScript
- **状态管理**：Pinia
- **路由**：Vue Router 4
- **UI组件库**：Element Plus
- **图表库**：ECharts
- **HTTP客户端**：Axios
- **WebSocket**：Socket.io-client
- **构建工具**：Vite

### 2.2 项目结构

```
frontend/
├── src/
│   ├── api/                    # API接口封装
│   │   ├── stock.ts           # 股票相关API
│   │   ├── model.ts           # 模型相关API
│   │   ├── backtest.ts        # 回测相关API
│   │   └── user.ts            # 用户相关API
│   ├── assets/                # 静态资源
│   ├── components/            # 通用组件
│   │   ├── charts/           # 图表组件
│   │   ├── tables/           # 表格组件
│   │   └── common/           # 通用组件
│   ├── composables/           # 组合式函数
│   │   ├── useWebSocket.ts   # WebSocket封装
│   │   ├── useAuth.ts        # 认证相关
│   │   └── useChart.ts       # 图表相关
│   ├── layouts/              # 布局组件
│   │   ├── DefaultLayout.vue
│   │   └── AuthLayout.vue
│   ├── router/                 # 路由配置
│   │   └── index.ts
│   ├── stores/                 # 状态管理
│   │   ├── auth.ts            # 用户认证
│   │   ├── stock.ts           # 股票数据
│   │   ├── model.ts           # 模型管理
│   │   └── backtest.ts        # 回测状态
│   ├── types/                  # TypeScript类型定义
│   │   ├── stock.ts
│   │   ├── model.ts
│   │   └── api.ts
│   ├── utils/                  # 工具函数
│   │   ├── request.ts         # HTTP请求封装
│   │   ├── format.ts          # 数据格式化
│   │   └── constants.ts       # 常量定义
│   ├── views/                  # 页面组件
│   │   ├── dashboard/         # 仪表盘
│   │   ├── stocks/           # 股票查询
│   │   ├── models/            # 模型管理
│   │   ├── backtest/          # 回测分析
│   │   ├── data/              # 数据管理
│   │   └── settings/          # 系统设置
│   ├── App.vue
│   └── main.ts
├── public/
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

### 2.3 核心功能模块

#### 2.3.1 股票查询模块
- **股票搜索**：支持按代码、名称模糊搜索
- **K线图表**：支持日K、周K、月K等多种周期
- **技术指标**：MA、MACD、RSI、布林带等
- **基本面数据**：财务报表、估值指标等

#### 2.3.2 模型训练模块
- **模型选择**：支持 Qlib 内置的所有模型
- **参数配置**：可视化参数配置界面
- **训练监控**：实时显示训练进度和指标
- **模型管理**：模型保存、加载、版本管理

#### 2.3.3 回测分析模块
- **策略配置**：支持自定义策略参数
- **回测执行**：异步执行回测任务
- **结果展示**：收益曲线、风险指标、交易记录
- **绩效分析**：夏普比率、最大回撤、年化收益等

#### 2.3.4 数据管理模块
- **数据更新**：手动/自动更新股票数据
- **数据质量**：数据完整性检查
- **特征工程**：自定义特征计算
- **数据导出**：支持多种格式导出

### 2.4 状态管理设计

```typescript
// stores/auth.ts
export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: localStorage.getItem('token') || '',
    user: null as User | null,
  }),
  actions: {
    async login(credentials: LoginCredentials) {
      const response = await authAPI.login(credentials);
      this.token = response.data.token;
      this.user = response.data.user;
      localStorage.setItem('token', this.token);
    },
    logout() {
      this.token = '';
      this.user = null;
      localStorage.removeItem('token');
    }
  }
});

// stores/stock.ts
export const useStockStore = defineStore('stock', {
  state: () => ({
    stocks: [] as Stock[],
    currentStock: null as Stock | null,
    stockData: {} as Record<string, StockData>,
  }),
  actions: {
    async searchStocks(query: string) {
      const response = await stockAPI.search(query);
      this.stocks = response.data;
    },
    async fetchStockData(code: string, period: string) {
      const response = await stockAPI.getData(code, period);
      this.stockData[code] = response.data;
    }
  }
});
```

## 3. 后端架构 (FastAPI)

### 3.1 技术栈

- **框架**：FastAPI + Python 3.9+
- **数据库**：PostgreSQL + SQLAlchemy
- **缓存**：Redis
- **任务队列**：Celery + Redis/RabbitMQ
- **文件存储**：MinIO/S3
- **认证**：JWT + OAuth2
- **API文档**：自动生成 OpenAPI/Swagger
- **测试**：pytest

### 3.2 项目结构

```
backend/
├── app/
│   ├── api/                    # API路由
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── stocks.py      # 股票相关API
│   │   │   ├── models.py      # 模型相关API
│   │   │   ├── backtest.py    # 回测相关API
│   │   │   ├── users.py       # 用户相关API
│   │   │   └── auth.py        # 认证相关API
│   ├── core/                   # 核心配置
│   │   ├── config.py          # 配置管理
│   │   ├── security.py        # 安全相关
│   │   └── database.py        # 数据库连接
│   ├── crud/                   # 数据库操作
│   │   ├── stock.py
│   │   ├── model.py
│   │   ├── backtest.py
│   │   └── user.py
│   ├── models/                 # 数据模型
│   │   ├── stock.py
│   │   ├── model.py
│   │   ├── backtest.py
│   │   └── user.py
│   ├── schemas/                # Pydantic模型
│   │   ├── stock.py
│   │   ├── model.py
│   │   ├── backtest.py
│   │   └── user.py
│   ├── services/               # 业务逻辑
│   │   ├── qlib_service.py    # Qlib集成服务
│   │   ├── stock_service.py   # 股票服务
│   │   ├── model_service.py   # 模型服务
│   │   ├── backtest_service.py # 回测服务
│   │   └── data_service.py    # 数据服务
│   ├── tasks/                  # Celery任务
│   │   ├── __init__.py
│   │   ├── model_tasks.py     # 模型训练任务
│   │   ├── backtest_tasks.py  # 回测任务
│   │   └── data_tasks.py      # 数据更新任务
│   ├── utils/                  # 工具函数
│   │   ├── qlib_utils.py      # Qlib工具函数
│   │   ├── cache.py           # 缓存工具
│   │   └── common.py          # 通用工具
│   ├── main.py                # 应用入口
│   └── deps.py                # 依赖注入
├── alembic/                    # 数据库迁移
├── tests/                      # 测试文件
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── docker-compose.yml
```

### 3.3 核心服务设计

#### 3.3.1 Qlib集成服务

```python
# services/qlib_service.py
from typing import Dict, List, Optional
import qlib
from qlib.data import D
from qlib.contrib.model.pytorch_lstm import LSTM
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest

class QlibService:
    def __init__(self):
        self.initialized = False
    
    def init_qlib(self, provider_uri: str = "~/.qlib/qlib_data/cn_data"):
        """初始化Qlib"""
        if not self.initialized:
            qlib.init(provider_uri=provider_uri, region="cn")
            self.initialized = True
    
    def get_stock_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """获取股票数据"""
        self.init_qlib()
        data = D.features(
            symbols,
            ["$open", "$high", "$low", "$close", "$volume"],
            start_time=start_date,
            end_time=end_date
        )
        return data.to_dict()
    
    def train_model(self, model_config: Dict, dataset_config: Dict) -> str:
        """训练模型"""
        self.init_qlib()
        # 模型训练逻辑
        model = LSTM(**model_config)
        # ... 训练过程
        return model_id
    
    def run_backtest(self, strategy_config: Dict, start_date: str, end_date: str) -> Dict:
        """运行回测"""
        self.init_qlib()
        strategy = TopkDropoutStrategy(**strategy_config)
        # ... 回测逻辑
        return backtest_results
```

#### 3.3.2 异步任务处理

```python
# tasks/model_tasks.py
from celery import Celery
from app.services.qlib_service import QlibService

app = Celery('tasks', broker='redis://localhost:6379')

@app.task
def train_model_task(model_id: str, model_config: dict, dataset_config: dict):
    """异步训练模型任务"""
    qlib_service = QlibService()
    try:
        # 更新任务状态
        update_task_status(model_id, "running")
        
        # 执行模型训练
        result = qlib_service.train_model(model_config, dataset_config)
        
        # 更新任务状态
        update_task_status(model_id, "completed", result)
        return result
    except Exception as e:
        update_task_status(model_id, "failed", str(e))
        raise

@app.task
def run_backtest_task(backtest_id: str, strategy_config: dict, date_range: dict):
    """异步回测任务"""
    qlib_service = QlibService()
    try:
        # 更新任务状态
        update_task_status(backtest_id, "running")
        
        # 执行回测
        results = qlib_service.run_backtest(
            strategy_config, 
            date_range["start"], 
            date_range["end"]
        )
        
        # 更新任务状态
        update_task_status(backtest_id, "completed", results)
        return results
    except Exception as e:
        update_task_status(backtest_id, "failed", str(e))
        raise
```

### 3.4 API设计

#### 3.4.1 股票API

```python
# api/v1/stocks.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from app.services.stock_service import StockService
from app.schemas.stock import StockResponse, StockDataResponse

router = APIRouter()

@router.get("/search", response_model=List[StockResponse])
async def search_stocks(
    query: str = Query(..., min_length=1),
    stock_service: StockService = Depends(get_stock_service)
):
    """搜索股票"""
    stocks = await stock_service.search_stocks(query)
    return stocks

@router.get("/{symbol}/data", response_model=StockDataResponse)
async def get_stock_data(
    symbol: str,
    start_date: str = Query(...),
    end_date: str = Query(...),
    period: str = Query("1d"),
    stock_service: StockService = Depends(get_stock_service)
):
    """获取股票数据"""
    data = await stock_service.get_stock_data(symbol, start_date, end_date, period)
    return StockDataResponse(symbol=symbol, data=data)

@router.get("/{symbol}/indicators")
async def get_technical_indicators(
    symbol: str,
    indicators: List[str] = Query(...),
    stock_service: StockService = Depends(get_stock_service)
):
    """获取技术指标"""
    indicators_data = await stock_service.get_technical_indicators(symbol, indicators)
    return indicators_data
```

#### 3.4.2 模型API

```python
# api/v1/models.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict
from app.services.model_service import ModelService
from app.schemas.model import ModelCreate, ModelResponse, ModelTrainRequest

router = APIRouter()

@router.post("/train", response_model=Dict[str, str])
async def train_model(
    train_request: ModelTrainRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """训练模型"""
    task_id = await model_service.create_training_task(train_request)
    return {"task_id": task_id, "status": "submitted"}

@router.get("/{model_id}/status")
async def get_training_status(
    model_id: str,
    model_service: ModelService = Depends(get_model_service)
):
    """获取训练状态"""
    status = await model_service.get_training_status(model_id)
    return status

@router.get("/{model_id}/predict")
async def predict(
    model_id: str,
    symbols: List[str],
    model_service: ModelService = Depends(get_model_service)
):
    """模型预测"""
    predictions = await model_service.predict(model_id, symbols)
    return predictions
```

## 4. 数据层设计

### 4.1 数据库设计

```sql
-- 用户表
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 股票表
CREATE TABLE stocks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    market VARCHAR(10) NOT NULL,
    sector VARCHAR(50),
    industry VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 模型表
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 回测表
CREATE TABLE backtests (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    strategy_config JSONB NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    results JSONB,
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 任务表
CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(100) UNIQUE NOT NULL,
    task_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    parameters JSONB,
    result JSONB,
    error_message TEXT,
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4.2 缓存策略

```python
# utils/cache.py
from typing import Optional, Any
import redis
import json
from datetime import timedelta

class CacheManager:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set(self, key: str, value: Any, expire: int = 3600):
        """设置缓存"""
        self.redis.setex(
            key, 
            timedelta(seconds=expire), 
            json.dumps(value, default=str)
        )
    
    def delete(self, key: str):
        """删除缓存"""
        self.redis.delete(key)
    
    def delete_pattern(self, pattern: str):
        """删除匹配模式的缓存"""
        for key in self.redis.scan_iter(match=pattern):
            self.redis.delete(key)

# 缓存装饰器
def cache_result(expire: int = 3600, key_prefix: str = ""):
    """结果缓存装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached_result = await cache_manager.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, expire)
            return result
        return wrapper
    return decorator
```

## 5. 安全设计

### 5.1 认证授权

```python
# core/security.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """获取密码哈希"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user
```

### 5.2 API限流

```python
# utils/rate_limit.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import redis
from typing import Optional

class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def check_rate_limit(
        self, 
        key: str, 
        limit: int = 100, 
        window: int = 60
    ) -> bool:
        """检查速率限制"""
        current = datetime.now()
        window_start = current - timedelta(seconds=window)
        
        # 使用滑动窗口算法
        pipeline = self.redis.pipeline()
        pipeline.zremrangebyscore(key, 0, window_start.timestamp())
        pipeline.zcard(key)
        pipeline.zadd(key, {current.timestamp(): current.timestamp()})
        pipeline.expire(key, window)
        
        results = pipeline.execute()
        request_count = results[1]
        
        return request_count < limit
    
    async def rate_limit_middleware(
        self, 
        request: Request, 
        call_next
    ):
        """速率限制中间件"""
        client_ip = request.client.host
        endpoint = request.url.path
        key = f"rate_limit:{client_ip}:{endpoint}"
        
        if not await self.check_rate_limit(key):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"}
            )
        
        response = await call_next(request)
        return response
```

## 6. 性能优化

### 6.1 数据库优化

```python
# 数据库索引优化
"""
-- 为常用查询字段添加索引
CREATE INDEX idx_stocks_symbol ON stocks(symbol);
CREATE INDEX idx_stocks_market ON stocks(market);
CREATE INDEX idx_models_user_id ON models(user_id);
CREATE INDEX idx_models_status ON models(status);
CREATE INDEX idx_backtests_user_id ON backtests(user_id);
CREATE INDEX idx_backtests_status ON backtests(status);
CREATE INDEX idx_tasks_user_id ON tasks(user_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_task_id ON tasks(task_id);

-- 复合索引
CREATE INDEX idx_stocks_symbol_market ON stocks(symbol, market);
CREATE INDEX idx_models_user_status ON models(user_id, status);
CREATE INDEX idx_backtests_user_status ON backtests(user_id, status);
"""
```

### 6.2 查询优化

```python
# crud/stock.py
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import List, Optional
from app.models.stock import Stock

class CRUDStock:
    def get_by_symbol(self, db: Session, symbol: str) -> Optional[Stock]:
        """根据股票代码获取股票"""
        return db.query(Stock).filter(Stock.symbol == symbol).first()
    
    def search_stocks(
        self, 
        db: Session, 
        query: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[Stock]:
        """搜索股票"""
        return db.query(Stock).filter(
            or_(
                Stock.symbol.ilike(f"%{query}%"),
                Stock.name.ilike(f"%{query}%")
            )
        ).offset(skip).limit(limit).all()
    
    def get_multi_by_market(
        self, 
        db: Session, 
        market: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[Stock]:
        """按市场获取股票"""
        return db.query(Stock).filter(
            Stock.market == market
        ).offset(skip).limit(limit).all()

# 使用连接池
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

### 6.3 缓存策略

```python
# 多级缓存策略
"""
1. 浏览器缓存：静态资源缓存
2. CDN缓存：前端静态资源
3. Redis缓存：热点数据缓存
4. 应用缓存：内存缓存
5. 数据库缓存：查询结果缓存
"""

# 缓存预热
@app.on_event("startup")
async def cache_warmup():
    """缓存预热"""
    # 预热常用股票数据
    popular_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    for symbol in popular_stocks:
        await stock_service.get_stock_data(symbol)
    
    # 预热模型配置
    await model_service.get_model_configs()

# 缓存失效策略
class CacheInvalidator:
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    async def invalidate_stock_cache(self, symbol: str):
        """失效股票相关缓存"""
        patterns = [
            f"stock:*:{symbol}:*",
            f"indicators:*:{symbol}:*",
            f"prediction:*:{symbol}:*"
        ]
        for pattern in patterns:
            await self.cache.delete_pattern(pattern)
    
    async def invalidate_model_cache(self, model_id: str):
        """失效模型相关缓存"""
        patterns = [
            f"model:*:{model_id}:*",
            f"prediction:*:{model_id}:*"
        ]
        for pattern in patterns:
            await self.cache.delete_pattern(pattern)
```

## 7. 部署架构

### 7.1 Docker容器化

```dockerfile
# backend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# frontend/Dockerfile
FROM node:16-alpine as build-stage

WORKDIR /app

# 复制依赖文件
COPY package*.json ./
RUN npm ci --only=production

# 复制源码
COPY . .

# 构建应用
RUN npm run build

# 生产阶段
FROM nginx:stable-alpine as production-stage

# 复制构建结果
COPY --from=build-stage /app/dist /usr/share/nginx/html

# 复制nginx配置
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### 7.2 Docker Compose配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  # PostgreSQL数据库
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: qlib_platform
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - qlib_network

  # Redis缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - qlib_network

  # MinIO对象存储
  minio:
    image: minio/minio
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    command: server /data --console-address ":9001"
    networks:
      - qlib_network

  # 后端API
  backend:
    build: ./backend
    environment:
      DATABASE_URL: postgresql://postgres:password@postgres:5432/qlib_platform
      REDIS_URL: redis://redis:6379
      MINIO_ENDPOINT: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
      - minio
    volumes:
      - ./backend:/app
      - qlib_data:/root/.qlib
    networks:
      - qlib_network

  # 前端应用
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    networks:
      - qlib_network

  # Celery Worker
  celery_worker:
    build: ./backend
    command: celery -A app.tasks worker --loglevel=info
    environment:
      DATABASE_URL: postgresql://postgres:password@postgres:5432/qlib_platform
      REDIS_URL: redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./backend:/app
      - qlib_data:/root/.qlib
    networks:
      - qlib_network

  # Celery Beat (定时任务)
  celery_beat:
    build: ./backend
    command: celery -A app.tasks beat --loglevel=info
    environment:
      DATABASE_URL: postgresql://postgres:password@postgres:5432/qlib_platform
      REDIS_URL: redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./backend:/app
    networks:
      - qlib_network

volumes:
  postgres_data:
  minio_data:
  qlib_data:

networks:
  qlib_network:
    driver: bridge
```

### 7.3 生产环境部署

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Nginx反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - ./nginx/logs:/var/log/nginx
    depends_on:
      - frontend
      - backend
    networks:
      - qlib_network

  # 后端API (多实例)
  backend:
    build: ./backend
    environment:
      DATABASE_URL: postgresql://postgres:password@postgres:5432/qlib_platform
      REDIS_URL: redis://redis:6379
      MINIO_ENDPOINT: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
    depends_on:
      - postgres
      - redis
      - minio
    networks:
      - qlib_network

  # 监控和日志
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - qlib_network

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - qlib_network

networks:
  qlib_network:
    driver: overlay
```

## 8. 监控和日志

### 8.1 应用监控

```python
# utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# 指标定义
request_count = Counter(
    'http_requests_total', 
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

active_models = Gauge(
    'active_models_total',
    'Total number of active models'
)

backtest_duration = Histogram(
    'backtest_duration_seconds',
    'Backtest execution duration'
)

def track_metrics(func):
    """性能监控装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            status = "success"
            return result
        except Exception as e:
            status = "error"
            raise
        finally:
            duration = time.time() - start_time
            
            # 记录指标
            request_count.labels(
                method="POST",  # 假设都是POST请求
                endpoint=func.__name__,
                status=status
            ).inc()
            
            request_duration.labels(
                method="POST",
                endpoint=func.__name__
            ).observe(duration)
    
    return wrapper
```

### 8.2 日志管理

```python
# core/logging.py
import logging
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger

def setup_logging():
    """配置日志"""
    # JSON格式日志
    logHandler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    logHandler.setFormatter(formatter)
    
    # 配置根日志器
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logHandler],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 访问日志
    access_logger = logging.getLogger("access")
    access_logger.setLevel(logging.INFO)
    
    # 错误日志
    error_logger = logging.getLogger("error")
    error_logger.setLevel(logging.ERROR)
    
    # 业务日志
    business_logger = logging.getLogger("business")
    business_logger.setLevel(logging.INFO)

# 日志中间件
async def logging_middleware(request, call_next):
    """请求日志中间件"""
    start_time = datetime.now()
    
    # 记录请求
    access_logger.info(
        "Request started",
        extra={
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent")
        }
    )
    
    response = await call_next(request)
    
    # 记录响应
    duration = (datetime.now() - start_time).total_seconds()
    access_logger.info(
        "Request completed",
        extra={
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "duration": duration
        }
    )
    
    return response
```

## 9. 扩展性设计

### 9.1 水平扩展

```yaml
# Kubernetes部署配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qlib-backend
spec:
  replicas: 3  # 初始副本数
  selector:
    matchLabels:
      app: qlib-backend
  template:
    metadata:
      labels:
        app: qlib-backend
    spec:
      containers:
      - name: backend
        image: qlib-platform/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: qlib-backend-service
spec:
  selector:
    app: qlib-backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: qlib-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: qlib-backend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 9.2 数据分片

```python
# 数据分片策略
class DataSharding:
    """数据分片管理"""
    
    def __init__(self, shard_count: int = 4):
        self.shard_count = shard_count
    
    def get_shard_key(self, symbol: str) -> int:
        """获取分片键"""
        return hash(symbol) % self.shard_count
    
    def get_shard_connection(self, shard_key: int):
        """获取分片连接"""
        # 根据分片键返回对应的数据库连接
        shard_config = {
            0: "postgresql://shard0:5432/qlib_shard0",
            1: "postgresql://shard1:5432/qlib_shard1", 
            2: "postgresql://shard2:5432/qlib_shard2",
            3: "postgresql://shard3:5432/qlib_shard3"
        }
        return shard_config.get(shard_key)

# 使用示例
class ShardedStockService:
    def __init__(self, sharding: DataSharding):
        self.sharding = sharding
    
    async def get_stock_data(self, symbol: str):
        """获取股票数据（分片）"""
        shard_key = self.sharding.get_shard_key(symbol)
        connection = self.sharding.get_shard_connection(shard_key)
        
        # 在对应的分片中查询数据
        async with connection.acquire() as conn:
            result = await conn.fetch(
                "SELECT * FROM stock_data WHERE symbol = $1",
                symbol
            )
            return result
```

## 10. 总结

本技术架构文档详细描述了基于Qlib的量化交易平台的完整技术方案，包括：

1. **系统架构**：采用前后端分离的微服务架构，支持高并发和水平扩展
2. **前端设计**：Vue 3 + TypeScript + Element Plus，提供现代化的用户界面
3. **后端设计**：FastAPI + SQLAlchemy + Redis，提供高性能的API服务
4. **Qlib集成**：深度集成Qlib的量化功能，提供模型训练和回测能力
5. **数据层设计**：PostgreSQL + Redis + MinIO，支持海量数据存储和缓存
6. **安全设计**：JWT认证 + OAuth2授权 + API限流，确保系统安全
7. **性能优化**：多级缓存 + 数据库优化 + 异步处理，保证系统性能
8. **部署架构**：Docker容器化 + Kubernetes编排，支持云原生部署
9. **监控日志**：Prometheus + Grafana + ELK，提供完整的监控体系
10. **扩展性设计**：支持水平扩展和数据分片，满足业务增长需求

该架构方案具有以下优势：

- **高性能**：多层缓存和异步处理，支持高并发访问
- **高可用**：微服务架构和容器化部署，支持故障自动恢复
- **可扩展**：水平扩展和数据分片，支持业务快速增长
- **易维护**：模块化设计和完善的监控，便于系统维护
- **安全性**：完整的认证授权和限流机制，保障系统安全

通过这套架构，可以构建一个功能完整、性能优异、安全可靠的量化交易平台，为用户提供专业的量化投资服务。