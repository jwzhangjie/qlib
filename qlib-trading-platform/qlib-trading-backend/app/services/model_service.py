import os
import pickle
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models import Model, Stock, StockPrice, Prediction
from app.core.config import settings
from app.core.redis import get_redis, RedisKeys

logger = logging.getLogger(__name__)


class ModelService:
    """模型服务"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.redis_client = None
    
    async def _get_redis(self):
        """获取Redis客户端"""
        if not self.redis_client:
            self.redis_client = await get_redis()
        return self.redis_client
    
    async def train_model(
        self,
        model: Model,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """训练模型"""
        try:
            # 获取训练数据
            training_data = await self._prepare_training_data(
                start_date, end_date, symbols
            )
            
            if training_data.empty:
                raise ValueError("没有可用的训练数据")
            
            # 根据模型类型选择训练方法
            if model.model_type == 'lightgbm':
                trained_model, metrics = await self._train_lightgbm(
                    training_data, model.config, hyperparameters
                )
            elif model.model_type == 'xgboost':
                trained_model, metrics = await self._train_xgboost(
                    training_data, model.config, hyperparameters
                )
            elif model.model_type == 'lstm':
                trained_model, metrics = await self._train_lstm(
                    training_data, model.config, hyperparameters
                )
            else:
                raise ValueError(f"不支持的模型类型: {model.model_type}")
            
            # 保存模型
            model_path = await self._save_model(trained_model, model.id)
            
            # 更新模型信息
            model.file_path = model_path
            model.is_trained = True
            model.trained_at = datetime.now()
            model.performance_metrics = metrics
            
            await self.db.commit()
            
            # 缓存模型
            await self._cache_model(model.id, trained_model)
            
            return {
                'model_id': model.id,
                'status': 'completed',
                'metrics': metrics,
                'training_samples': len(training_data),
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f"训练模型失败: {e}")
            raise
    
    async def predict(
        self,
        model: Model,
        symbol: str,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """使用模型进行预测"""
        try:
            # 加载模型
            trained_model = await self._load_model(model)
            
            # 准备预测数据
            prediction_data = await self._prepare_prediction_data(symbol, date)
            
            if prediction_data.empty:
                raise ValueError("没有可用的预测数据")
            
            # 进行预测
            predictions = trained_model.predict(prediction_data)
            
            # 保存预测结果
            prediction_result = await self._save_prediction(
                model.id, symbol, date, predictions
            )
            
            return {
                'symbol': symbol,
                'date': date or datetime.now().strftime("%Y-%m-%d"),
                'prediction': float(predictions[0]),
                'model_type': model.model_type,
                'confidence': self._calculate_confidence(predictions)
            }
            
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            raise
    
    async def _prepare_training_data(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """准备训练数据"""
        try:
            # 获取股票数据
            query = select(StockPrice).join(Stock).where(
                and_(
                    StockPrice.date >= start_date,
                    StockPrice.date <= end_date
                )
            )
            
            if symbols:
                query = query.where(Stock.symbol.in_(symbols))
            
            result = await self.db.execute(query)
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
            
            # 计算特征
            df = self._calculate_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"准备训练数据失败: {e}")
            raise
    
    async def _prepare_prediction_data(
        self,
        symbol: str,
        date: Optional[str] = None
    ) -> pd.DataFrame:
        """准备预测数据"""
        try:
            # 获取最新的股票数据
            if date:
                target_date = datetime.strptime(date, "%Y-%m-%d")
            else:
                target_date = datetime.now()
            
            # 获取过去30天的数据用于计算特征
            start_date = target_date - timedelta(days=30)
            
            result = await self.db.execute(
                select(StockPrice).join(Stock).where(
                    and_(
                        Stock.symbol == symbol,
                        StockPrice.date >= start_date,
                        StockPrice.date <= target_date
                    )
                ).order_by(StockPrice.date.desc())
            )
            prices = result.scalars().all()
            
            if not prices:
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame([
                {
                    'date': price.date,
                    'open': price.open,
                    'high': price.high,
                    'low': price.low,
                    'close': price.close,
                    'volume': price.volume
                }
                for price in prices
            ])
            
            # 计算特征（只使用最后一天的数据）
            df = self._calculate_features(df)
            
            # 返回最后一行数据
            return df.tail(1)
            
        except Exception as e:
            logger.error(f"准备预测数据失败: {e}")
            raise
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算特征"""
        df = df.copy()
        
        # 价格特征
        df['price_change'] = df['close'].pct_change()
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # 成交量特征
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=10).mean()
        
        # 移动平均特征
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma_ratio'] = df['ma5'] / df['ma20']
        
        # 技术指标
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], df['macd_signal'], _ = self._calculate_macd(df['close'])
        
        # 滞后特征
        for lag in [1, 2, 3, 5]:
            df[f'price_lag_{lag}'] = df['price_change'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume_change'].shift(lag)
        
        # 目标变量（未来1天的收益率）
        df['target'] = df['close'].shift(-1) / df['close'] - 1
        
        # 删除包含NaN的行
        df = df.dropna()
        
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
        return macd, macd_signal, macd - macd_signal
    
    async def _train_lightgbm(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """训练LightGBM模型"""
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
            
            # 准备特征和目标
            feature_cols = [col for col in data.columns if col not in ['target', 'date', 'symbol']]
            X = data[feature_cols]
            y = data['target']
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 创建模型
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
            
            if hyperparameters:
                params.update(hyperparameters)
            
            # 训练模型
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            return model, metrics
            
        except ImportError:
            raise ImportError("LightGBM未安装，请安装: pip install lightgbm")
        except Exception as e:
            logger.error(f"训练LightGBM模型失败: {e}")
            raise
    
    async def _train_xgboost(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """训练XGBoost模型"""
        try:
            from xgboost import XGBRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
            
            # 准备特征和目标
            feature_cols = [col for col in data.columns if col not in ['target', 'date', 'symbol']]
            X = data[feature_cols]
            y = data['target']
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 创建模型
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
            
            if hyperparameters:
                params.update(hyperparameters)
            
            # 训练模型
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            return model, metrics
            
        except ImportError:
            raise ImportError("XGBoost未安装，请安装: pip install xgboost")
        except Exception as e:
            logger.error(f"训练XGBoost模型失败: {e}")
            raise
    
    async def _train_lstm(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """训练LSTM模型"""
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
            
            # 准备序列数据
            feature_cols = [col for col in data.columns if col not in ['target', 'date', 'symbol']]
            sequence_data = data[feature_cols].values
            target_data = data['target'].values
            
            # 数据标准化
            scaler = MinMaxScaler()
            sequence_data = scaler.fit_transform(sequence_data)
            
            # 创建序列
            def create_sequences(data, target, seq_length):
                X, y = [], []
                for i in range(seq_length, len(data)):
                    X.append(data[i-seq_length:i])
                    y.append(target[i])
                return np.array(X), np.array(y)
            
            seq_length = 20
            X, y = create_sequences(sequence_data, target_data, seq_length)
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 转换为PyTorch张量
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train)
            X_test = torch.FloatTensor(X_test)
            y_test = torch.FloatTensor(y_test)
            
            # 创建LSTM模型
            class LSTMModel(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, output_size):
                    super(LSTMModel, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.fc(out[:, -1, :])
                    return out
            
            # 模型参数
            input_size = X_train.shape[2]
            hidden_size = 50
            num_layers = 2
            output_size = 1
            
            model = LSTMModel(input_size, hidden_size, num_layers, output_size)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # 训练模型
            num_epochs = 100
            for epoch in range(num_epochs):
                outputs = model(X_train)
                optimizer.zero_grad()
                loss = criterion(outputs.squeeze(), y_train)
                loss.backward()
                optimizer.step()
            
            # 评估模型
            model.eval()
            with torch.no_grad():
                predictions = model(X_test).squeeze()
                mse = mean_squared_error(y_test.numpy(), predictions.numpy())
                rmse = np.sqrt(mse)
            
            # 保存标准化器
            model_info = {
                'model': model,
                'scaler': scaler,
                'seq_length': seq_length,
                'feature_cols': feature_cols
            }
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            return model_info, metrics
            
        except ImportError:
            raise ImportError("PyTorch未安装，请安装: pip install torch")
        except Exception as e:
            logger.error(f"训练LSTM模型失败: {e}")
            raise
    
    async def _save_model(self, model: Any, model_id: int) -> str:
        """保存模型"""
        try:
            # 创建模型目录
            model_dir = os.path.join(settings.MODEL_CACHE_DIR, str(model_id))
            os.makedirs(model_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"model_{timestamp}.pkl"
            model_path = os.path.join(model_dir, model_filename)
            
            # 保存模型
            joblib.dump(model, model_path)
            
            return model_path
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise
    
    async def _load_model(self, model: Model) -> Any:
        """加载模型"""
        try:
            if not model.file_path or not os.path.exists(model.file_path):
                raise ValueError("模型文件不存在")
            
            # 尝试从缓存加载
            redis = await self._get_redis()
            cache_key = RedisKeys.model_cache(str(model.id))
            cached_model = await redis.get(cache_key)
            
            if cached_model:
                return pickle.loads(cached_model)
            
            # 从文件加载
            loaded_model = joblib.load(model.file_path)
            
            # 缓存模型
            await redis.setex(cache_key, 3600, pickle.dumps(loaded_model))  # 缓存1小时
            
            return loaded_model
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    async def _cache_model(self, model_id: int, model: Any):
        """缓存模型"""
        try:
            redis = await self._get_redis()
            cache_key = RedisKeys.model_cache(str(model_id))
            await redis.setex(cache_key, 3600, pickle.dumps(model))  # 缓存1小时
            
        except Exception as e:
            logger.error(f"缓存模型失败: {e}")
    
    async def _save_prediction(
        self,
        model_id: int,
        symbol: str,
        date: Optional[str],
        predictions: Any
    ) -> Dict[str, Any]:
        """保存预测结果"""
        try:
            # 获取股票ID
            result = await self.db.execute(select(Stock).where(Stock.symbol == symbol))
            stock = result.scalar_one_or_none()
            
            if not stock:
                raise ValueError(f"股票 {symbol} 不存在")
            
            # 创建预测记录
            prediction = Prediction(
                model_id=model_id,
                stock_id=stock.id,
                date=datetime.strptime(date, "%Y-%m-%d") if date else datetime.now(),
                prediction=float(predictions[0]),
                confidence=self._calculate_confidence(predictions)
            )
            
            self.db.add(prediction)
            await self.db.commit()
            
            return {
                'prediction_id': prediction.id,
                'symbol': symbol,
                'prediction': float(predictions[0]),
                'confidence': prediction.confidence
            }
            
        except Exception as e:
            logger.error(f"保存预测结果失败: {e}")
            raise
    
    def _calculate_confidence(self, predictions: Any) -> float:
        """计算预测置信度"""
        try:
            if hasattr(predictions, 'std'):
                # 如果有标准差信息
                std = predictions.std()
                confidence = max(0, 1 - std)
            else:
                # 默认置信度
                confidence = 0.8
            
            return float(confidence)
            
        except Exception:
            return 0.8