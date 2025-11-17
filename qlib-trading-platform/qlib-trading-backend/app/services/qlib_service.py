import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 添加qlib到Python路径
qlib_path = Path(__file__).parent.parent.parent.parent / "qlib"
if qlib_path.exists():
    sys.path.insert(0, str(qlib_path))

try:
    import qlib
    from qlib.data import D
    from qlib.data.ops import Ops
    from qlib.contrib.strategy import TopkDropoutStrategy, WeightStrategyBase
    from qlib.contrib.evaluate import backtest as normal_backtest
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.model.pytorch_lstm import LSTM
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.contrib.model.linear import LinearModel
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
except ImportError as e:
    logging.warning(f"Qlib导入失败: {e}")
    qlib = None

logger = logging.getLogger(__name__)


class QlibService:
    """Qlib集成服务"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "../../data")
        self.qlib_initialized = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """初始化Qlib"""
        try:
            if not qlib:
                raise ImportError("Qlib未安装")
            
            # 创建数据目录
            os.makedirs(self.data_dir, exist_ok=True)
            
            # 初始化Qlib
            qlib.init(provider_uri=self.data_dir, region="cn")
            self.qlib_initialized = True
            
            logger.info("Qlib初始化成功")
            
        except Exception as e:
            logger.error(f"Qlib初始化失败: {e}")
            raise
    
    async def prepare_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """准备Qlib数据"""
        try:
            if not self.qlib_initialized:
                await self.initialize()
            
            # 在后台线程中执行Qlib操作
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._prepare_data_sync,
                symbols, start_date, end_date
            )
            
            return result
            
        except Exception as e:
            logger.error(f"准备Qlib数据失败: {e}")
            raise
    
    def _prepare_data_sync(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """同步准备Qlib数据"""
        try:
            # 获取股票数据
            instruments = D.instruments(market="all")
            
            # 过滤指定的股票
            if symbols:
                instruments = [inst for inst in instruments if inst in symbols]
            
            # 获取价格数据
            price_fields = ["$open", "$high", "$low", "$close", "$volume"]
            price_data = D.features(instruments, price_fields, start_time=start_date, end_time=end_date)
            
            # 计算基础特征
            features = self._calculate_basic_features(price_data)
            
            return {
                "instruments": instruments,
                "price_data": price_data,
                "features": features,
                "start_date": start_date,
                "end_date": end_date
            }
            
        except Exception as e:
            logger.error(f"同步准备Qlib数据失败: {e}")
            raise
    
    def _calculate_basic_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """计算基础特征"""
        try:
            features = price_data.copy()
            
            # 收益率特征
            features["return_1d"] = features["$close"].pct_change()
            features["return_5d"] = features["$close"].pct_change(5)
            features["return_20d"] = features["$close"].pct_change(20)
            
            # 技术指标
            features["ma_5"] = features["$close"].rolling(5).mean()
            features["ma_20"] = features["$close"].rolling(20).mean()
            features["ma_ratio"] = features["ma_5"] / features["ma_20"]
            
            # 波动率
            features["volatility_20d"] = features["return_1d"].rolling(20).std()
            
            # 成交量特征
            features["volume_ma_20"] = features["$volume"].rolling(20).mean()
            features["volume_ratio"] = features["$volume"] / features["volume_ma_20"]
            
            # 价格位置
            features["price_position"] = (features["$close"] - features["$low"].rolling(20).min()) / \
                                       (features["$high"].rolling(20).max() - features["$low"].rolling(20).min())
            
            return features
            
        except Exception as e:
            logger.error(f"计算基础特征失败: {e}")
            raise
    
    async def train_model(
        self,
        model_type: str,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        features: List[str],
        target: str,
        model_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """训练Qlib模型"""
        try:
            if not self.qlib_initialized:
                await self.initialize()
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._train_model_sync,
                model_type, train_data, valid_data, features, target, model_params
            )
            
            return result
            
        except Exception as e:
            logger.error(f"训练Qlib模型失败: {e}")
            raise
    
    def _train_model_sync(
        self,
        model_type: str,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        features: List[str],
        target: str,
        model_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """同步训练Qlib模型"""
        try:
            # 准备数据集配置
            data_handler_config = {
                "class": "DataHandlerLP",
                "module_path": "qlib.data.dataset.handler",
                "kwargs": {
                    "instruments": "all",
                    "start_time": train_data.index.get_level_values('datetime').min(),
                    "end_time": valid_data.index.get_level_values('datetime').max(),
                    "data_loader": {
                        "class": "QlibDataLoader",
                        "module_path": "qlib.data.data",
                        "kwargs": {
                            "config": {
                                "feature": features,
                                "label": [target]
                            }
                        }
                    }
                }
            }
            
            # 创建数据集处理器
            handler = init_instance_by_config(data_handler_config)
            
            # 创建数据集
            dataset_config = {
                "class": "DatasetH",
                "kwargs": {
                    "handler": handler,
                    "segments": {
                        "train": (train_data.index.get_level_values('datetime').min(), 
                                 train_data.index.get_level_values('datetime').max()),
                        "valid": (valid_data.index.get_level_values('datetime').min(), 
                                 valid_data.index.get_level_values('datetime').max())
                    }
                }
            }
            
            dataset = init_instance_by_config(dataset_config)
            
            # 选择模型
            if model_type == "lgb":
                model_config = {
                    "class": "LGBModel",
                    "module_path": "qlib.contrib.model.gbdt",
                    "kwargs": model_params or {
                        "loss": "mse",
                        "colsample_bytree": 0.8879,
                        "learning_rate": 0.0421,
                        "subsample": 0.8789,
                        "lambda_l1": 205.6999,
                        "lambda_l2": 580.9768,
                        "max_depth": 8,
                        "num_leaves": 210,
                        "num_threads": 20
                    }
                }
            elif model_type == "lstm":
                model_config = {
                    "class": "LSTM",
                    "module_path": "qlib.contrib.model.pytorch_lstm",
                    "kwargs": model_params or {
                        "d_feat": 20,
                        "hidden_size": 64,
                        "num_layers": 2,
                        "dropout": 0.2,
                        "n_epochs": 200,
                        "lr": 0.001,
                        "early_stop": 10,
                        "batch_size": 800
                    }
                }
            elif model_type == "linear":
                model_config = {
                    "class": "LinearModel",
                    "module_path": "qlib.contrib.model.linear",
                    "kwargs": model_params or {}
                }
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 初始化模型
            model = init_instance_by_config(model_config)
            
            # 训练模型
            with R.start() as recorder:
                model.fit(dataset)
                
                # 保存模型
                model_path = os.path.join(self.data_dir, f"models/{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                
                # 评估模型
                train_pred = model.predict(dataset, segment="train")
                valid_pred = model.predict(dataset, segment="valid")
                
                # 计算指标
                train_metrics = self._calculate_metrics(train_pred, dataset, "train")
                valid_metrics = self._calculate_metrics(valid_pred, dataset, "valid")
                
                return {
                    "model_path": model_path,
                    "train_metrics": train_metrics,
                    "valid_metrics": valid_metrics,
                    "train_predictions": train_pred,
                    "valid_predictions": valid_pred,
                    "model_type": model_type
                }
                
        except Exception as e:
            logger.error(f"同步训练Qlib模型失败: {e}")
            raise
    
    def _calculate_metrics(self, predictions: pd.Series, dataset: DatasetH, segment: str) -> Dict[str, Any]:
        """计算模型指标"""
        try:
            # 获取真实标签
            labels = dataset.prepare(segment, col_set="label")
            
            # 计算IC（信息系数）
            ic = np.corrcoef(predictions.values, labels.values.flatten())[0, 1]
            
            # 计算Rank IC
            rank_ic = np.corrcoef(
                pd.Series(predictions.values).rank().values,
                pd.Series(labels.values.flatten()).rank().values
            )[0, 1]
            
            # 计算MSE
            mse = np.mean((predictions.values - labels.values.flatten()) ** 2)
            
            return {
                "ic": float(ic) if not np.isnan(ic) else 0.0,
                "rank_ic": float(rank_ic) if not np.isnan(rank_ic) else 0.0,
                "mse": float(mse),
                "mean_prediction": float(predictions.mean()),
                "std_prediction": float(predictions.std())
            }
            
        except Exception as e:
            logger.error(f"计算模型指标失败: {e}")
            return {"ic": 0.0, "rank_ic": 0.0, "mse": 0.0, "mean_prediction": 0.0, "std_prediction": 0.0}
    
    async def run_backtest(
        self,
        strategy_config: Dict[str, Any],
        start_date: str,
        end_date: str,
        initial_cash: float = 1000000.0
    ) -> Dict[str, Any]:
        """运行Qlib回测"""
        try:
            if not self.qlib_initialized:
                await self.initialize()
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_backtest_sync,
                strategy_config, start_date, end_date, initial_cash
            )
            
            return result
            
        except Exception as e:
            logger.error(f"运行Qlib回测失败: {e}")
            raise
    
    def _run_backtest_sync(
        self,
        strategy_config: Dict[str, Any],
        start_date: str,
        end_date: str,
        initial_cash: float = 1000000.0
    ) -> Dict[str, Any]:
        """同步运行Qlib回测"""
        try:
            # 策略配置
            strategy_conf = {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy",
                "kwargs": {
                    "topk": strategy_config.get("topk", 50),
                    "n_drop": strategy_config.get("n_drop", 5),
                    "risk_degree": strategy_config.get("risk_degree", 0.95),
                    "hold_thresh": strategy_config.get("hold_thresh", 1),
                    "only_tradable": strategy_config.get("only_tradable", True),
                    "market": strategy_config.get("market", "all")
                }
            }
            
            # 回测配置
            backtest_config = {
                "start_time": start_date,
                "end_time": end_date,
                "account": initial_cash,
                "benchmark": strategy_config.get("benchmark", "SH000300"),
                "exchange_kwargs": {
                    "freq": strategy_config.get("freq", "day"),
                    "limit_threshold": strategy_config.get("limit_threshold", 0.095),
                    "deal_price": strategy_config.get("deal_price", "close"),
                    "open_cost": strategy_config.get("open_cost", 0.0005),
                    "close_cost": strategy_config.get("close_cost", 0.0015),
                    "min_cost": strategy_config.get("min_cost", 5)
                }
            }
            
            # 初始化策略
            strategy = init_instance_by_config(strategy_conf)
            
            # 运行回测
            with R.start() as recorder:
                # 运行回测
                portfolio_metrics, indicator_metrics = normal_backtest(
                    executor_config=backtest_config,
                    strategy=strategy,
                    benchmark=backtest_config["benchmark"]
                )
                
                # 风险分析
                risk_analysis_result = risk_analysis(indicator_metrics["returns"])
                
                # 保存结果
                results = {
                    "portfolio_metrics": portfolio_metrics,
                    "indicator_metrics": indicator_metrics,
                    "risk_analysis": risk_analysis_result,
                    "strategy_config": strategy_config,
                    "backtest_config": backtest_config
                }
                
                # 保存到文件
                results_path = os.path.join(self.data_dir, f"backtests/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                
                with open(results_path, "wb") as f:
                    pickle.dump(results, f)
                
                return {
                    "results_path": results_path,
                    "summary": {
                        "total_return": float(indicator_metrics["returns"].sum()),
                        "annual_return": float(indicator_metrics["returns"].mean() * 252),
                        "volatility": float(indicator_metrics["returns"].std() * np.sqrt(252)),
                        "sharpe_ratio": float(indicator_metrics["returns"].mean() / indicator_metrics["returns"].std() * np.sqrt(252)),
                        "max_drawdown": float(risk_analysis_result["max_drawdown"].iloc[0]),
                        "win_rate": float((indicator_metrics["returns"] > 0).mean())
                    }
                }
                
        except Exception as e:
            logger.error(f"同步运行Qlib回测失败: {e}")
            raise
    
    async def get_alpha_factors(self, symbols: List[str], date: str, factors: List[str] = None) -> Dict[str, Any]:
        """获取Alpha因子"""
        try:
            if not self.qlib_initialized:
                await self.initialize()
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._get_alpha_factors_sync,
                symbols, date, factors
            )
            
            return result
            
        except Exception as e:
            logger.error(f"获取Alpha因子失败: {e}")
            raise
    
    def _get_alpha_factors_sync(self, symbols: List[str], date: str, factors: List[str] = None) -> Dict[str, Any]:
        """同步获取Alpha因子"""
        try:
            # 默认因子
            if not factors:
                factors = [
                    "Return($close, 5)",  # 5日收益率
                    "Return($close, 20)",  # 20日收益率
                    "Mean($close, 5) / $close",  # 5日均线比率
                    "Mean($close, 20) / $close",  # 20日均线比率
                    "Std($close, 20)",  # 20日标准差
                    "Corr($close, Log($volume + 1), 20)",  # 价格和成交量的相关性
                    "Mean($volume, 5) / $volume",  # 5日成交量均线比率
                    "Rank($close, 20)",  # 20日价格排名
                ]
            
            # 获取因子数据
            factor_data = D.features(symbols, factors, start_time=date, end_time=date)
            
            # 处理数据
            result = {}
            for symbol in symbols:
                symbol_data = factor_data.loc[symbol] if symbol in factor_data.index else None
                if symbol_data is not None:
                    result[symbol] = {
                        factor: float(symbol_data[factor]) if not pd.isna(symbol_data[factor]) else 0.0
                        for factor in factors
                    }
                else:
                    result[symbol] = {factor: 0.0 for factor in factors}
            
            return {
                "date": date,
                "factors": factors,
                "factor_data": result
            }
            
        except Exception as e:
            logger.error(f"同步获取Alpha因子失败: {e}")
            raise
    
    async def cleanup(self):
        """清理资源"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
                
            logger.info("Qlib服务清理完成")
            
        except Exception as e:
            logger.error(f"清理Qlib服务失败: {e}")


# 创建全局实例
qlib_service = QlibService()