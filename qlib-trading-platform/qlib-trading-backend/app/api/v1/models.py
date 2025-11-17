from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, HTTPBearer
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from typing import List, Optional, Dict, Any
import json
import os
import uuid
import logging

from app.core.database import get_db
from app.models import Model, User
from app.schemas import (
    ModelCreate, ModelInDB, ModelListResponse, ModelResponse,
    ModelTrainRequest, ModelTrainResponse, ResponseBase
)
from app.core.security import verify_token
from app.services.model_service import ModelService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    """获取当前用户"""
    token_data = verify_token(credentials.credentials)
    return token_data


@router.get("/", response_model=ModelListResponse)
async def get_models(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    model_type: Optional[str] = Query(None, description="模型类型"),
    is_active: Optional[bool] = Query(None, description="是否活跃"),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """获取模型列表"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        # 构建查询条件
        conditions = [Model.user_id == user_id]
        if model_type:
            conditions.append(Model.model_type == model_type)
        if is_active is not None:
            conditions.append(Model.is_active == is_active)
        
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 查询总数
        count_stmt = select(Model).where(and_(*conditions))
        total_result = await db.execute(select(Model).where(and_(*conditions)))
        total = len(total_result.scalars().all())
        
        # 查询数据
        stmt = select(Model).where(and_(*conditions)).offset(offset).limit(page_size)
        result = await db.execute(stmt)
        models = result.scalars().all()
        
        return ModelListResponse(
            success=True,
            message="获取模型列表成功",
            data=models,
            total=total,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取模型列表失败")


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model_detail(
    model_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """获取模型详情"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        result = await db.execute(
            select(Model).where(and_(Model.id == model_id, Model.user_id == user_id))
        )
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        return ModelResponse(success=True, message="获取模型详情成功", data=model)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型详情失败: {e}")
        raise HTTPException(status_code=500, detail="获取模型详情失败")


@router.post("/", response_model=ModelResponse)
async def create_model(
    model: ModelCreate,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """创建模型"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        # 创建新模型
        db_model = Model(
            **model.dict(),
            user_id=user_id,
            is_trained=False,
            is_active=True
        )
        
        db.add(db_model)
        await db.commit()
        await db.refresh(db_model)
        
        return ModelResponse(success=True, message="创建模型成功", data=db_model)
    except Exception as e:
        logger.error(f"创建模型失败: {e}")
        raise HTTPException(status_code=500, detail="创建模型失败")


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: int,
    model_update: ModelCreate,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """更新模型"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        result = await db.execute(
            select(Model).where(and_(Model.id == model_id, Model.user_id == user_id))
        )
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        # 更新字段
        for field, value in model_update.dict(exclude_unset=True).items():
            setattr(model, field, value)
        
        await db.commit()
        await db.refresh(model)
        
        return ModelResponse(success=True, message="更新模型成功", data=model)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新模型失败: {e}")
        raise HTTPException(status_code=500, detail="更新模型失败")


@router.delete("/{model_id}", response_model=ResponseBase)
async def delete_model(
    model_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """删除模型"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        result = await db.execute(
            select(Model).where(and_(Model.id == model_id, Model.user_id == user_id))
        )
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        # 删除模型文件
        if model.file_path and os.path.exists(model.file_path):
            os.remove(model.file_path)
        
        await db.delete(model)
        await db.commit()
        
        return ResponseBase(success=True, message="删除模型成功")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除模型失败: {e}")
        raise HTTPException(status_code=500, detail="删除模型失败")


@router.post("/{model_id}/train", response_model=ModelTrainResponse)
async def train_model(
    model_id: int,
    train_request: ModelTrainRequest,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """训练模型"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        result = await db.execute(
            select(Model).where(and_(Model.id == model_id, Model.user_id == user_id))
        )
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        # 创建模型服务
        model_service = ModelService(db)
        
        # 开始训练
        training_result = await model_service.train_model(
            model=model,
            start_date=train_request.start_date,
            end_date=train_request.end_date,
            symbols=train_request.symbols,
            hyperparameters=train_request.hyperparameters
        )
        
        return ModelTrainResponse(
            success=True,
            message="模型训练已启动",
            data=training_result
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"训练模型失败: {e}")
        raise HTTPException(status_code=500, detail="训练模型失败")


@router.post("/{model_id}/predict", response_model=ResponseBase)
async def predict_with_model(
    model_id: int,
    symbol: str = Query(..., description="股票代码"),
    date: Optional[str] = Query(None, description="预测日期 (YYYY-MM-DD)"),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """使用模型进行预测"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        result = await db.execute(
            select(Model).where(and_(Model.id == model_id, Model.user_id == user_id))
        )
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        if not model.is_trained:
            raise HTTPException(status_code=400, detail="模型尚未训练")
        
        # 创建模型服务
        model_service = ModelService(db)
        
        # 进行预测
        prediction = await model_service.predict(
            model=model,
            symbol=symbol,
            date=date
        )
        
        return ResponseBase(
            success=True,
            message="预测成功",
            data=prediction
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail="预测失败")


@router.get("/{model_id}/performance", response_model=ResponseBase)
async def get_model_performance(
    model_id: int,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """获取模型性能指标"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        result = await db.execute(
            select(Model).where(and_(Model.id == model_id, Model.user_id == user_id))
        )
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        return ResponseBase(
            success=True,
            message="获取模型性能成功",
            data=model.performance_metrics
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型性能失败: {e}")
        raise HTTPException(status_code=500, detail="获取模型性能失败")


@router.post("/{model_id}/upload", response_model=ResponseBase)
async def upload_model_file(
    model_id: int,
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """上传模型文件"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        result = await db.execute(
            select(Model).where(and_(Model.id == model_id, Model.user_id == user_id))
        )
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        # 检查文件类型
        if not file.filename.endswith(('.pkl', '.joblib', '.h5', '.pt', '.pth')):
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        
        # 创建模型目录
        model_dir = os.path.join(settings.MODEL_CACHE_DIR, str(user_id))
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存文件
        file_extension = os.path.splitext(file.filename)[1]
        file_name = f"{model_id}_{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(model_dir, file_name)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 更新模型文件路径
        model.file_path = file_path
        await db.commit()
        
        return ResponseBase(success=True, message="模型文件上传成功")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传模型文件失败: {e}")
        raise HTTPException(status_code=500, detail="上传模型文件失败")