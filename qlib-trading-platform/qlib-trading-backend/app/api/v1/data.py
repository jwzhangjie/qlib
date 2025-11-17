from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, HTTPBearer
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from typing import List, Optional, Dict, Any
import logging
import pandas as pd
from datetime import datetime, timedelta
import io

from app.core.database import get_db
from app.models import DataUpdateTask, Stock, StockPrice
from app.schemas import ResponseBase
from app.core.security import verify_token
from app.services.data_service import DataService
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/stats", response_model=ResponseBase)
async def get_data_stats(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """获取数据统计"""
    try:
        # 获取股票总数
        stock_count_result = await db.execute(select(func.count(Stock.id)))
        total_stocks = stock_count_result.scalar()
        
        # 获取更新任务数
        task_count_result = await db.execute(
            select(func.count(DataUpdateTask.id)).where(DataUpdateTask.status.in_(['pending', 'running']))
        )
        update_tasks = task_count_result.scalar()
        
        # 获取最后更新时间
        last_update_result = await db.execute(
            select(func.max(StockPrice.created_at))
        )
        last_update = last_update_result.scalar()
        
        # 获取数据源状态
        data_service = DataService(db)
        data_source_status = await data_service.get_data_source_status()
        
        return ResponseBase(
            success=True,
            message="获取数据统计成功",
            data={
                "totalStocks": total_stocks,
                "updateTasks": update_tasks,
                "lastUpdate": last_update.strftime("%Y-%m-%d") if last_update else "N/A",
                "dataSourceStatus": data_source_status.get("status", "unknown")
            }
        )
    except Exception as e:
        logger.error(f"获取数据统计失败: {e}")
        raise HTTPException(status_code=500, detail="获取数据统计失败")


@router.get("/list", response_model=ResponseBase)
async def get_data_list(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    market: Optional[str] = Query(None, description="市场代码"),
    status: Optional[str] = Query(None, description="数据状态"),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """获取数据列表"""
    try:
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建查询条件
        conditions = []
        if search:
            conditions.append(
                or_(
                    Stock.symbol.contains(search),
                    Stock.name.contains(search)
                )
            )
        if market:
            conditions.append(Stock.market == market)
        if status:
            conditions.append(Stock.is_active == (status == "active"))
        
        # 查询总数
        count_stmt = select(func.count(Stock.id)).where(and_(*conditions))
        total_result = await db.execute(count_stmt)
        total = total_result.scalar()
        
        # 查询数据
        stmt = select(Stock).where(and_(*conditions)).offset(offset).limit(page_size)
        result = await db.execute(stmt)
        stocks = result.scalars().all()
        
        # 获取每个股票的数据统计
        data_list = []
        for stock in stocks:
            # 获取数据记录数
            price_count_result = await db.execute(
                select(func.count(StockPrice.id)).where(StockPrice.stock_id == stock.id)
            )
            record_count = price_count_result.scalar()
            
            # 获取数据范围
            date_range_result = await db.execute(
                select(
                    func.min(StockPrice.date),
                    func.max(StockPrice.date)
                ).where(StockPrice.stock_id == stock.id)
            )
            date_range = date_range_result.first()
            
            data_list.append({
                "symbol": stock.symbol,
                "name": stock.name,
                "market": stock.market,
                "dataRange": f"{date_range[0].strftime('%Y-%m-%d')} 至 {date_range[1].strftime('%Y-%m-%d')}" if date_range and date_range[0] else "无数据",
                "lastUpdate": stock.updated_at.strftime("%Y-%m-%d %H:%M:%S") if stock.updated_at else "N/A",
                "dataQuality": "优秀" if record_count > 1000 else "良好" if record_count > 500 else "一般",
                "status": "正常" if stock.is_active else "停用",
                "recordCount": record_count
            })
        
        return ResponseBase(
            success=True,
            message="获取数据列表成功",
            data={
                "items": data_list,
                "total": total,
                "page": page,
                "page_size": page_size
            }
        )
    except Exception as e:
        logger.error(f"获取数据列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取数据列表失败")


@router.post("/update", response_model=ResponseBase)
async def update_data(
    scope: str = Query(..., description="更新范围: all, selected, custom"),
    symbols: Optional[List[str]] = Query(None, description="股票代码列表"),
    data_types: List[str] = Query(..., description="数据类型"),
    start_date: Optional[str] = Query(None, description="开始日期 (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="结束日期 (YYYY-MM-DD)"),
    frequency: str = Query("daily", description="更新频率: daily, minute, tick"),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """更新数据"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        # 创建数据更新任务
        data_service = DataService(db)
        task = await data_service.create_update_task(
            user_id=user_id,
            name=f"数据更新任务_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            scope=scope,
            symbols=symbols,
            data_types=data_types,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency
        )
        
        # 启动更新任务（这里简化处理，实际应该使用Celery）
        try:
            update_result = await data_service.execute_update_task(task)
            
            return ResponseBase(
                success=True,
                message="数据更新任务已启动",
                data={
                    "task_id": task.task_id,
                    "status": update_result.get("status", "completed"),
                    "processed_items": update_result.get("processed_items", 0),
                    "total_items": update_result.get("total_items", 0)
                }
            )
        except Exception as e:
            # 更新任务状态为失败
            task.status = "failed"
            task.error_message = str(e)
            await db.commit()
            
            raise HTTPException(status_code=500, detail=f"数据更新失败: {str(e)}")
            
    except Exception as e:
        logger.error(f"更新数据失败: {e}")
        raise HTTPException(status_code=500, detail="更新数据失败")


@router.post("/import", response_model=ResponseBase)
async def import_data(
    file: UploadFile = File(...),
    data_type: str = Query(..., description="数据类型"),
    format: str = Query(..., description="文件格式: csv, excel, json"),
    date_column: Optional[str] = Query(None, description="日期列名"),
    symbol_column: Optional[str] = Query(None, description="股票代码列名"),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """导入数据"""
    try:
        # 检查文件类型
        if not file.filename.endswith(('.csv', '.xlsx', '.json')):
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        
        # 读取文件内容
        content = await file.read()
        
        # 根据格式解析数据
        if format == 'csv':
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif format == 'excel':
            df = pd.read_excel(io.BytesIO(content))
        elif format == 'json':
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        # 导入数据
        data_service = DataService(db)
        import_result = await data_service.import_data(
            df=df,
            data_type=data_type,
            date_column=date_column,
            symbol_column=symbol_column
        )
        
        return ResponseBase(
            success=True,
            message="数据导入成功",
            data=import_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导入数据失败: {e}")
        raise HTTPException(status_code=500, detail="导入数据失败")


@router.post("/export", response_model=ResponseBase)
async def export_data(
    symbols: List[str] = Query(..., description="股票代码列表"),
    data_types: List[str] = Query(..., description="数据类型"),
    start_date: str = Query(..., description="开始日期 (YYYY-MM-DD)"),
    end_date: str = Query(..., description="结束日期 (YYYY-MM-DD)"),
    format: str = Query("csv", description="导出格式: csv, excel, json"),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """导出数据"""
    try:
        # 导出数据
        data_service = DataService(db)
        export_result = await data_service.export_data(
            symbols=symbols,
            data_types=data_types,
            start_date=start_date,
            end_date=end_date,
            format=format
        )
        
        return ResponseBase(
            success=True,
            message="数据导出成功",
            data=export_result
        )
        
    except Exception as e:
        logger.error(f"导出数据失败: {e}")
        raise HTTPException(status_code=500, detail="导出数据失败")


@router.get("/{symbol}/detail", response_model=ResponseBase)
async def get_data_detail(
    symbol: str,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """获取数据详情"""
    try:
        # 获取股票信息
        result = await db.execute(select(Stock).where(Stock.symbol == symbol))
        stock = result.scalar_one_or_none()
        
        if not stock:
            raise HTTPException(status_code=404, detail="股票不存在")
        
        # 获取数据记录数
        price_count_result = await db.execute(
            select(func.count(StockPrice.id)).where(StockPrice.stock_id == stock.id)
        )
        record_count = price_count_result.scalar()
        
        # 获取数据范围
        date_range_result = await db.execute(
            select(
                func.min(StockPrice.date),
                func.max(StockPrice.date)
            ).where(StockPrice.stock_id == stock.id)
        )
        date_range = date_range_result.first()
        
        # 获取最近数据预览
        preview_result = await db.execute(
            select(StockPrice).where(StockPrice.stock_id == stock.id)
            .order_by(StockPrice.date.desc()).limit(10)
        )
        preview_data = preview_result.scalars().all()
        
        preview_list = [
            {
                "date": price.date.strftime("%Y-%m-%d"),
                "open": price.open,
                "high": price.high,
                "low": price.low,
                "close": price.close,
                "volume": price.volume
            }
            for price in preview_data
        ]
        
        return ResponseBase(
            success=True,
            message="获取数据详情成功",
            data={
                "symbol": stock.symbol,
                "name": stock.name,
                "market": stock.market,
                "dataRange": f"{date_range[0].strftime('%Y-%m-%d')} 至 {date_range[1].strftime('%Y-%m-%d')}" if date_range and date_range[0] else "无数据",
                "lastUpdate": stock.updated_at.strftime("%Y-%m-%d %H:%M:%S") if stock.updated_at else "N/A",
                "dataQuality": "优秀" if record_count > 1000 else "良好" if record_count > 500 else "一般",
                "status": "正常" if stock.is_active else "停用",
                "recordCount": record_count,
                "previewData": preview_list
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据详情失败: {e}")
        raise HTTPException(status_code=500, detail="获取数据详情失败")


@router.delete("/{symbol}", response_model=ResponseBase)
async def delete_stock_data(
    symbol: str,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """删除股票数据"""
    try:
        # 获取当前用户
        token_data = verify_token(credentials.credentials)
        user_id = token_data["user_id"]
        
        # 获取股票信息
        result = await db.execute(select(Stock).where(Stock.symbol == symbol))
        stock = result.scalar_one_or_none()
        
        if not stock:
            raise HTTPException(status_code=404, detail="股票不存在")
        
        # 删除相关的价格数据
        await db.execute(
            select(StockPrice).where(StockPrice.stock_id == stock.id).delete()
        )
        
        # 删除股票
        await db.delete(stock)
        await db.commit()
        
        return ResponseBase(success=True, message="删除股票数据成功")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除股票数据失败: {e}")
        raise HTTPException(status_code=500, detail="删除股票数据失败")


@router.get("/tasks", response_model=ResponseBase)
async def get_update_tasks(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    status: Optional[str] = Query(None, description="任务状态"),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: AsyncSession = Depends(get_db)
):
    """获取更新任务列表"""
    try:
        # 计算偏移量
        offset = (page - 1) * page_size
        
        # 构建查询条件
        conditions = []
        if status:
            conditions.append(DataUpdateTask.status == status)
        
        # 查询总数
        count_stmt = select(func.count(DataUpdateTask.id)).where(and_(*conditions))
        total_result = await db.execute(count_stmt)
        total = total_result.scalar()
        
        # 查询数据
        stmt = select(DataUpdateTask).where(and_(*conditions)).offset(offset).limit(page_size)
        stmt = stmt.order_by(DataUpdateTask.created_at.desc())
        result = await db.execute(stmt)
        tasks = result.scalars().all()
        
        task_list = [
            {
                "id": task.id,
                "task_id": task.task_id,
                "name": task.name,
                "status": task.status,
                "progress": task.progress,
                "total_items": task.total_items,
                "processed_items": task.processed_items,
                "created_at": task.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "started_at": task.started_at.strftime("%Y-%m-%d %H:%M:%S") if task.started_at else None,
                "completed_at": task.completed_at.strftime("%Y-%m-%d %H:%M:%S") if task.completed_at else None
            }
            for task in tasks
        ]
        
        return ResponseBase(
            success=True,
            message="获取更新任务列表成功",
            data={
                "tasks": task_list,
                "total": total,
                "page": page,
                "page_size": page_size
            }
        )
    except Exception as e:
        logger.error(f"获取更新任务列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取更新任务列表失败")