# Qlib Trading Platform

基于Qlib的现代化量化交易平台，提供股票查询、模型训练、回测分析和数据管理功能。

## 功能特性

### 🚀 核心功能
- **股票数据管理**: 支持多数据源（Yahoo Finance、Akshare、Tushare）
- **机器学习模型**: LightGBM、XGBoost、LSTM等模型训练
- **策略回测**: 多种策略（买入持有、动量、均值回归、机器学习）
- **投资组合管理**: 实时跟踪和绩效分析
- **数据可视化**: 交互式图表和仪表板

### 📊 技术栈
- **前端**: Vue 3 + TypeScript + Element Plus + Tailwind CSS
- **后端**: FastAPI + SQLAlchemy + Pydantic
- **数据库**: PostgreSQL + Redis
- **量化分析**: Qlib + Pandas + NumPy + Scikit-learn
- **部署**: Docker + Docker Compose

## 快速开始

### 环境要求
- Docker 和 Docker Compose
- Python 3.11+ (开发环境)
- Node.js 18+ (前端开发)

### 1. 克隆项目
```bash
git clone https://github.com/your-repo/qlib-trading-platform.git
cd qlib-trading-platform
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，设置必要的配置
```

### 3. 使用Docker启动
```bash
docker-compose up -d
```

### 4. 访问应用
- 前端: http://localhost
- 后端API: http://localhost:8000
- API文档: http://localhost:8000/docs

## 开发指南

### 后端开发

#### 安装依赖
```bash
cd qlib-trading-backend
pip install -r requirements.txt
```

#### 数据库迁移
```bash
alembic init alembic
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head
```

#### 启动开发服务器
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 前端开发

#### 安装依赖
```bash
cd qlib-trading-platform
npm install
```

#### 启动开发服务器
```bash
npm run dev
```

#### 构建生产版本
```bash
npm run build
```

## API文档

### 认证相关
- `POST /api/v1/auth/register` - 用户注册
- `POST /api/v1/auth/login` - 用户登录
- `POST /api/v1/auth/logout` - 用户登出

### 股票数据
- `GET /api/v1/stocks/` - 获取股票列表
- `GET /api/v1/stocks/{symbol}` - 获取股票详情
- `GET /api/v1/stocks/{symbol}/data` - 获取股票历史数据
- `POST /api/v1/stocks/{symbol}/sync` - 同步股票数据

### 机器学习模型
- `GET /api/v1/models/` - 获取模型列表
- `POST /api/v1/models/` - 创建模型
- `POST /api/v1/models/{model_id}/train` - 训练模型
- `POST /api/v1/models/{model_id}/predict` - 模型预测

### 回测分析
- `GET /api/v1/backtest/` - 获取回测列表
- `POST /api/v1/backtest/` - 创建回测
- `GET /api/v1/backtest/{backtest_id}` - 获取回测结果
- `POST /api/v1/backtest/{backtest_id}/run` - 运行回测

### 投资组合
- `GET /api/v1/portfolio/` - 获取投资组合列表
- `POST /api/v1/portfolio/` - 创建投资组合
- `GET /api/v1/portfolio/{portfolio_id}` - 获取投资组合详情
- `POST /api/v1/portfolio/{portfolio_id}/stocks` - 添加股票

### 数据管理
- `GET /api/v1/data/tasks` - 获取数据任务列表
- `POST /api/v1/data/tasks` - 创建数据更新任务
- `GET /api/v1/data/quality` - 获取数据质量报告

## 配置说明

### 环境变量

#### 数据库配置
```bash
POSTGRES_DB=qlib_trading
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/qlib_trading
```

#### Redis配置
```bash
REDIS_URL=redis://localhost:6379/0
```

#### 安全密钥
```bash
SECRET_KEY=your_very_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

#### 数据源配置
```bash
TUSHARE_TOKEN=your_tushare_token
AKSHARE_ENABLE=true
YFINANCE_ENABLE=true
```

#### Qlib配置
```bash
QLIB_DATA_PATH=~/.qlib/qlib_data/cn_data
QLIB_REGION=cn
```

## 部署指南

### 生产环境部署

1. **准备服务器**
   - 安装Docker和Docker Compose
   - 配置域名和SSL证书

2. **配置环境变量**
   ```bash
   # 编辑 .env 文件，设置生产环境配置
   DEBUG=false
   SECRET_KEY=your_production_secret_key
   ```

3. **构建和启动服务**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

4. **配置反向代理**
   - 配置Nginx或Apache
   - 设置SSL证书
   - 配置负载均衡（可选）

### 监控和维护

#### 日志查看
```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend
```

#### 数据备份
```bash
# 备份数据库
docker-compose exec postgres pg_dump -U postgres qlib_trading > backup.sql

# 备份Redis
docker-compose exec redis redis-cli save
docker cp $(docker-compose ps -q redis):/data/dump.rdb ./redis_backup.rdb
```

#### 性能监控
- 使用Prometheus和Grafana进行监控
- 配置告警规则
- 定期查看性能指标

## 开发计划

### 已完成功能
- ✅ 基础项目架构
- ✅ 用户认证系统
- ✅ 股票数据管理
- ✅ 机器学习模型训练
- ✅ 策略回测系统
- ✅ 投资组合管理
- ✅ 数据可视化
- ✅ Docker容器化

### 待开发功能
- 📋 实时数据推送（WebSocket）
- 📋 高级图表分析
- 📋 多因子模型
- 📋 风险管理系统
- 📋 移动端适配
- 📋 多语言支持
- 📋 社交功能（策略分享）

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 项目维护者: [Your Name](mailto:your.email@example.com)
- 项目主页: https://github.com/your-repo/qlib-trading-platform
- 问题反馈: https://github.com/your-repo/qlib-trading-platform/issues

## 致谢

- [Microsoft Qlib](https://github.com/microsoft/qlib) - 量化投资平台
- [FastAPI](https://fastapi.tiangolo.com/) - 现代Web框架
- [Vue.js](https://vuejs.org/) - 渐进式JavaScript框架
- [Element Plus](https://element-plus.org/) - Vue组件库

---

**⭐ 如果这个项目对你有帮助，请给个Star！**