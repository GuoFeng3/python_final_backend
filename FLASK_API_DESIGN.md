# Flask API 设计文档

## 1. 项目概述

本项目采用 Flask 框架构建 RESTful API，实现房屋交易数据的统计分析与预测功能。API 提供数据查询、统计分析、趋势预测等功能，支持多维度数据筛选与聚合。

## 2. 技术栈

- **Web框架**: Flask 2.0+
- **ORM**: SQLAlchemy
- **数据库**: MySQL/PostgreSQL
- **数据处理**: NumPy, Pandas
- **部署**: Gunicorn + Nginx

## 3. 项目结构

```
flask_house_api/
├── app/
│   ├── __init__.py           # 应用初始化
│   ├── models.py            # 数据模型定义
│   ├── routes/              # 路由模块
│   │   ├── __init__.py
│   │   ├── data.py          # 数据导入路由
│   │   ├── statistics.py    # 统计分析路由
│   │   └── prediction.py    # 预测路由
│   ├── services/            # 业务逻辑层
│   │   ├── __init__.py
│   │   ├── data_service.py  # 数据处理服务
│   │   ├── stats_service.py # 统计分析服务
│   │   └── predict_service.py # 预测服务
│   └── utils/               # 工具函数
│       ├── __init__.py
│       └── helpers.py
├── config.py                # 配置文件
├── run.py                   # 应用入口
├── requirements.txt         # 依赖项
└── README.md                # 项目说明
```

## 4. 数据模型

基于 Django 模型转换为 SQLAlchemy 模型：

```python
# app/models.py
from datetime import date
from sqlalchemy import Column, Integer, String, Float, Date, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class HouseDeal(Base):
    __tablename__ = 'house_deal'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    house_id = Column(String(50), index=True, nullable=False)  # 房屋唯一标识ID
    title = Column(String(200), nullable=False)  # 房屋标题
    city = Column(String(50), default="北京", index=True, nullable=False)  # 城市
    district = Column(String(50), index=True, nullable=False)  # 行政区
    district_area = Column(String(100), index=True, nullable=False)  # 细分区域
    layout = Column(String(50), index=True, nullable=False)  # 户型
    area = Column(Float, nullable=False)  # 房屋面积（平方米）
    floor = Column(String(50), nullable=False)  # 楼层信息
    direction = Column(String(20), index=True, nullable=False)  # 房屋朝向
    deal_price = Column(Integer, nullable=False)  # 成交总价（万元）
    deal_unit_price = Column(String(50))  # 成交单价
    deal_date = Column(Date, index=True, nullable=False)  # 成交日期
    
    # 联合索引
    __table_args__ = (
        Index('idx_district_layout', 'district', 'layout'),
        Index('idx_district_area_date', 'district_area', 'deal_date'),
    )
```

## 5. API 接口设计

### 5.1 基础配置

- **API前缀**: `/api/v1`
- **请求格式**: JSON
- **响应格式**: JSON
- **错误处理**: 统一返回错误码与错误信息

### 5.2 数据导入接口

#### 5.2.1 批量导入房屋交易数据

- **URL**: `/api/v1/data/import`
- **方法**: `POST`
- **参数**:
  - `file`: 上传的 CSV/XLSX 文件
  - `city` (可选): 城市名称（默认：北京）
- **响应**:
  ```json
  {
    "code": 200,
    "message": "成功导入 100 条数据",
    "data": {
      "imported_count": 100
    }
  }
  ```

### 5.3 统计分析接口

#### 5.3.1 整体数据统计

- **URL**: `/api/v1/statistics/total`
- **方法**: `GET`
- **参数**:
  - `city` (可选): 城市名称
  - `district` (可选): 行政区名称（默认：all）
- **响应**:
  ```json
  {
    "code": 200,
    "message": "success",
    "data": {
      "avg_unit_price": 65000.50,
      "median_unit_price": 62000.00,
      "avg_deal_price": 450.25,
      "median_deal_price": 420.00,
      "avg_area": 78.50,
      "deal_count": 1500
    }
  }
  ```

#### 5.3.2 各城区平均单价排名

- **URL**: `/api/v1/statistics/district-rank`
- **方法**: `GET`
- **参数**:
  - `city` (可选): 城市名称
- **响应**:
  ```json
  {
    "code": 200,
    "message": "success",
    "data": [
      {"district": "东城", "unit_price": 85000.00},
      {"district": "西城", "unit_price": 82000.00},
      {"district": "海淀", "unit_price": 78000.00}
    ]
  }
  ```

#### 5.3.3 月度平均单价趋势

- **URL**: `/api/v1/statistics/month-trend`
- **方法**: `GET`
- **参数**:
  - `city` (可选): 城市名称
  - `district` (可选): 行政区名称（默认：all）
- **响应**:
  ```json
  {
    "code": 200,
    "message": "success",
    "data": [
      {"month": "2023-01", "avg_unit_price": 65000.00},
      {"month": "2023-02", "avg_unit_price": 65500.00},
      {"month": "2023-03", "avg_unit_price": 66000.00}
    ]
  }
  ```

#### 5.3.4 季度平均单价趋势

- **URL**: `/api/v1/statistics/quarter-trend`
- **方法**: `GET`
- **参数**:
  - `city` (可选): 城市名称
  - `district` (可选): 行政区名称（默认：all）
- **响应**:
  ```json
  {
    "code": 200,
    "message": "success",
    "data": [
      {"quarter": "2023Q1", "avg_unit_price": 65500.00},
      {"quarter": "2023Q2", "avg_unit_price": 66500.00},
      {"quarter": "2023Q3", "avg_unit_price": 67000.00}
    ]
  }
  ```

#### 5.3.5 细分区域房价排名

- **URL**: `/api/v1/statistics/district-area-rank`
- **方法**: `GET`
- **参数**:
  - `city` (可选): 城市名称
  - `district` (可选): 行政区名称（默认：all）
- **响应**:
  ```json
  {
    "code": 200,
    "message": "success",
    "data": [
      {"district_area": "长阳", "avg_unit_price": 58000.00},
      {"district_area": "回龙观", "avg_unit_price": 55000.00},
      {"district_area": "天通苑", "avg_unit_price": 52000.00}
    ]
  }
  ```

#### 5.3.6 房屋朝向价格分析

- **URL**: `/api/v1/statistics/direction-price`
- **方法**: `GET`
- **参数**:
  - `city` (可选): 城市名称
  - `district` (可选): 行政区名称（默认：all）
- **响应**:
  ```json
  {
    "code": 200,
    "message": "success",
    "data": [
      {"direction": "南", "avg_unit_price": 68000.00},
      {"direction": "南北", "avg_unit_price": 66000.00},
      {"direction": "东", "avg_unit_price": 64000.00}
    ]
  }
  ```

#### 5.3.7 面积区间价格分析

- **URL**: `/api/v1/statistics/squaremeter-avgprice`
- **方法**: `GET`
- **参数**:
  - `city` (可选): 城市名称
  - `district` (可选): 行政区名称（默认：all）
- **响应**:
  ```json
  {
    "code": 200,
    "message": "success",
    "data": [
      {"area_range": "50㎡以下", "avg_unit_price": 72000.00, "avg_total_price": 320.00, "count": 150},
      {"area_range": "50-70㎡", "avg_unit_price": 70000.00, "avg_total_price": 420.00, "count": 280},
      {"area_range": "70-90㎡", "avg_unit_price": 68000.00, "avg_total_price": 540.00, "count": 420}
    ]
  }
  ```

#### 5.3.8 户型成交量与价格分析

- **URL**: `/api/v1/statistics/room-hall`
- **方法**: `GET`
- **参数**:
  - `city` (可选): 城市名称
  - `district` (可选): 行政区名称（默认：all）
- **响应**:
  ```json
  {
    "code": 200,
    "message": "success",
    "data": [
      {"layout": "1室1厅", "avg_unit_price": 70000.00, "count": 210},
      {"layout": "2室1厅", "avg_unit_price": 68000.00, "count": 520},
      {"layout": "3室2厅", "avg_unit_price": 66000.00, "count": 380}
    ]
  }
  ```

### 5.4 预测接口

#### 5.4.1 未来三个月房价预测

- **URL**: `/api/v1/prediction/future-prices`
- **方法**: `GET`
- **参数**:
  - `city` (可选): 城市名称
  - `district` (可选): 行政区名称（默认：all）
  - `layout` (可选): 户型（默认：all）
- **响应**:
  ```json
  {
    "code": 200,
    "message": "success",
    "data": [
      {"month": "2023-10", "predicted_price": 67500.00},
      {"month": "2023-11", "predicted_price": 68000.00},
      {"month": "2023-12", "predicted_price": 68500.00}
    ]
  }
  ```

## 6. 依赖项

```txt
# requirements.txt
Flask==2.3.2
Flask-RESTful==0.3.9
Flask-SQLAlchemy==3.0.5
Flask-CORS==4.0.0
pandas==2.0.3
numpy==1.25.0
openpyxl==3.1.2
xlrd==2.0.1
python-dateutil==2.8.2
mysql-connector-python==8.0.33
```

## 7. 部署建议

1. **开发环境**: 
   - 使用 `Flask` 内置服务器
   - 配置环境变量 `FLASK_ENV=development`

2. **生产环境**:
   - 使用 `Gunicorn` 作为 WSGI 服务器
   - 配置 `Nginx` 作为反向代理
   - 使用 `Supervisor` 管理进程
   - 配置环境变量 `FLASK_ENV=production`

3. **数据库**:
   - 使用 MySQL/PostgreSQL
   - 配置连接池
   - 定期备份数据

## 8. 安全措施

1. **身份验证**: 考虑添加 JWT 身份验证
2. **权限控制**: 基于角色的访问控制
3. **请求限流**: 防止 API 滥用
4. **输入验证**: 严格验证所有输入参数
5. **SQL注入防护**: 使用 SQLAlchemy ORM 避免直接 SQL 拼接
6. **XSS防护**: 对输出数据进行适当转义

## 9. 性能优化

1. **数据库索引**: 为查询频繁的字段创建索引
2. **缓存**: 使用 Redis 缓存热点数据
3. **异步处理**: 对耗时操作使用异步处理
4. **分页**: 对大量数据查询进行分页
5. **批量操作**: 使用批量插入/更新减少数据库交互

## 10. 监控与日志

1. **日志记录**: 使用 `logging` 模块记录 API 请求与错误
2. **性能监控**: 使用 `Prometheus` + `Grafana` 监控 API 性能
3. **错误跟踪**: 使用 `Sentry` 等工具跟踪错误
4. **访问统计**: 记录 API 访问量与响应时间

---

本设计文档基于现有 Django 项目的功能需求，提供了完整的 Flask API 设计方案。实际开发中可根据需求进行调整与扩展。