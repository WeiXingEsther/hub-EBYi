# CLAUDE.md — 多模态 RAG Chatbot 项目指引

## 项目概述

多模态检索增强生成 (Multimodal RAG) 系统，支持从 PDF 知识库中检索文本和图像，结合 Qwen-VL 多模态大模型进行推理问答。

## 标准文件路径

| 文件 | 路径 | 说明 |
|------|------|------|
| 开发需求 | [docs/开发需求.md](docs/开发需求.md) | 项目目标、核心能力、评价标准 |
| 技术方案 | [docs/技术方案.md](docs/技术方案.md) | 技术栈、系统架构、数据流、关键决策 |
| 设计规范 | [docs/设计规范.md](docs/设计规范.md) | 代码风格、项目结构、API/DB/错误处理规范 |
| 执行步骤 | [docs/执行步骤.md](docs/执行步骤.md) | 分步开发计划与验证标准 |
| 开发日志 | [dev_logs/](dev_logs/) | 每日开发记录（按日期命名：YYYY-MM-DD.md） |
| 全局配置 | [config.py](config.py) | 所有环境变量、路径、常量的统一出口 |
| Docker 编排 | [docker-compose.yml](docker-compose.yml) | Kafka + Zookeeper |

## 工作说明

### 开发前
1. 阅读 [docs/执行步骤.md](docs/执行步骤.md) 确认当前阶段
2. 阅读 [docs/设计规范.md](docs/设计规范.md) 遵循项目约定
3. 检查 [config.py](config.py) 是否需要新增配置项

### 开发中
1. 每个步骤独立完成，完成后再进入下一步
2. 新增代码遵循 PEP 8，使用类型注解
3. 所有配置从 config.py 导入，不硬编码
4. 模块间依赖：所有 service 只能依赖 common 层
5. 每个步骤完成后验证：代码可 import，模块可独立测试

### 开发后
1. 更新 [dev_logs/](dev_logs/) 记录当日完成事项和待办
2. 更新 [docs/执行步骤.md](docs/执行步骤.md) 勾选完成项
3. 如有架构变更，同步更新 [docs/技术方案.md](docs/技术方案.md)

### 技术栈速查

- **框架**: FastAPI + Uvicorn
- **PDF 解析**: mineru (本地 GPU，HTTP API 调用)
- **消息队列**: Kafka (docker-compose 部署)
- **文本向量**: BGE via DashScope `text-embedding-v2` (1024维)
- **图像向量**: CLIP via DashScope `multimodal-embedding-v1` (512维)
- **多模态 LLM**: Qwen-VL via DashScope `qwen-vl-plus`
- **向量数据库**: Milvus Lite (嵌入式，pip install)
- **元数据库**: SQLite + SQLAlchemy

### 三个服务进程

| 服务 | 端口 | 职责 |
|------|------|------|
| upload_service | 8100 | PDF 上传 + Kafka 生产者 |
| offline_worker | — | Kafka 消费者 + 文档解析 + 向量化入库 |
| chat_service | 8200 | 多模态检索 + Qwen-VL 问答 |
