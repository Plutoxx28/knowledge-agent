# 知识整理Agent系统

一个智能的知识整理工具，使用AI帮你处理文章、文档等内容，生成结构化的知识笔记。

## 这能做什么？

**输入**：杂乱的文本内容（对话记录、文章、笔记等）
**输出**：结构化的Markdown文档，包含：
- 提取的核心概念和定义
- 自动生成的知识链接
- 完整保留的原始内容
- 相关的扩展知识点

**典型使用场景**：
- 整理AI对话记录，提取有价值的知识点
- 处理学习笔记，建立概念间的联系
- 分析技术文档，生成知识库
- 组织研究资料，形成结构化笔记

## 快速开始

### 1. 环境要求
- Python 3.8+
- OpenRouter API账号（用于AI处理）

### 2. 安装配置

```bash
# 1. 进入后端目录
cd knowledge_agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置API密钥
cp .env.example .env
# 编辑 .env 文件，添加你的 OpenRouter API Key
```

### 3. 启动服务

```bash
# 启动后端服务
python3 run_api.py
```

### 4. 第一次使用

```python
from simple_processor import process_content_smart

# 处理你的第一个文档
result = await process_content_smart(
    content="你的文本内容...",
    content_type="text"
)

if result["success"]:
    print("处理成功！生成的文档：")
    print(result["result"]["structured_content"])
```

## 基本使用

### 输出格式示例

系统会将你的内容转换成这样的结构化文档：

```markdown
# 文档标题

## 相关反向链接
- [[相关概念1]] - 关联说明
- [[相关概念2]] - 关联说明

## 相关概念
- **RAG**：检索增强生成技术，结合检索和生成的AI技术
- **向量数据库**：存储嵌入向量的专用数据库系统

## 原始内容
原始输入内容（完全保留，不做任何修改）

## 扩展知识
- 语义搜索
- 大语言模型
- 嵌入向量
- 相似度计算
```

### 常用功能

**处理对话记录**：
```python
result = await process_content_smart(
    content="用户：什么是RAG？\n助手：RAG是检索增强生成...",
    content_type="conversation"
)
```

**处理文章笔记**：
```python
result = await process_content_smart(
    content="文章内容...",
    content_type="text"
)
```

**Web界面使用**：
1. 访问 http://localhost:8080
2. 粘贴你的内容
3. 点击处理，等待结果
4. 下载生成的Markdown文件

**处理模式说明**：

系统采用**AI自动控制**的智能处理策略：

1. **优先使用AI编排模式**：
   - 自动使用Flash+Pro双模型策略
   - Flash模型快速处理，Pro模型质量检查和最终合成
   - 获得最佳的质量和性能平衡

2. **自动降级到标准模式**：
   - 当AI编排模式失败时（网络问题、API限制等）
   - 自动切换到`MODEL_NAME`配置的单一模型
   - 确保系统始终可用

**无需手动选择模式** - AI会自动管理，确保最佳体验

### Web界面

如果你想要图形界面：

```bash
# 启动前端（需要Node.js）
cd knowledge-agent-console-ui-main
npm install
npm run dev
```

访问 http://localhost:8080 使用Web界面。

### API接口

系统提供REST API供其他程序调用：

- **健康检查**: `GET http://localhost:8000/health`
- **处理内容**: `POST http://localhost:8000/process`
- **API文档**: http://localhost:8000/docs

## 故障排除

### 常见问题

**"OPENROUTER_API_KEY环境变量未设置"**
- 确认`.env`文件存在且包含正确的API Key
- 重启服务

**处理失败或质量不佳**
- 系统会自动尝试最佳处理模式
- 检查API Key是否有效
- 检查输入内容格式是否合适

**Web界面无法访问**
- 确认前端服务已启动（npm run dev）
- 检查端口8080是否被占用

### 系统测试

```bash
# 测试API服务器
curl http://localhost:8000/health

# 测试处理功能
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"content": "测试内容", "type": "text"}'
```

### 系统架构

系统采用模块化设计：

```
knowledge_agent/
├── core/              # 核心业务逻辑
├── tools/             # 处理工具模块
├── link_system/       # 链接管理系统
├── utils/             # 工具函数
├── agents/            # Agent模块
├── data/              # 数据存储
├── api_server.py      # API服务器
├── run_api.py         # 启动脚本
└── simple_processor.py # 主要入口
```

### 核心特性

- **智能处理**: AI自动选择最适合的处理策略
- **概念提取**: 自动识别和定义核心概念
- **知识链接（当前版本暂未使用）**: 建立概念间的关联关系
- **向量搜索（当前版本暂未使用）**: 基于语义的相似内容检索
- **模块化设计**: 易于扩展和维护

### 高级编程接口

如果你需要在程序中集成系统：

```python
from core.knowledge_processor import SimpleKnowledgeProcessor
from tools.concept_extractor import ConceptExtractor

# 直接使用处理器
processor = SimpleKnowledgeProcessor()
result = await processor.process_content(
    content="你的内容",
    content_type="text",
    options={"enable_ai_orchestration": True}
)

# 或单独使用概念提取
extractor = ConceptExtractor()
concepts = await extractor.extract_concepts(content)
```

### 配置选项

参考knowledge_agent/.env.example

## 开发和贡献

### 项目状态

**已完成功能**:
- 单Agent工具架构 - 完整的AI工具编排系统
- 内容解析和结构化 - 支持多种格式的内容处理
- 长文本处理（暂未使用到当前版本中） - 层次化和流式处理策略
- 向量数据库集成（暂未使用到当前版本中） - 基于ChromaDB的语义搜索
- 概念链接发现（暂未使用到当前版本中） - 自动建立概念间的关系
- 模块化架构 - 完全模块化的代码结构
- API服务器 - 完整的REST API和WebSocket支持


### 技术架构特点

- **单Agent工具调用架构**: AI自动选择和组合专用工具函数
- **模块化设计**: 核心业务逻辑、工具模块、链接系统独立
- **双模型策略**: Flash模型快速处理 + Pro模型质量保证
- **异步处理**: 支持高并发的异步内容处理
- **REST API**: 完整的API接口和WebSocket支持

### 贡献指南

欢迎提交Issue和Pull Request来改进系统！

**如何贡献**:
- 报告Bug或提出功能建议
- 提交代码改进
- 完善文档和示例
- 分享使用经验

### 许可证

MIT License - 详见LICENSE文件

### 使用建议

1. **新手入门**: 从标准模式开始，熟悉后再尝试AI编排模式
2. **质量优先**: 使用AI编排模式获得最佳处理效果
3. **大文件处理**: 系统自动处理长文档，无需特殊配置
4. **定期更新**: 定期更新API Key和检查系统状态


---

**快速开始，智能整理你的知识！**