# 知识整理Agent系统

一个基于多Agent架构的智能知识整理系统，能够自动处理AI对话记录、文章、文档等内容，生成结构化的知识笔记并建立概念间的链接关系。

## ✨ 特性

- 🤖 **多Agent协作**: 基于编排者-工作者模式的Agent架构
- 📝 **智能解析**: 支持对话记录、URL、Markdown、纯文本等多种格式
- 🔗 **自动链接**: 智能发现概念间关系，生成双向链接
- 📊 **向量检索**: 本地向量数据库，支持语义搜索和相似度匹配
- 🧠 **长文本处理**: 层次化和流式处理策略，突破上下文限制
- 📈 **增量更新**: 智能合并新内容到现有知识库
- 🎯 **格式统一**: 标准化的Markdown输出格式

## 🏗️ 系统架构

```
Lead Knowledge Organizer (主编排Agent)
├── Content Parser (内容解析工作者)
├── Structure Builder (结构构建工作者) 
├── Link Discoverer (链接发现工作者)
├── Text Processor (长文本处理模块)
└── Vector DB (本地向量数据库)
```

## 🚀 快速启动

### 系统要求

- Python 3.8+
- Node.js 16+
- npm 或 yarn

### 后端启动

```bash
# 进入后端目录
cd knowledge_agent

# 安装Python依赖（如果需要）
pip3 install -r requirements.txt

# 启动后端API服务器
python3 run_api.py
```

**后端服务地址**：http://localhost:8000
- API文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

### 前端启动

```bash
# 进入前端目录
cd knowledge-agent-console-ui-main

# 安装前端依赖（首次运行）
npm install

# 启动前端开发服务器
npm run dev
```

**前端服务地址**：http://localhost:8080

## 📋 API 联通状态

### ✅ 后端API测试结果

1. **健康检查** - `/health`
   ```json
   {
     "status": "healthy",
     "timestamp": "2025-07-05T16:52:48.929940",
     "services": {
       "orchestrator": true,
       "vector_db": true,
       "link_manager": true,
       "progress_server": true
     }
   }
   ```

2. **文档处理** - `/process`
   - ✅ 接收处理请求正常
   - ✅ 返回结构化结果
   - ⚠️ 向量数据库元数据格式需要调整

3. **WebSocket进度推送** - `/ws/progress`
   - ✅ 连接建立正常

### ✅ 前端服务状态

- ✅ Vite开发服务器运行正常 (端口8080)
- ✅ React应用加载成功
- ✅ UI界面可访问

### 3. 基本使用

```python
from agents.orchestrator import KnowledgeOrchestrator

# 创建编排器
orchestrator = KnowledgeOrchestrator("/path/to/your/knowledge_base")

# 处理对话记录
input_data = {
    "content": "用户：什么是RAG？\\n助手：RAG是检索增强生成...",
    "type": "conversation",
    "operation": "create",
    "metadata": {"source": "AI对话", "topic": "RAG技术"}
}

result = orchestrator.process(input_data)

if result["success"]:
    print(f"生成文件: {result['output_file']}")
    print(f"发现概念: {len(result['result']['concepts'])}个")
```

### 4. 运行测试

```bash
python test_system.py
```

## 📋 支持的操作

### 创建新文档

```python
input_data = {
    "content": "你的内容...",
    "type": "auto",  # auto/text/url/conversation/markdown
    "operation": "create",
    "metadata": {"source": "来源", "topic": "主题"},
    "options": {
        "enable_linking": True,
        "enable_vector_db": True
    }
}
```

### 更新现有文档

```python
input_data = {
    "content": "补充内容...",
    "operation": "update",
    "target_file": "/path/to/existing/file.md",
    "metadata": {"update_reason": "新增信息"}
}
```

### 分析内容

```python
input_data = {
    "content": "待分析内容...",
    "operation": "analyze",
    "metadata": {"analysis_type": "quick"}
}
```

## 🔧 配置选项

### 环境变量配置

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `OPENROUTER_API_KEY` | OpenRouter API密钥 | - |
| `MODEL_NAME` | 使用的模型名称 | `google/gemini-2.5-pro` |
| `CHROMA_DB_PATH` | 向量数据库路径 | `./data/chroma_db` |
| `MAX_CHUNK_SIZE` | 最大块大小 | `3000` |
| `CHUNK_OVERLAP` | 块重叠大小 | `500` |
| `KNOWLEDGE_BASE_PATH` | 知识库路径 | `/Users/pluto/Desktop/知识库` |

### 处理选项

```python
options = {
    "enable_linking": True,        # 启用链接发现
    "enable_vector_db": True,      # 启用向量数据库
    "batch_mode": False,           # 批量处理模式
    "force_structure": False,      # 强制结构化
}
```

## 📂 输出格式

系统生成的Markdown文件包含：

```markdown
# 文档标题

## 文档信息
- **类型**: 对话记录
- **复杂度**: 中级
- **主要概念数**: 5

## 核心概念
- **[[RAG]]**: 检索增强生成技术
- **[[向量数据库]]**: 存储嵌入向量的数据库

## 主要内容
处理后的结构化内容...

## 知识链接
### 主要概念链
[[RAG]] → [[向量检索]] → [[文本生成]]

### 相关概念
- [[语义搜索]]
- [[大语言模型]]
```

## 🔍 长文本处理

系统支持多种长文本处理策略：

### 层次化处理
- 适用于结构化文档（论文、报告）
- 按章节层次分解处理
- 保持上下文连贯性

### 流式处理
- 适用于序列化内容（对话、日志）
- 滑动窗口机制
- 实时增量处理

### 混合策略
- 自动选择最优处理方式
- 结构化部分用层次化
- 序列化部分用流式

## 📊 向量数据库

### 功能特性
- 基于ChromaDB的本地向量存储
- 语义相似度搜索
- 概念关系发现
- 增量更新支持

### 使用示例

```python
from utils.vector_db import LocalVectorDB

# 初始化向量数据库
vector_db = LocalVectorDB("./data/chroma_db")

# 搜索相似文档
similar_docs = vector_db.search_similar_documents("机器学习", n_results=5)

# 搜索相关概念
related_concepts = vector_db.search_related_concepts("深度学习", n_results=10)

# 获取统计信息
stats = vector_db.get_collection_stats()
```

## 🔗 概念链接发现

系统能自动发现多种类型的概念关系：

### 关系类型
- **语义相似**: 基于向量相似度
- **共现关系**: 基于文本中的距离
- **层级关系**: 包含、属于、分类
- **因果关系**: 导致、影响、结果

### 链接建议
- 孤立概念的链接建议
- 高质量外部链接推荐
- 概念关系强度评分

## 🧪 测试

运行完整测试套件：

```bash
python test_system.py
```

测试包括：
- ✅ 对话记录处理
- ✅ 普通文本处理  
- ✅ 内容分析模式
- ✅ 向量数据库操作
- ⚠️ URL处理（需要网络）

## 📁 项目结构

```
knowledge_agent/
├── agents/
│   ├── base_agent.py          # 基础Agent类
│   ├── content_parser.py      # 内容解析工作者
│   ├── structure_builder.py   # 结构构建工作者
│   ├── link_discoverer.py     # 链接发现工作者
│   └── orchestrator.py        # 主编排Agent
├── utils/
│   ├── text_processor.py      # 长文本处理模块
│   └── vector_db.py           # 向量数据库模块
├── config.py                  # 配置管理
├── requirements.txt           # 依赖包
├── test_system.py            # 测试脚本
└── README.md                 # 说明文档
```

## 🚧 开发状态

- ✅ 核心Agent架构
- ✅ 内容解析和结构化
- ✅ 长文本处理
- ✅ 向量数据库集成
- ✅ 概念链接发现
- ⏳ OpenRouter API集成（待配置）
- ⏳ 知识图谱可视化（规划中）
- ⏳ Web界面（规划中）

## 🤝 贡献

欢迎提交Issue和Pull Request来改进系统！

## 📄 许可证

MIT License

## 💡 使用建议

1. **首次使用**: 先运行测试脚本验证系统功能
2. **大文件处理**: 利用层次化处理策略处理长文档
3. **增量更新**: 使用update操作向现有文档添加新内容
4. **概念管理**: 定期查看向量数据库中的概念关系
5. **性能优化**: 根据使用情况调整chunk大小和重叠参数

---

🎉 享受智能化的知识整理体验！