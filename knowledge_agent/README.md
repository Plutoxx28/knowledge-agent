# 知识整理Agent系统

一个基于单Agent智能工具调用架构的知识整理系统，能够自动处理AI对话记录、文章、文档等内容，生成结构化的知识笔记并建立概念间的链接关系。

## 🚀 最新更新

### 🧠 AI策略优化系统上线

系统新增完整的AI策略优化能力，实现真正的智能化自主决策：

#### ✨ 策略优化核心功能
- **🤖 智能策略选择**: AI基于历史数据自动选择最优处理策略
- **📊 性能监控分析**: 实时跟踪和分析6种处理策略的表现
- **🔄 持续学习优化**: 系统自动从执行历史中学习和改进
- **📈 模式识别预测**: 识别成功/失败模式，预测最佳策略

#### 🎯 智能化突破
- **从固定规则** → **基于数据的智能决策**
- **从单一策略** → **6种专门化策略动态选择**
- **从静态处理** → **自适应学习和持续优化**
- **从通用处理** → **个性化内容匹配**

#### 📊 策略监控API
- 11个专用API端点实时监控策略性能
- 策略排名、趋势分析、失败模式识别
- 手动触发学习周期、获取优化建议

### 架构重构完成

经过全面的代码重构，系统现已实现完全模块化，具备更好的可维护性和扩展性：

#### ✨ 核心改进
- **模块化架构**: 将2100+行的大文件拆分为清晰的模块结构
- **职责分离**: 每个模块单一职责，边界清晰
- **代码复用**: 统一的工具接口，消除重复代码
- **智能降级**: AI增强 → 混合模式 → 规则备用的完整降级机制

#### 🏗️ 新的目录结构
- **`core/`** - 核心业务逻辑（进度跟踪、AI编排、知识处理）
- **`tools/`** - 通用工具模块（概念提取、内容分析、结构构建）  
- **`link_system/`** - 完整的链接系统（数据库、解析、查询、分析）

#### 📊 重构成果
- **代码行数**: 主要文件从2100+行减少到75行
- **模块数量**: 新增12个独立模块
- **功能完整性**: 100%保持原有功能
- **向后兼容**: 所有现有接口保持不变

#### 🔧 技术优势
- **可维护性**: 小模块易于理解和维护
- **可测试性**: 独立模块便于单元测试
- **可扩展性**: 新功能可轻松添加到对应模块
- **依赖管理**: 清晰的模块间依赖关系

### 核心特性

- **🧠 AI策略优化**: 基于历史数据的智能策略选择和持续学习
- **🤖 智能决策引擎**: 6种专门化处理策略自动匹配最优方案
- **📊 性能监控**: 实时策略性能分析、趋势识别、失败模式检测
- **🔄 自适应学习**: 系统自动从执行历史中学习并优化策略权重
- **Agent工具调用**: 基于AI编排的智能工具调用架构，22个专用工具函数
- **智能工具选择**: AI自动选择和组合最适合的工具链进行处理
- **动态工具组合**: 根据内容类型和复杂度动态调整工具使用策略
- **智能解析**: 支持Markdown、纯文本解析
- **自动链接**: 智能发现概念间关系，生成双向链接
- **向量检索**: 本地向量数据库，支持语义搜索和相似度匹配
- **长文本处理**: 层次化和流式处理策略，突破上下文限制
- **增量更新**: 智能合并新内容到现有知识库
- **格式统一**: 标准化的Markdown输出格式

## 系统架构

```
Knowledge Agent System (单Agent架构)
├── AI Tool Orchestrator (AI工具编排器)
│   ├── 22个专用工具函数
│   ├── 智能工具选择与组合
│   └── 动态工具调用链
├── Core Modules (核心模块)
│   ├── Knowledge Processor (知识处理器)
│   ├── Progress Tracker (进度跟踪器)
│   └── AI Orchestrator (AI编排器)
├── Tool Modules (工具模块)
│   ├── Concept Extractor (概念提取器)
│   ├── Content Analyzer (内容分析器)
│   └── Structure Builder (结构构建器)
├── Link System (链接系统)
│   ├── Database Manager (数据库管理)
│   ├── Content Parser (内容解析)
│   ├── Link Resolver (链接解析)
│   └── Analysis Service (分析服务)
└── Storage Layer (存储层)
    ├── Vector DB (向量数据库)
    ├── SQLite (关系数据库)
    └── File System (文件系统)
```

## AI工具编排系统

### 工具类型

系统包含22个专用工具函数，分为以下类型：

- **内容分析工具**: 文本类型识别、结构分析、复杂度评估
- **概念提取工具**: 关键词提取、实体识别、概念关系发现
- **结构化工具**: 大纲生成、分段组织、层次化构建
- **链接处理工具**: 双向链接创建、引用关系维护
- **质量控制工具**: 内容验证、格式检查、一致性保证
- **增强工具**: 内容丰富化、示例添加、经验总结

### 工具选择机制

```python
# AI自动分析内容并选择工具
tools_to_use = await orchestrator.select_tools_for_content(
    content=content,
    content_type="conversation",
    complexity_level="medium"
)

# 示例输出：
# [
#     "dialogue_parser",
#     "general_concept_extractor", 
#     "relationship_analyzer",
#     "summary_generator",
#     "advanced_markdown_structurer"
# ]
```

### 智能降级机制

- **AI增强模式**: 使用大模型进行智能分析和处理
- **混合模式**: AI + 规则结合，平衡效果与性能
- **规则备用**: 纯规则处理，确保基本功能可用

## 快速启动

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

## API 联通状态

### 后端API测试结果

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
   - 接收处理请求正常
   - 返回结构化结果
   - 向量数据库元数据格式需要调整

3. **WebSocket进度推送** - `/ws/progress`
   - 连接建立正常

### 前端服务状态

- Vite开发服务器运行正常 (端口8080)
- React应用加载成功
- UI界面可访问

### 3. 基本使用

#### 单Agent工具调用接口（推荐）

```python
from core.knowledge_processor import SimpleKnowledgeProcessor
from core.ai_orchestrator import AIToolOrchestrator
from tools.concept_extractor import ConceptExtractor
from tools.content_analyzer import ContentAnalyzer
from tools.structure_builder import StructureBuilder

# 使用AI工具编排器（智能工具调用）
orchestrator = AIToolOrchestrator()

# AI编排处理 - 自动选择和组合22个工具
result = await orchestrator.process_content_with_orchestration(
    content="用户：什么是RAG？\n助手：RAG是检索增强生成...",
    content_type="conversation",
    metadata={"source": "AI对话", "topic": "RAG技术"}
)

if result["success"]:
    print(f"AI编排处理成功！")
    print(f"使用工具: {result['tools_used']}")
    print(f"发现概念: {len(result['result']['concepts'])}个")

# 使用统一的知识处理器（传统处理）
processor = SimpleKnowledgeProcessor()

# 传统处理模式
result = await processor.process_content(
    content="用户：什么是RAG？\n助手：RAG是检索增强生成...",
    content_type="conversation",
    options={
        "enable_ai_orchestration": True,  # 启用AI编排
        "enable_linking": True,
        "enable_vector_db": True
    }
)

# 直接使用工具模块（精准控制）
extractor = ConceptExtractor()
analyzer = ContentAnalyzer()
builder = StructureBuilder()

# 概念提取
concepts = await extractor.extract_concepts(content, method="ai_enhanced")

# 内容分析
analysis = await analyzer.analyze_content(content, method="hybrid")

# 结构构建
structured = await builder.build_structure(content, concepts, analysis)
```

#### 便捷处理接口（推荐）

```python
from simple_processor import process_content_smart

# 智能处理入口 - 自动选择最佳处理方式
result = await process_content_smart(
    content="用户：什么是RAG？\n助手：RAG是检索增强生成...",
    content_type="conversation",
    metadata={"source": "AI对话", "topic": "RAG技术"},
    enable_ai_orchestration=True  # 启用AI编排
)

if result["success"]:
    print(f"智能处理成功！")
    print(f"处理方式: {result['processing_method']}")
    print(f"发现概念: {len(result['result']['concepts'])}个")
```

#### 原有接口（已迁移）

```python
# 注意：旧的多Agent架构已迁移到新的AI策略优化系统
# 请使用上述的新接口进行处理

# 旧代码：
# from agents.orchestrator import KnowledgeOrchestrator
# orchestrator = KnowledgeOrchestrator("/path/to/your/knowledge_base")

# 新代码：
from simple_processor import process_content_smart
result = await process_content_smart(
    content="用户：什么是RAG？\n助手：RAG是检索增强生成...",
    content_type="conversation",
    metadata={"source": "AI对话", "topic": "RAG技术"}
)
```

#### 链接系统使用

```python
from link_system import LinkManager

# 创建链接管理器
link_manager = LinkManager("/path/to/knowledge_base")

# 扫描知识库
stats = link_manager.scan_knowledge_base()
print(f"处理了 {stats['processed_files']} 个文件")

# 查询概念
concept_links = link_manager.get_concept_links("RAG")
print(f"找到 {len(concept_links)} 个相关链接")

# 生成分析报告
report = link_manager.generate_comprehensive_report()
print(f"链接质量分数: {report['link_quality']['quality_score']}")
```

### 4. 系统测试

```bash
# 测试API服务器
curl http://localhost:8000/health

# 测试处理功能
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"content": "测试内容", "type": "text"}'
```

## 支持的操作

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

## 配置选项

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

## 输出格式

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

## 长文本处理

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

## 向量数据库

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

## 概念链接发现

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

## 测试

系统功能验证：

```bash
# 启动API服务器
python3 run_api.py

# 健康检查
curl http://localhost:8000/health

# 策略优化状态检查
curl http://localhost:8000/strategy/performance
```

主要功能：
- AI策略优化处理
- 智能内容分析
- 概念提取和链接发现
- 向量数据库语义搜索
- 实时进度跟踪

## 项目结构

```
knowledge_agent/
├── core/                       # 核心业务逻辑模块
│   ├── __init__.py
│   ├── progress_tracker.py     # 统一进度跟踪
│   ├── ai_orchestrator.py      # AI工具编排系统
│   └── knowledge_processor.py  # 知识处理核心
├── tools/                      # 通用工具模块
│   ├── __init__.py
│   ├── concept_extractor.py    # 概念提取工具
│   ├── content_analyzer.py     # 内容分析工具
│   └── structure_builder.py    # 结构构建工具
├── link_system/               # 链接系统模块
│   ├── __init__.py
│   ├── data_models.py         # 数据模型定义
│   ├── database_manager.py    # 数据库操作管理
│   ├── content_parser.py      # 内容解析器
│   ├── link_resolver.py       # 链接解析器
│   ├── query_service.py       # 统一查询服务
│   ├── path_utils.py          # 路径处理工具
│   ├── analysis_service.py    # 分析服务
│   └── link_manager.py        # 主要管理接口
├── utils/                     # 工具模块
│   ├── text_processor.py      # 长文本处理模块
│   ├── vector_db.py           # 向量数据库模块
│   ├── progress_websocket.py  # WebSocket进度服务
│   ├── file_watcher.py        # 文件监控
│   └── link_renderer.py       # 链接渲染工具
├── simple_processor.py        # 便捷处理入口
├── api_server.py              # API服务器
├── run_api.py                 # API启动脚本
├── config.py                  # 配置管理
├── requirements.txt           # 依赖包
└── README.md                 # 说明文档
```

## 开发状态

### ✅ 已完成功能
- ✅ **单Agent工具架构** - 完整的AI工具编排系统
- ✅ **内容解析和结构化** - 支持多种格式的内容处理
- ✅ **长文本处理** - 层次化和流式处理策略
- ✅ **向量数据库集成** - 基于ChromaDB的语义搜索
- ✅ **概念链接发现** - 自动建立概念间的关系
- ✅ **模块化架构重构** - 完全模块化的代码结构
- ✅ **统一工具接口** - 标准化的工具调用方式
- ✅ **智能降级机制** - AI → 混合 → 规则的完整降级
- ✅ **链接系统重构** - 完整的链接管理和分析功能
- ✅ **API服务器** - 完整的REST API和WebSocket支持
- ✅ **向后兼容性** - 保持所有现有接口不变

### 🔧 配置中
- 🔧 **OpenRouter API集成** - 需要配置API密钥

### 📋 规划中
- 📋 **知识图谱可视化** - 概念关系的图形化展示
- 📋 **Web界面优化** - 更好的用户体验
- 📋 **批量处理优化** - 大规模文档处理性能优化
- 📋 **插件系统** - 支持自定义处理插件

### 🚀 重构成果
- **架构升级**: 从多Agent架构升级为单Agent工具调用架构
- **代码质量**: 从单体架构重构为模块化架构
- **智能化提升**: AI自动选择和组合22个专用工具函数
- **可维护性**: 大幅提升代码可读性和维护性
- **可扩展性**: 新功能可轻松添加到对应模块
- **性能优化**: 智能降级机制确保系统稳定性
- **功能完整性**: 100%保持原有功能，新增多种便捷接口

## 贡献

欢迎提交Issue和Pull Request来改进系统！

## 许可证

MIT License

## 使用建议

1. **首次使用**: 先运行测试脚本验证系统功能
2. **AI编排模式**: 优先使用AI工具编排器获得最佳效果
3. **大文件处理**: 利用层次化处理策略处理长文档
4. **增量更新**: 使用update操作向现有文档添加新内容
5. **概念管理**: 定期查看向量数据库中的概念关系
6. **性能优化**: 根据使用情况调整chunk大小和重叠参数

---

享受智能化的知识整理体验!