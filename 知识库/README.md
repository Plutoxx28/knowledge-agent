# 知识库

这是一个基于双向链接的智能知识库，包含AI技术、系统设计和工具相关的知识文档。

## 📁 目录结构

### 🤖 AI技术
包含人工智能相关的核心技术文档：
- [继续预训练](AI技术/继续预训练.md) - 继续预训练技术详解
- [模型概念](AI技术/模型概念.md) - AI模型基础概念
- [AI 模型训练过程](AI技术/AI模型训练过程.md) - 模型训练流程
- [RAG相关](AI技术/RAG相关.md) - 检索增强生成技术

### 🏗️ 系统设计
包含系统架构和设计相关文档：
- [知识整理Agent系统设计方案](系统设计/知识整理Agent系统设计方案.md) - 核心系统设计
- [实现前准备清单](系统设计/实现前准备清单.md) - 项目实现指南

### 🛠️ 工具相关
包含开发工具和平台相关文档：
- [MCP相关](工具相关/MCP相关.md) - Model Context Protocol相关知识

## 🔗 双向链接系统

本知识库支持双向链接功能：
- 使用 `[[概念名]]` 语法创建概念链接
- 系统会自动识别概念并建立文档间的关联
- 支持生成可视化的概念图谱

## 🚀 使用方法

### 查看链接版本
```bash
# 生成HTML版本（带可点击链接）
python3 knowledge_agent/tools/link_cli.py /Users/yanglulu10/knowledge-agent/知识库 render 文档路径 --format html

# 扫描知识库更新链接
python3 knowledge_agent/tools/link_cli.py /Users/yanglulu10/knowledge-agent/知识库 scan

# 生成概念图谱
python3 knowledge_agent/tools/link_cli.py /Users/yanglulu10/knowledge-agent/知识库 graph --output graph.html
```

### 查询概念
```bash
# 查看特定概念的链接信息
python3 knowledge_agent/tools/link_cli.py /Users/yanglulu10/knowledge-agent/知识库 concept "概念名"

# 生成链接报告
python3 knowledge_agent/tools/link_cli.py /Users/yanglulu10/knowledge-agent/知识库 report
```

## 📝 贡献指南

1. **创建新文档**：文档标题应直接对应概念名，便于建立双向链接
2. **使用链接语法**：用 `[[概念名]]` 引用其他概念
3. **分类存放**：根据内容性质放入对应的分类目录
4. **更新链接**：添加新文档后运行扫描命令更新链接关系

---

*此知识库由 Knowledge Agent 系统管理，支持智能链接和概念图谱生成。*