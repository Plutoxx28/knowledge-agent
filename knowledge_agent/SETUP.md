# Knowledge Agent 配置指南

## 环境变量配置

### 1. 创建 .env 文件

复制示例配置文件：
```bash
cp .env.example .env
```

### 2. 配置 OpenRouter API Key

1. 访问 [OpenRouter](https://openrouter.ai/) 
2. 注册账号并获取API Key
3. 在 `.env` 文件中设置：
```
OPENROUTER_API_KEY=your_actual_api_key_here
```

### 3. 其他配置项

```bash
# OpenRouter API配置
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxx  # 你的实际API Key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=google/gemini-2.5-pro

# 知识库路径
KNOWLEDGE_BASE_PATH=/path/to/your/knowledge/base

# 其他配置保持默认值即可
```

## 安全提醒

⚠️ **重要安全事项**：
- 永远不要将 `.env` 文件提交到版本控制系统
- 不要在代码中硬编码API Key
- 定期更换API Key
- 限制API Key的权限范围

## 启动服务

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动后端API服务器：
```bash
python3 run_api.py
```

3. 启动前端（在另一个终端）：
```bash
cd ../knowledge-agent-console-ui-main
npm run dev
```

## 验证配置

启动后访问 http://localhost:8000/health 检查服务状态。

## 故障排除

如果遇到 "OPENROUTER_API_KEY环境变量未设置" 错误：
1. 确认 `.env` 文件存在且位于项目根目录
2. 确认 `.env` 文件中包含正确的API Key
3. 重启API服务器