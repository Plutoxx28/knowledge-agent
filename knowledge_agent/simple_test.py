"""
简化版知识整理测试 - 直接测试核心功能
"""
from openai import OpenAI

def test_knowledge_processing():
    """测试知识整理核心功能"""
    print("🧠 测试知识整理核心功能...")
    
    # 示例对话内容
    conversation = """
用户：什么是向量数据库？

助手：向量数据库是一种专门用于存储和查询高维向量数据的数据库系统。它的主要特点包括：

1. **向量存储**：可以高效存储嵌入向量（embeddings）
2. **相似度搜索**：支持基于向量相似度的快速检索
3. **高维数据处理**：专门优化了高维数据的索引和查询
4. **应用场景**：广泛用于AI应用、推荐系统、图像检索等

常见的向量数据库包括Pinecone、Weaviate、Chroma、Milvus等。

用户：向量数据库和传统数据库有什么区别？

助手：向量数据库和传统数据库的主要区别：

**数据类型**：
- 传统数据库：存储结构化数据（文本、数字、日期等）
- 向量数据库：存储高维向量数据（通常是AI模型生成的嵌入）

**查询方式**：
- 传统数据库：基于精确匹配或条件过滤
- 向量数据库：基于相似度搜索（如余弦相似度）

**索引机制**：
- 传统数据库：B树、哈希索引等
- 向量数据库：HNSW、IVF、LSH等专门的向量索引

**应用场景**：
- 传统数据库：事务处理、数据分析、业务系统
- 向量数据库：语义搜索、推荐系统、AI应用
"""
    
    try:
        # 创建OpenAI客户端
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="YOUR_API_KEY_HERE",  # 请设置你的OpenRouter API密钥
        )
        
        # 分析对话内容
        print("📝 分析对话内容...")
        analysis_prompt = f"""请分析以下AI对话记录，提取关键信息：

对话内容：
{conversation}

请按以下格式返回分析结果：

## 主要话题
[识别的主要话题]

## 核心概念
- **[[概念1]]**: 简短定义
- **[[概念2]]**: 简短定义
...

## 概念关系
[概念1] → [概念2] → [概念3]

## 内容摘要
[100字以内的摘要]
"""
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://knowledge-agent.local",
                "X-Title": "Knowledge Agent System",
            },
            model="google/gemini-2.5-pro",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的知识分析专家，擅长从对话中提取结构化信息。请严格按照要求的格式输出。"
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            max_tokens=1000
        )
        
        result = completion.choices[0].message.content
        print("✅ 对话分析完成!")
        print("\n" + "="*50)
        print("📊 分析结果:")
        print("="*50)
        print(result)
        
        # 测试概念链接生成
        print("\n🔗 测试概念链接生成...")
        linking_prompt = """基于刚才的分析结果，请为以下概念生成更多的链接关系：

已识别概念：向量数据库、相似度搜索、嵌入向量、传统数据库

请补充：
1. 每个概念的详细定义（50字以内）
2. 概念间的关系类型（如：包含、对比、依赖等）
3. 建议的扩展概念

格式：
**[[概念名]]**: 定义
- 关系：与[[其他概念]]的关系类型
- 扩展：[[相关概念1]]、[[相关概念2]]
"""
        
        completion2 = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://knowledge-agent.local",
                "X-Title": "Knowledge Agent System",
            },
            model="google/gemini-2.5-pro",
            messages=[
                {
                    "role": "user",
                    "content": linking_prompt
                }
            ],
            max_tokens=800
        )
        
        links_result = completion2.choices[0].message.content
        print("✅ 概念链接生成完成!")
        print("\n" + "="*50)
        print("🔗 概念链接结果:")
        print("="*50)
        print(links_result)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

def test_markdown_generation():
    """测试Markdown格式生成"""
    print("\n📝 测试Markdown知识笔记生成...")
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="YOUR_API_KEY_HERE",  # 请设置你的OpenRouter API密钥
        )
        
        markdown_prompt = """请将以下内容整理成标准的知识笔记Markdown格式：

主题：深度学习基础

内容：深度学习是机器学习的一个分支，使用多层神经网络来学习数据的复杂模式。它包含输入层、隐藏层和输出层。训练过程使用反向传播算法。常见模型有CNN、RNN、Transformer等。

要求：
1. 使用[[概念]]格式标记关键概念
2. 包含概念定义部分
3. 包含知识链接部分
4. 使用中文
5. 格式要清晰规范

参考格式：
# 标题
## 核心概念
## 详细内容  
## 知识链接
"""
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://knowledge-agent.local",
                "X-Title": "Knowledge Agent System",
            },
            model="google/gemini-2.5-pro",
            messages=[
                {
                    "role": "system",
                    "content": "你是专业的知识整理专家，擅长创建结构化的Markdown文档。"
                },
                {
                    "role": "user",
                    "content": markdown_prompt
                }
            ],
            max_tokens=1000
        )
        
        markdown_result = completion.choices[0].message.content
        print("✅ Markdown生成完成!")
        print("\n" + "="*50)
        print("📄 生成的Markdown笔记:")
        print("="*50)
        print(markdown_result)
        
        # 保存到文件
        output_path = "/Users/pluto/Desktop/知识库/测试生成笔记.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_result)
        print(f"\n💾 笔记已保存到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Markdown生成失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 知识整理系统核心功能测试")
    print("=" * 60)
    
    # 测试1: 对话分析和概念提取
    success1 = test_knowledge_processing()
    
    # 测试2: Markdown格式生成
    success2 = test_markdown_generation()
    
    print("\n" + "=" * 60)
    print("📊 测试总结:")
    print(f"   对话分析: {'✅' if success1 else '❌'}")
    print(f"   Markdown生成: {'✅' if success2 else '❌'}")
    
    if success1 and success2:
        print("\n🎉 核心功能测试通过！知识整理系统可以正常工作！")
        print("💡 接下来可以:")
        print("   1. 安装完整依赖包")  
        print("   2. 运行完整的Agent系统")
        print("   3. 开始处理你的知识内容")
    else:
        print("\n⚠️ 部分功能有问题，请检查API配置。")