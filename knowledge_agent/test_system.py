"""
知识整理Agent系统测试脚本
"""
import os
import sys
from agents.orchestrator import KnowledgeOrchestrator
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_conversation_processing():
    """测试对话记录处理"""
    print("=== 测试对话记录处理 ===")
    
    # 示例对话内容
    conversation_content = """
用户：什么是RAG？

助手：RAG是检索增强生成（Retrieval-Augmented Generation）的缩写，是一种将信息检索与文本生成相结合的AI技术框架。

RAG的核心思想是：
1. 首先从外部知识库中检索相关信息
2. 然后将检索到的信息作为上下文，输入给大语言模型
3. 最后由模型基于这些上下文生成更准确的回答

RAG系统通常包含以下组件：
- 向量数据库：存储知识的嵌入向量
- 检索器：根据查询找到相关信息
- 生成器：基于检索结果生成最终答案

用户：RAG有什么优势？

助手：RAG的主要优势包括：

1. **实时性**：可以接入最新的外部数据，不受模型训练时间限制
2. **准确性**：基于权威知识源生成答案，减少幻觉问题
3. **可解释性**：可以追溯答案的来源，提供引用信息
4. **效率**：不需要重新训练模型就能更新知识
5. **领域适应**：可以快速适应特定领域的知识需求

这些优势使RAG成为企业级AI应用的热门选择。
"""
    
    # 创建编排器
    orchestrator = KnowledgeOrchestrator("/Users/pluto/Desktop/知识库")
    
    # 处理对话
    input_data = {
        "content": conversation_content,
        "type": "conversation",
        "metadata": {
            "source": "AI对话",
            "topic": "RAG技术"
        },
        "operation": "create",
        "options": {
            "enable_linking": True,
            "enable_vector_db": True
        }
    }
    
    try:
        result = orchestrator.process(input_data)
        
        if result["success"]:
            print("✅ 对话处理成功!")
            print(f"📄 输出文件: {result['output_file']}")
            print(f"🆔 文档ID: {result['doc_id']}")
            print(f"📊 统计信息: {result['statistics']}")
            
            # 显示部分结果
            structured_content = result["result"]["content"]
            print("\n📝 生成的结构化内容（前500字符）:")
            print(structured_content[:500] + "..." if len(structured_content) > 500 else structured_content)
            
            print(f"\n🔗 发现的概念数: {len(result['result']['concepts'])}")
            print("主要概念:", [c['term'] for c in result['result']['concepts'][:5]])
            
        else:
            print("❌ 对话处理失败!")
            print("错误信息:", result["errors"])
            
    except Exception as e:
        print(f"❌ 测试过程中出现异常: {str(e)}")

def test_url_processing():
    """测试URL内容处理"""
    print("\n=== 测试URL内容处理 ===")
    
    # 使用一个示例URL（实际测试时需要有效的URL）
    url_content = "https://example.com/article"
    
    orchestrator = KnowledgeOrchestrator("/Users/pluto/Desktop/知识库")
    
    input_data = {
        "content": url_content,
        "type": "url",
        "metadata": {
            "source": "外部文章",
            "url": url_content
        },
        "operation": "create",
        "options": {
            "enable_linking": True,
            "enable_vector_db": True
        }
    }
    
    try:
        result = orchestrator.process(input_data)
        
        if result["success"]:
            print("✅ URL处理成功!")
            print(f"📄 输出文件: {result['output_file']}")
        else:
            print("⚠️ URL处理失败（可能是网络问题）:")
            print("错误信息:", result["errors"])
            
    except Exception as e:
        print(f"❌ URL测试中出现异常: {str(e)}")

def test_text_processing():
    """测试普通文本处理"""
    print("\n=== 测试普通文本处理 ===")
    
    text_content = """
深度学习是机器学习的一个分支，它模拟人脑神经网络的工作方式。深度学习使用多层神经网络来学习数据的复杂模式。

神经网络由多个层组成，包括输入层、隐藏层和输出层。每一层都包含多个神经元（节点），这些神经元通过权重连接。

训练过程通过反向传播算法来调整这些权重，以最小化预测误差。常见的深度学习模型包括：

1. 卷积神经网络（CNN）- 主要用于图像处理
2. 循环神经网络（RNN）- 主要用于序列数据
3. 转换器（Transformer）- 现代NLP的基础
4. 生成对抗网络（GAN）- 用于生成新数据

深度学习在计算机视觉、自然语言处理、语音识别等领域都有广泛应用。
"""
    
    orchestrator = KnowledgeOrchestrator("/Users/pluto/Desktop/知识库")
    
    input_data = {
        "content": text_content,
        "type": "text",
        "metadata": {
            "source": "技术文档",
            "topic": "深度学习"
        },
        "operation": "create",
        "options": {
            "enable_linking": True,
            "enable_vector_db": True
        }
    }
    
    try:
        result = orchestrator.process(input_data)
        
        if result["success"]:
            print("✅ 文本处理成功!")
            print(f"📄 输出文件: {result['output_file']}")
            print(f"🔗 内部链接数: {result['statistics']['internal_links']}")
            print(f"🌐 外部链接数: {result['statistics']['external_links']}")
            
        else:
            print("❌ 文本处理失败!")
            print("错误信息:", result["errors"])
            
    except Exception as e:
        print(f"❌ 文本测试中出现异常: {str(e)}")

def test_analysis_mode():
    """测试内容分析模式"""
    print("\n=== 测试内容分析模式 ===")
    
    analysis_content = """
GPT（Generative Pre-trained Transformer）是OpenAI开发的大型语言模型。GPT基于Transformer架构，
通过大规模文本数据的预训练来学习语言模式。GPT-4是目前最新的版本，具有强大的文本生成和理解能力。
"""
    
    orchestrator = KnowledgeOrchestrator("/Users/pluto/Desktop/知识库")
    
    input_data = {
        "content": analysis_content,
        "type": "text",
        "metadata": {
            "source": "分析测试"
        },
        "operation": "analyze",
        "options": {}
    }
    
    try:
        result = orchestrator.process(input_data)
        
        if result["success"]:
            print("✅ 内容分析成功!")
            analysis = result["result"]["analysis"]
            print(f"📊 内容类型: {analysis['content_type']}")
            print(f"📊 复杂度: {analysis['complexity']}")
            print(f"📊 主要概念: {[c['term'] for c in analysis['main_concepts']]}")
            print(f"📊 相关文档数: {len(result['result']['related_documents'])}")
            print(f"📊 相关概念数: {len(result['result']['related_concepts'])}")
            
        else:
            print("❌ 内容分析失败!")
            print("错误信息:", result["errors"])
            
    except Exception as e:
        print(f"❌ 分析测试中出现异常: {str(e)}")

def test_vector_db_operations():
    """测试向量数据库操作"""
    print("\n=== 测试向量数据库操作 ===")
    
    orchestrator = KnowledgeOrchestrator("/Users/pluto/Desktop/知识库")
    
    try:
        # 获取数据库统计
        stats = orchestrator.vector_db.get_collection_stats()
        print(f"📊 数据库统计: {stats}")
        
        # 测试概念搜索
        related_concepts = orchestrator.vector_db.search_related_concepts("机器学习", n_results=5)
        print(f"🔍 '机器学习'相关概念数: {len(related_concepts)}")
        
        # 测试文档搜索
        similar_docs = orchestrator.vector_db.search_similar_documents("深度学习", n_results=3)
        print(f"🔍 '深度学习'相似文档数: {len(similar_docs)}")
        
        print("✅ 向量数据库操作正常!")
        
    except Exception as e:
        print(f"❌ 向量数据库测试失败: {str(e)}")

def main():
    """主测试函数"""
    print("🚀 开始知识整理Agent系统测试")
    print("=" * 50)
    
    # 确保目录存在
    os.makedirs("/Users/pluto/Desktop/知识库", exist_ok=True)
    os.makedirs("./data/chroma_db", exist_ok=True)
    
    try:
        # 运行各项测试
        test_conversation_processing()
        test_text_processing() 
        test_analysis_mode()
        test_vector_db_operations()
        # test_url_processing()  # 可能需要网络，可选
        
        print("\n" + "=" * 50)
        print("🎉 测试完成！检查上面的输出结果。")
        print("📁 生成的文件保存在: /Users/pluto/Desktop/知识库/")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现严重错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()