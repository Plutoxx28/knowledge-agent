"""
测试OpenRouter API连接
"""
from openai import OpenAI
from config import settings

def test_openrouter_connection():
    """测试OpenRouter API连接"""
    print("🔗 测试OpenRouter API连接...")
    
    try:
        # 创建OpenAI客户端
        client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
        
        # 发送测试请求
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://knowledge-agent.local",
                "X-Title": "Knowledge Agent System",
            },
            model=settings.model_name,
            messages=[
                {
                    "role": "user", 
                    "content": "请简单回答：你好，这是一个API连接测试。"
                }
            ],
            max_tokens=100
        )
        
        response = completion.choices[0].message.content
        print(f"✅ API连接成功!")
        print(f"📝 响应内容: {response}")
        return True
        
    except Exception as e:
        print(f"❌ API连接失败: {str(e)}")
        print("💡 请检查:")
        print("   1. API Key是否正确")
        print("   2. 网络连接是否正常")
        print("   3. 模型名称是否正确")
        return False

def test_with_your_api_key():
    """使用你提供的API Key测试"""
    print("\n🔑 使用提供的API Key测试...")
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="YOUR_API_KEY_HERE",  # 请设置你的OpenRouter API密钥
        )
        
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://knowledge-agent.local",
                "X-Title": "Knowledge Agent System",
            },
            model="google/gemini-2.5-pro",
            messages=[
                {
                    "role": "user",
                    "content": "请用中文简单介绍一下什么是RAG技术？"
                }
            ],
            max_tokens=200
        )
        
        response = completion.choices[0].message.content
        print(f"✅ 直接API调用成功!")
        print(f"📝 RAG技术介绍: {response[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ 直接API调用失败: {str(e)}")
        return False

def test_agent_llm_call():
    """测试Agent的LLM调用"""
    print("\n🤖 测试Agent LLM调用...")
    
    try:
        from agents.content_parser import ContentParser
        
        parser = ContentParser()
        
        messages = [
            {"role": "system", "content": "你是一个专业的内容分析专家。"},
            {"role": "user", "content": "请分析这句话的主要概念：机器学习是人工智能的一个重要分支。"}
        ]
        
        response = parser.call_llm(messages, max_tokens=150)
        
        if response:
            print(f"✅ Agent LLM调用成功!")
            print(f"📝 分析结果: {response}")
            return True
        else:
            print("❌ Agent LLM调用失败 - 返回空响应")
            return False
            
    except Exception as e:
        print(f"❌ Agent LLM调用失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 OpenRouter API连接测试")
    print("=" * 50)
    
    # 测试1: 配置文件的API连接
    success1 = test_openrouter_connection()
    
    # 测试2: 直接使用提供的API Key
    success2 = test_with_your_api_key()
    
    # 测试3: Agent的LLM调用
    success3 = test_agent_llm_call()
    
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    print(f"   配置文件API: {'✅' if success1 else '❌'}")
    print(f"   直接API调用: {'✅' if success2 else '❌'}")
    print(f"   Agent调用: {'✅' if success3 else '❌'}")
    
    if success2 or success3:
        print("\n🎉 API连接正常，可以开始使用知识整理系统!")
    else:
        print("\n⚠️ API连接有问题，请检查配置或网络连接。")