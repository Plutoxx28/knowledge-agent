"""
基础Agent类，所有工作者Agent的父类
"""
from openai import OpenAI
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """所有Agent的基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
        # 初始化OpenAI客户端用于OpenRouter
        self.client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
        
    def call_llm(self, messages: List[Dict[str, str]], max_tokens: int = 2000) -> str:
        """调用大语言模型"""
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://knowledge-agent.local",
                    "X-Title": "Knowledge Agent System",
                },
                model=settings.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            
            return completion.choices[0].message.content
                
        except Exception as e:
            logger.error(f"LLM调用异常: {str(e)}")
            return ""
    
    def split_long_text(self, text: str, max_size: int = None) -> List[str]:
        """分割长文本"""
        if max_size is None:
            max_size = settings.max_chunk_size
            
        if len(text) <= max_size:
            return [text]
            
        chunks = []
        overlap = settings.chunk_overlap
        
        start = 0
        while start < len(text):
            end = min(start + max_size, len(text))
            
            # 尝试在句号或换行符处分割
            if end < len(text):
                # 向后查找最近的句号或换行符
                for i in range(end, max(start, end - 200), -1):
                    if text[i] in '.。\n':
                        end = i + 1
                        break
                        
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                
            start = max(start + 1, end - overlap)
            
        return chunks
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """处理输入数据，子类必须实现"""
        pass
    
    def get_system_prompt(self) -> str:
        """获取系统提示词，子类可以覆盖"""
        return f"""你是一个专业的{self.name}，专门负责{self.description}。
请按照以下要求处理用户输入：
1. 保持专业和准确
2. 输出格式要规范
3. 遵循既定的处理流程"""