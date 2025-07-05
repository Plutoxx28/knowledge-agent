"""
知识整理Agent系统配置文件
"""
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Settings:
    def __init__(self):
        # OpenRouter API配置
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY环境变量未设置，请在.env文件中配置")
        
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = os.getenv("MODEL_NAME", "google/gemini-2.5-pro")
        
        # 本地数据库配置
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        # 处理配置
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "3000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "500"))
        self.max_tokens_per_request = int(os.getenv("MAX_TOKENS_PER_REQUEST", "8000"))
        
        # 知识库配置
        self.knowledge_base_path = os.getenv("KNOWLEDGE_BASE_PATH", "/Users/pluto/Desktop/知识库/知识库")
        self.output_format = os.getenv("OUTPUT_FORMAT", "markdown")
        self.concept_link_format = os.getenv("CONCEPT_LINK_FORMAT", "[[概念名]]")

# 全局配置实例
settings = Settings()

# 保留兼容性，但现在主要使用OpenAI客户端
def get_openrouter_headers():
    return {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://knowledge-agent.local",
        "X-Title": "Knowledge Agent System",
    }