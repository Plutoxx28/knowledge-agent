"""
内容解析工作者 - 负责解析和清洗各种格式的输入内容
"""
import re
import json
from typing import Dict, List, Any, Tuple
from bs4 import BeautifulSoup
import requests
from agents.base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)

class ContentParser(BaseAgent):
    """内容解析工作者Agent"""
    
    def __init__(self):
        super().__init__(
            name="内容解析专家",
            description="解析和清洗AI对话记录、文章、文档等多种格式的内容"
        )
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入内容
        
        Args:
            input_data: {
                "content": str,  # 内容文本或URL
                "type": str,     # 类型: "text", "url", "conversation", "markdown"
                "metadata": dict # 元数据
            }
        
        Returns:
            {
                "parsed_content": str,      # 解析后的纯文本
                "content_type": str,        # 识别的内容类型
                "structure": dict,          # 文档结构信息
                "metadata": dict,           # 提取的元数据
                "chunks": List[str]         # 分块后的内容
            }
        """
        
        content = input_data.get("content", "")
        content_type = input_data.get("type", "auto")
        metadata = input_data.get("metadata", {})
        
        # 自动识别内容类型
        if content_type == "auto":
            content_type = self._detect_content_type(content)
        
        # 根据类型解析内容
        if content_type == "url":
            parsed_content = self._parse_url(content)
        elif content_type == "conversation":
            parsed_content = self._parse_conversation(content)
        elif content_type == "markdown":
            parsed_content = self._parse_markdown(content)
        else:
            parsed_content = self._parse_plain_text(content)
        
        # 提取结构信息
        structure = self._extract_structure(parsed_content)
        
        # 分块处理
        chunks = self._intelligent_chunking(parsed_content, structure)
        
        # 提取元数据
        extracted_metadata = self._extract_metadata(parsed_content, structure)
        extracted_metadata.update(metadata)
        
        return {
            "parsed_content": parsed_content,
            "content_type": content_type,
            "structure": structure,
            "metadata": extracted_metadata,
            "chunks": chunks
        }
    
    def _detect_content_type(self, content: str) -> str:
        """自动检测内容类型"""
        # URL检测
        if content.strip().startswith(('http://', 'https://')):
            return "url"
        
        # 对话格式检测
        conversation_patterns = [
            r'用户[:：]\s*',
            r'助手[:：]\s*',
            r'User:\s*',
            r'Assistant:\s*',
            r'"role":\s*"(user|assistant)"'
        ]
        
        for pattern in conversation_patterns:
            if re.search(pattern, content[:1000]):
                return "conversation"
        
        # Markdown格式检测
        markdown_patterns = [
            r'^#{1,6}\s+',  # 标题
            r'\*\*.*?\*\*', # 粗体
            r'\[.*?\]\(.*?\)', # 链接
            r'```.*?```'    # 代码块
        ]
        
        for pattern in markdown_patterns:
            if re.search(pattern, content[:1000], re.MULTILINE):
                return "markdown"
        
        return "text"
    
    def _parse_url(self, url: str) -> str:
        """解析URL内容"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # 检测内容类型
            content_type = response.headers.get('content-type', '').lower()
            
            if 'html' in content_type:
                return self._parse_html(response.text)
            else:
                return response.text
                
        except Exception as e:
            logger.error(f"URL解析失败 {url}: {str(e)}")
            return f"无法获取URL内容: {url}\n错误: {str(e)}"
    
    def _parse_html(self, html: str) -> str:
        """解析HTML内容"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # 移除不需要的标签
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        # 提取主要内容
        text = soup.get_text()
        
        # 清理文本
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 3:  # 过滤太短的行
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _parse_conversation(self, content: str) -> str:
        """解析对话内容"""
        try:
            # 尝试解析JSON格式
            if content.strip().startswith('{'):
                data = json.loads(content)
                if 'conversation' in data:
                    return self._format_conversation(data['conversation'])
                elif 'messages' in data:
                    return self._format_conversation(data['messages'])
        except:
            pass
        
        # 解析文本格式对话
        lines = content.split('\n')
        formatted_conversation = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 识别角色和内容
            role_patterns = [
                (r'^用户[:：]\s*(.*)', 'User'),
                (r'^助手[:：]\s*(.*)', 'Assistant'),
                (r'^User:\s*(.*)', 'User'),
                (r'^Assistant:\s*(.*)', 'Assistant'),
                (r'^Human:\s*(.*)', 'User'),
                (r'^AI:\s*(.*)', 'Assistant')
            ]
            
            matched = False
            for pattern, role in role_patterns:
                match = re.match(pattern, line)
                if match:
                    content_text = match.group(1).strip()
                    if content_text:
                        formatted_conversation.append(f"**{role}**: {content_text}")
                    matched = True
                    break
            
            if not matched and line:
                # 没有明确角色标识的内容
                formatted_conversation.append(line)
        
        return '\n\n'.join(formatted_conversation)
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        """格式化消息列表"""
        formatted = []
        
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            
            role_map = {
                'user': 'User',
                'assistant': 'Assistant',
                'human': 'User',
                'ai': 'Assistant'
            }
            
            display_role = role_map.get(role.lower(), role.title())
            formatted.append(f"**{display_role}**: {content}")
        
        return '\n\n'.join(formatted)
    
    def _parse_markdown(self, content: str) -> str:
        """解析Markdown内容"""
        # 移除某些Markdown格式但保留结构
        # 这里主要是清理，不完全转换
        cleaned = re.sub(r'```[^`]*```', '[代码块]', content, flags=re.DOTALL)
        cleaned = re.sub(r'!\[.*?\]\(.*?\)', '[图片]', cleaned)
        
        return cleaned
    
    def _parse_plain_text(self, content: str) -> str:
        """解析纯文本"""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_structure(self, content: str) -> Dict[str, Any]:
        """提取文档结构"""
        structure = {
            "headings": [],
            "paragraphs": 0,
            "lists": 0,
            "code_blocks": 0,
            "estimated_reading_time": 0
        }
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 检测标题
            if re.match(r'^#{1,6}\s+', line):
                level = len(re.match(r'^(#+)', line).group(1))
                title = re.sub(r'^#{1,6}\s+', '', line)
                structure["headings"].append({"level": level, "title": title})
            
            # 检测段落
            elif line and not line.startswith(('-', '*', '1.', '2.')):
                structure["paragraphs"] += 1
            
            # 检测列表
            elif re.match(r'^[-*]\s+', line) or re.match(r'^\d+\.\s+', line):
                structure["lists"] += 1
        
        # 检测代码块
        structure["code_blocks"] = len(re.findall(r'```.*?```', content, re.DOTALL))
        
        # 估算阅读时间（按每分钟200字计算）
        word_count = len(content)
        structure["estimated_reading_time"] = max(1, word_count // 200)
        
        return structure
    
    def _intelligent_chunking(self, content: str, structure: Dict[str, Any]) -> List[str]:
        """智能分块"""
        chunks = []
        
        # 如果有明确的标题结构，按标题分块
        if structure.get("headings"):
            chunks = self._chunk_by_headings(content, structure["headings"])
        else:
            # 否则按段落和大小分块
            chunks = self._chunk_by_size(content)
        
        return chunks
    
    def _chunk_by_headings(self, content: str, headings: List[Dict]) -> List[str]:
        """按标题分块"""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        
        for line in lines:
            # 检查是否是新的一级或二级标题
            if re.match(r'^#{1,2}\s+', line.strip()):
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    current_chunk = []
            
            current_chunk.append(line)
        
        # 添加最后一块
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks
    
    def _chunk_by_size(self, content: str) -> List[str]:
        """按大小分块"""
        return self.split_long_text(content)
    
    def _extract_metadata(self, content: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """提取元数据"""
        metadata = {
            "word_count": len(content),
            "line_count": len(content.split('\n')),
            "structure_info": structure,
            "topics": [],
            "difficulty": "medium"
        }
        
        # 使用LLM提取主题和评估难度
        if len(content) > 100:  # 只对有意义长度的内容进行分析
            llm_analysis = self._analyze_with_llm(content[:2000])  # 只分析前2000字符
            metadata.update(llm_analysis)
        
        return metadata
    
    def _analyze_with_llm(self, content: str) -> Dict[str, Any]:
        """使用LLM分析内容"""
        prompt = f"""请分析以下内容，提取关键信息：

内容：
{content}

请以JSON格式返回分析结果，包含：
1. topics: 主要话题列表（最多5个）
2. difficulty: 难度等级（beginner/intermediate/advanced）
3. content_summary: 内容摘要（50字以内）
4. key_concepts: 关键概念列表（最多8个）

JSON格式：
{{
    "topics": ["话题1", "话题2"],
    "difficulty": "intermediate",
    "content_summary": "内容摘要",
    "key_concepts": ["概念1", "概念2"]
}}"""

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.call_llm(messages, max_tokens=1000)
            
            # 尝试解析JSON响应
            if response:
                # 提取JSON部分
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                    
        except Exception as e:
            logger.warning(f"LLM分析失败: {str(e)}")
        
        # 返回默认值
        return {
            "topics": [],
            "difficulty": "intermediate",
            "content_summary": "内容分析失败",
            "key_concepts": []
        }
    
    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的内容分析专家，负责分析各种文档和对话内容。
你的任务是：
1. 准确识别内容的主要话题和概念
2. 评估内容的难度等级
3. 生成简洁准确的摘要
4. 严格按照JSON格式返回结果

请保持分析的客观性和准确性。"""