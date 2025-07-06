#!/usr/bin/env python3
"""
简化的知识处理器 - 重新实现一个可工作的版本
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)

class SimpleProgressTracker:
    """简化的进度跟踪器"""
    
    def __init__(self, websocket_broadcast_func: Optional[Callable] = None):
        self.websocket_broadcast = websocket_broadcast_func
        self.task_id = str(uuid.uuid4())
        
    async def update_progress(self, stage: str, message: str, progress_percent: int, workers: List[str] = None):
        """更新进度"""
        progress_data = {
            "task_id": self.task_id,
            "stage": stage,
            "current_step": message,
            "progress_percent": progress_percent,
            "workers": workers or [],
            "timestamp": time.time()
        }
        
        logger.info(f"进度更新: {stage} - {message} ({progress_percent}%)")
        
        if self.websocket_broadcast:
            try:
                await self.websocket_broadcast(progress_data)
                logger.info(f"WebSocket进度广播成功")
            except Exception as e:
                logger.error(f"WebSocket进度广播失败: {e}")
        else:
            logger.info("没有WebSocket广播函数，跳过进度广播")

class SimpleKnowledgeProcessor:
    """简化的知识处理器"""
    
    def __init__(self, websocket_broadcast_func: Optional[Callable] = None):
        self.websocket_broadcast = websocket_broadcast_func
        
        # 初始化AI客户端
        try:
            self.ai_client = OpenAI(
                base_url=settings.openrouter_base_url,
                api_key=settings.openrouter_api_key,
                default_headers={
                    "HTTP-Referer": "https://knowledge-agent.local",
                    "X-Title": "Knowledge Agent System",
                }
            )
            logger.info(f"AI客户端初始化成功，使用模型: {settings.model_name}")
            logger.info(f"API Key前缀: {settings.openrouter_api_key[:20]}...")
            logger.info(f"Base URL: {settings.openrouter_base_url}")
        except Exception as e:
            logger.error(f"AI客户端初始化失败: {e}")
            self.ai_client = None
        
        logger.info("简化知识处理器初始化完成")
    
    async def process_content(self, content: str, content_type: str = "text", 
                            metadata: Dict[str, Any] = None, 
                            options: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理内容的主要方法"""
        
        # 初始化进度跟踪
        tracker = SimpleProgressTracker(self.websocket_broadcast)
        
        try:
            # 阶段1: 开始分析 (0-20%)
            await tracker.update_progress("analyzing", "AI分析内容中...", 10, ["AI分析器"])
            
            # 使用AI分析内容
            analysis = await self._ai_analyze_content(content)
            await tracker.update_progress("analyzing", "AI分析完成", 20)
            
            # 阶段2: 概念提取 (20-50%)
            await tracker.update_progress("worker_processing", "AI概念提取中...", 30, ["概念提取器"])
            
            concepts = await self._ai_extract_concepts(content)
            await tracker.update_progress("worker_processing", "概念提取完成", 50)
            
            # 阶段3: 结构化处理 (50-80%)
            await tracker.update_progress("worker_processing", "结构化构建中...", 60, ["结构构建器"])
            
            structured_content = await self._ai_structure_content(content, concepts, analysis)
            await tracker.update_progress("worker_processing", "结构化完成", 80)
            
            # 阶段4: 完成处理 (80-100%)
            await tracker.update_progress("finalizing", "保存处理结果...", 90, ["文件管理器"])
            
            # 检查AI处理是否成功
            ai_success = not (analysis.get('error') or any('error' in str(c) for c in concepts))
            processing_method = "AI增强" if ai_success else "基础模板"
            
            # 生成最终结果
            result = {
                "content": structured_content,
                "concepts": concepts,
                "analysis": analysis,
                "metadata": metadata or {},
                "statistics": {
                    "original_length": len(content),
                    "processed_length": len(structured_content),
                    "concept_count": len(concepts),
                    "ai_enhanced": ai_success,
                    "processing_method": processing_method,
                    "ai_calls_successful": ai_success
                }
            }
            
            await tracker.update_progress("completed", "处理完成！", 100)
            
            return {
                "success": True,
                "result": result,
                "doc_id": tracker.task_id,
                "task_id": tracker.task_id
            }
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            await tracker.update_progress("error", f"处理失败: {str(e)}", 0)
            return {
                "success": False,
                "error": str(e),
                "task_id": tracker.task_id
            }
    
    async def _ai_analyze_content(self, content: str) -> Dict[str, Any]:
        """AI分析内容"""
        if not self.ai_client:
            logger.error("AI客户端未初始化，使用fallback分析")
            return {
                "main_topic": "内容分析",
                "complexity": "medium",
                "content_type": "general",
                "key_themes": [],
                "error": "AI客户端未初始化"
            }
            
        try:
            logger.info(f"开始AI内容分析... 使用模型: {settings.model_name}")
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.ai_client.chat.completions.create(
                    model=settings.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的内容分析专家。请分析给定内容的主题、复杂度和结构。"
                        },
                        {
                            "role": "user",
                            "content": f"""请分析以下内容：

{content[:1500]}

请以JSON格式返回分析结果：
{{
  "main_topic": "主要话题",
  "complexity": "simple|medium|complex",
  "content_type": "technical|educational|general",
  "key_themes": ["主题1", "主题2"]
}}"""
                        }
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
            )
            
            ai_response = response.choices[0].message.content
            logger.info(f"AI分析响应: {ai_response[:200]}...")
            
            try:
                return json.loads(ai_response)
            except json.JSONDecodeError:
                return {
                    "main_topic": "未知主题",
                    "complexity": "medium",
                    "content_type": "general",
                    "key_themes": [],
                    "ai_raw_response": ai_response
                }
                
        except Exception as e:
            logger.error(f"AI分析失败: {e}")
            return {
                "main_topic": "分析失败",
                "complexity": "medium",
                "content_type": "general",
                "key_themes": [],
                "error": str(e)
            }
    
    async def _ai_extract_concepts(self, content: str) -> List[Dict[str, Any]]:
        """AI概念提取"""
        if not self.ai_client:
            logger.error("AI客户端未初始化，使用fallback概念提取")
            return self._fallback_concept_extraction(content)
            
        try:
            logger.info(f"开始AI概念提取... 使用模型: {settings.model_name}")
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ai_client.chat.completions.create(
                    model=settings.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的概念提取专家。请从内容中提取核心概念和重要术语。"
                        },
                        {
                            "role": "user",
                            "content": f"""请从以下内容中提取核心概念：

{content[:1500]}

请以JSON数组格式返回概念列表：
[
  {{"term": "概念名称", "definition": "概念定义", "type": "概念类型", "confidence": 0.9}},
  {{"term": "术语名称", "definition": "术语解释", "type": "technical_term", "confidence": 0.8}}
]
"""
                        }
                    ],
                    max_tokens=800,
                    temperature=0.3
                )
            )
            
            ai_response = response.choices[0].message.content
            logger.info(f"AI概念提取响应: {ai_response[:200]}...")
            
            try:
                # 清理响应中的markdown标记
                clean_response = ai_response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                concepts = json.loads(clean_response)
                # 添加AI标记
                for concept in concepts:
                    concept['source'] = 'ai_enhanced'
                    concept['final_score'] = concept.get('confidence', 0.7)
                
                logger.info(f"AI成功提取了 {len(concepts)} 个概念")
                return concepts
                
            except json.JSONDecodeError as e:
                logger.warning(f"AI概念提取JSON解析失败: {e}，使用备用方法")
                return self._fallback_concept_extraction(content)
                
        except Exception as e:
            logger.error(f"AI概念提取失败: {e}")
            return self._fallback_concept_extraction(content)
    
    def _fallback_concept_extraction(self, content: str) -> List[Dict[str, Any]]:
        """备用概念提取方法"""
        import re
        concepts = []
        
        # 简单的关键词提取
        patterns = [
            (r'\b([A-Z]{2,})\b', 'acronym'),
            (r'([一-龟]{2,8})', 'chinese_term'),
            (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', 'proper_noun'),
        ]
        
        for pattern, term_type in patterns:
            matches = re.findall(pattern, content)
            for match in matches[:5]:  # 限制数量
                concepts.append({
                    'term': match,
                    'definition': '',
                    'type': term_type,
                    'confidence': 0.6,
                    'source': 'fallback_extraction',
                    'final_score': 0.6
                })
        
        return concepts[:10]  # 最多10个概念
    
    async def _ai_structure_content(self, content: str, concepts: List[Dict], 
                                  analysis: Dict[str, Any]) -> str:
        """AI结构化内容"""
        if not self.ai_client:
            logger.error("AI客户端未初始化，使用fallback结构化")
            return self._create_fallback_structure(content, concepts, analysis)
            
        try:
            logger.info(f"开始AI结构化... 使用模型: {settings.model_name}")
            
            concept_names = [c['term'] for c in concepts[:10]]  # 最多10个概念
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ai_client.chat.completions.create(
                    model=settings.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的知识整理专家。请将内容重新组织为结构化的Markdown格式。"
                        },
                        {
                            "role": "user",
                            "content": f"""请将以下内容重新整理为结构化的Markdown格式：

原始内容：
{content}

主要概念：{', '.join(concept_names)}

要求：
1. 使用清晰的Markdown标题结构
2. 为重要概念添加双链格式：[[概念名]]
3. 保持原意不变，但组织更清晰
4. 添加核心概念列表

请返回重新整理后的Markdown内容。"""
                        }
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
            )
            
            structured = response.choices[0].message.content
            logger.info(f"AI结构化完成，长度: {len(structured)}")
            
            return structured
            
        except Exception as e:
            logger.error(f"AI结构化失败: {e}")
            # 使用fallback方法
            return self._create_fallback_structure(content, concepts, analysis)
    
    def _create_fallback_structure(self, content: str, concepts: List[Dict], analysis: Dict[str, Any]) -> str:
        """创建fallback结构化内容"""
        main_topic = analysis.get('main_topic', '知识整理')
        if main_topic == "未知主题" or main_topic == "分析失败":
            # 尝试从内容中提取标题
            lines = content.strip().split('\n')
            for line in lines[:5]:
                line = line.strip()
                if line.startswith('#'):
                    main_topic = line.lstrip('#').strip()
                    break
                elif len(line) > 5 and len(line) < 100:
                    main_topic = line
                    break
            else:
                main_topic = "知识整理"
        
        # 构建更好的fallback内容
        structured_parts = [f"# {main_topic}"]
        
        # 添加概念部分
        if concepts and len(concepts) > 0:
            structured_parts.append("\n## 核心概念\n")
            valid_concepts = [c for c in concepts[:8] if c.get('term') and len(c['term'].strip()) > 1]
            if valid_concepts:
                for concept in valid_concepts:
                    term = concept['term'].strip()
                    definition = concept.get('definition', '').strip()
                    if definition:
                        structured_parts.append(f"- **[[{term}]]**: {definition}")
                    else:
                        structured_parts.append(f"- **[[{term}]]**")
            else:
                structured_parts.append("- 暂无提取到有效概念")
        
        # 添加内容部分
        structured_parts.append(f"\n## 详细内容\n\n{content}")
        
        # 添加处理信息
        complexity = analysis.get('complexity', 'medium')
        content_type = analysis.get('content_type', 'general')
        structured_parts.append(f"\n---\n*本文档复杂度: {complexity} | 内容类型: {content_type} | 处理方式: 基础模板*")
        
        return '\n'.join(structured_parts)