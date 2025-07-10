"""
知识处理器 - 提供传统的知识处理和AI增强处理功能
"""
import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional, Callable
from openai import OpenAI
from config import settings
from .progress_tracker import SimpleProgressTracker
from .ai_orchestrator import AIToolOrchestrator

logger = logging.getLogger(__name__)

# 停止检查函数
def check_should_stop(task_id: str):
    """检查任务是否应该停止"""
    if not task_id:
        return
    
    try:
        # 导入api_server的active_tasks
        from api_server import active_tasks
        if task_id in active_tasks and active_tasks[task_id] == "stopped":
            from api_server import ProcessingStoppedException
            raise ProcessingStoppedException(f"任务 {task_id} 被用户停止")
    except ImportError:
        # 如果无法导入，跳过检查
        pass


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
        """处理内容的主要方法 - 支持AI编排和传统处理"""
        
        # 检查是否启用AI编排
        if options and options.get("enable_ai_orchestration", False):
            try:
                # 使用AI工具编排器
                orchestrator = AIToolOrchestrator(self.websocket_broadcast)
                return await orchestrator.process_content_with_orchestration(content, content_type, metadata, options)
            except Exception as e:
                logger.error(f"AI编排处理失败，降级到传统处理: {e}")
                # 继续执行传统处理逻辑
        
        # 传统处理逻辑（保持不变）
        tracker = SimpleProgressTracker(self.websocket_broadcast)
        task_id = options.get("task_id") if options else None
        
        try:
            # 检查是否应该停止
            check_should_stop(task_id)
            
            # 阶段1: 分析阶段
            await tracker.update_progress("analysis", "AI分析内容中...", stage_progress=0.5, workers=["AI分析器"])
            
            # 使用AI分析内容
            analysis = await self._ai_analyze_content(content)
            await tracker.update_progress("analysis", "AI分析完成", stage_progress=1.0)
            tracker._complete_stage("analysis")
            
            # 检查是否应该停止
            check_should_stop(task_id)
            
            # 阶段2: 概念提取阶段
            await tracker.update_progress("extraction", "AI概念提取中...", stage_progress=0.5, workers=["概念提取器"])
            
            concepts = await self._ai_extract_concepts(content)
            await tracker.update_progress("extraction", "概念提取完成", stage_progress=1.0)
            tracker._complete_stage("extraction")
            
            # 检查是否应该停止
            check_should_stop(task_id)
            
            # 阶段3: 结构化处理阶段
            await tracker.update_progress("enhancement", "结构化构建中...", stage_progress=0.5, workers=["结构构建器"])
            
            structured_content = await self._ai_structure_content(content, concepts, analysis)
            await tracker.update_progress("enhancement", "结构化完成", stage_progress=1.0)
            tracker._complete_stage("enhancement")
            
            # 检查是否应该停止
            check_should_stop(task_id)
            
            # 阶段4: 合成阶段
            await tracker.update_progress("synthesis", "保存处理结果...", stage_progress=0.5, workers=["文件管理器"])
            
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
            
            await tracker.update_progress("synthesis", "处理完成", stage_progress=1.0)
            tracker._complete_stage("synthesis")
            
            await tracker.update_progress("completed", "处理完成！", progress_percent=100)
            
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
                    max_tokens=1000,
                    temperature=0
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
                    temperature=0
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
                            "content": """你是一个专业的知识整理专家。请将内容重新组织为结构化的Markdown格式。

要求：
1. 生成清晰的五部分结构
2. 为重要概念添加双链格式：[[概念名]]
3. 完整保留原始内容
4. 生成相关反向链接
5. 提取扩展知识点

输出格式：
# 标题

## 相关反向链接
- [[相关概念1]] - 关联说明
- [[相关概念2]] - 关联说明

## 相关概念
- [[概念A]]：定义
- [[概念B]]：定义

## 原始内容
原始输入内容（完全保留，不做任何修改）

## 扩展知识
- 扩展知识点1
- 扩展知识点2"""
                        },
                        {
                            "role": "user",
                            "content": f"""请将以下内容重新整理为结构化的Markdown格式：

原始内容：
{content}

主要概念：{', '.join(concept_names)}

请严格按照以下五部分结构生成文档：

1. **相关反向链接**：基于主要概念，生成可能相关的主题和概念链接
2. **相关概念**：使用[[双链]]格式列出所有重要概念及其定义
3. **原始内容**：原始输入内容完全保留，不做任何修改或删减
4. **扩展知识**：从原始内容中识别出的可以深入学习的知识点

注意：
- 标题应该基于内容主题
- 反向链接应该基于核心概念生成相关主题
- 原始内容部分必须完全保留原始输入
- 扩展知识应该提取文章中涉及但未详细展开的知识点"""
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.3
                )
            )
            
            structured = response.choices[0].message.content
            logger.info(f"AI结构化完成，长度: {len(structured)}")
            
            # 验证AI输出是否包含所有必需的部分
            required_sections = ["## 相关反向链接", "## 相关概念", "## 原始内容", "## 扩展知识"]
            missing_sections = []
            for section in required_sections:
                if section not in structured:
                    missing_sections.append(section)
            
            if missing_sections:
                logger.warning(f"AI结构化输出缺少以下部分: {missing_sections}, 使用fallback方法")
                return self._create_fallback_structure(content, concepts, analysis)
            
            # 验证原始内容部分是否包含原始内容
            if content.strip() not in structured:
                logger.warning("AI结构化输出没有完整保留原始内容，使用fallback方法")
                return self._create_fallback_structure(content, concepts, analysis)
            
            return structured
            
        except Exception as e:
            logger.error(f"AI结构化失败: {e}")
            # 使用fallback方法
            return self._create_fallback_structure(content, concepts, analysis)
    
    def _create_fallback_structure(self, content: str, concepts: List[Dict] = None, analysis: Dict[str, Any] = None) -> str:
        """创建fallback结构化内容"""
        if concepts is None:
            concepts = []
        if analysis is None:
            analysis = {}
            
        main_topic = analysis.get('main_topic', '知识整理')
        if main_topic in ["未知主题", "分析失败"]:
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
        
        # 构建五部分结构
        structured_parts = [f"# {main_topic}"]
        
        # 1. 相关反向链接
        structured_parts.append("\n## 相关反向链接\n")
        if concepts and len(concepts) > 0:
            # 基于概念生成简单的反向链接
            valid_concepts = [c for c in concepts[:5] if c.get('term') and len(c['term'].strip()) > 1]
            if valid_concepts:
                for concept in valid_concepts:
                    term = concept['term'].strip()
                    concept_type = concept.get('type', 'general')
                    structured_parts.append(f"- [[{term}]] - {concept_type}相关主题")
            else:
                structured_parts.append("- 暂无相关链接")
        else:
            structured_parts.append("- 暂无相关链接")
        
        # 2. 相关概念
        structured_parts.append("\n## 相关概念\n")
        if concepts and len(concepts) > 0:
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
        else:
            structured_parts.append("- 暂无提取到有效概念")
        
        # 3. 原始内容（完全保留原始输入）
        structured_parts.append(f"\n## 原始内容\n\n{content}")
        
        # 4. 扩展知识
        structured_parts.append("\n## 扩展知识\n")
        # 基于内容和概念生成简单的扩展知识
        if concepts and len(concepts) > 0:
            concept_types = list(set([c.get('type', 'general') for c in concepts if c.get('type')]))
            if concept_types:
                for concept_type in concept_types[:3]:
                    structured_parts.append(f"- {concept_type}相关的深入学习")
            else:
                structured_parts.append("- 相关领域的深入学习")
        else:
            structured_parts.append("- 相关领域的深入学习")
        
        # 添加处理信息
        complexity = analysis.get('complexity', 'medium')
        content_type = analysis.get('content_type', 'general')
        structured_parts.append(f"\n---\n*本文档复杂度: {complexity} | 内容类型: {content_type} | 处理方式: 基础模板*")
        
        return '\n'.join(structured_parts)