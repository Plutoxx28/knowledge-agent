"""
概念提取器 - 统一的概念提取工具
"""
import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)


class ConceptExtractor:
    """统一的概念提取工具"""
    
    def __init__(self, ai_client: Optional[OpenAI] = None):
        """
        初始化概念提取器
        
        Args:
            ai_client: OpenAI客户端实例，如果为None则创建新实例
        """
        if ai_client is None:
            try:
                self.ai_client = OpenAI(
                    base_url=settings.openrouter_base_url,
                    api_key=settings.openrouter_api_key,
                    default_headers={
                        "HTTP-Referer": "https://knowledge-agent.local",
                        "X-Title": "Knowledge Agent - Concept Extractor",
                    }
                )
                logger.info("概念提取器AI客户端初始化成功")
            except Exception as e:
                logger.error(f"概念提取器AI客户端初始化失败: {e}")
                self.ai_client = None
        else:
            self.ai_client = ai_client
    
    async def extract_concepts(self, content: str, method: str = "ai_enhanced") -> List[Dict[str, Any]]:
        """
        提取内容中的概念
        
        Args:
            content: 要提取概念的内容
            method: 提取方法 ("ai_enhanced", "fallback", "hybrid")
            
        Returns:
            概念列表，每个概念包含term, definition, type, confidence, source等字段
        """
        if method == "ai_enhanced" and self.ai_client:
            return await self._ai_extract_concepts(content)
        elif method == "hybrid":
            # 尝试AI提取，失败时降级到fallback
            try:
                return await self._ai_extract_concepts(content)
            except Exception as e:
                logger.warning(f"AI概念提取失败，使用fallback方法: {e}")
                return self._fallback_concept_extraction(content)
        else:
            return self._fallback_concept_extraction(content)
    
    async def _ai_extract_concepts(self, content: str) -> List[Dict[str, Any]]:
        """AI增强的概念提取"""
        if not self.ai_client:
            raise Exception("AI客户端未初始化")
            
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

概念类型包括：
- concept: 抽象概念
- technical_term: 技术术语
- proper_noun: 专有名词
- acronym: 缩写词
- chinese_term: 中文术语
- methodology: 方法论
- framework: 框架
- principle: 原则

请确保提取的概念具有代表性且与内容高度相关。"""
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.2
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
                
                # 验证和增强概念数据
                validated_concepts = []
                for concept in concepts:
                    if isinstance(concept, dict) and concept.get('term'):
                        # 标准化概念数据
                        validated_concept = {
                            'term': concept['term'].strip(),
                            'definition': concept.get('definition', '').strip(),
                            'type': concept.get('type', 'concept'),
                            'confidence': float(concept.get('confidence', 0.7)),
                            'source': 'ai_enhanced',
                            'final_score': float(concept.get('confidence', 0.7))
                        }
                        
                        # 过滤过短或无效的概念
                        if len(validated_concept['term']) > 1:
                            validated_concepts.append(validated_concept)
                
                logger.info(f"AI成功提取了 {len(validated_concepts)} 个有效概念")
                return validated_concepts[:15]  # 最多返回15个概念
                
            except json.JSONDecodeError as e:
                logger.warning(f"AI概念提取JSON解析失败: {e}")
                raise Exception(f"JSON解析失败: {e}")
                
        except Exception as e:
            logger.error(f"AI概念提取失败: {e}")
            raise e
    
    def _fallback_concept_extraction(self, content: str) -> List[Dict[str, Any]]:
        """备用概念提取方法 - 基于规则的提取"""
        concepts = []
        
        # 多种模式的概念提取
        patterns = [
            (r'\b([A-Z]{2,})\b', 'acronym', 0.6),  # 缩写词
            (r'([一-龟]{2,8})', 'chinese_term', 0.5),  # 中文术语
            (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', 'proper_noun', 0.5),  # 专有名词
            (r'([A-Z][a-z]+(?:[-_][A-Z][a-z]+)+)', 'technical_term', 0.7),  # 技术术语
            (r'(\w+(?:\s+\w+)*)\s*[:：]\s*([^。\n]{5,100})', 'definition_pair', 0.8),  # 定义对
        ]
        
        seen_terms = set()  # 避免重复
        
        for pattern, term_type, base_confidence in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            
            for match in matches:
                if isinstance(match, tuple):
                    # 处理定义对
                    if term_type == 'definition_pair':
                        term, definition = match
                        term = term.strip()
                        definition = definition.strip()
                    else:
                        term = match[0] if match[0] else match[1]
                        definition = ''
                else:
                    term = match.strip()
                    definition = ''
                
                # 过滤条件
                if (len(term) > 1 and len(term) < 50 and 
                    term.lower() not in seen_terms and
                    not term.isdigit() and
                    not re.match(r'^[一二三四五六七八九十\d\s\-\.]+$', term)):
                    
                    concepts.append({
                        'term': term,
                        'definition': definition,
                        'type': term_type,
                        'confidence': base_confidence,
                        'source': 'fallback_extraction',
                        'final_score': base_confidence
                    })
                    
                    seen_terms.add(term.lower())
                    
                    # 限制数量
                    if len(concepts) >= 20:
                        break
        
        # 按置信度排序并返回前10个
        concepts.sort(key=lambda x: x['final_score'], reverse=True)
        return concepts[:10]
    
    def enhance_concepts_with_context(self, concepts: List[Dict], content: str) -> List[Dict[str, Any]]:
        """
        基于上下文增强概念信息
        
        Args:
            concepts: 概念列表
            content: 原始内容
            
        Returns:
            增强后的概念列表
        """
        enhanced_concepts = []
        
        for concept in concepts:
            term = concept['term']
            enhanced_concept = concept.copy()
            
            # 在内容中查找概念的上下文
            context_patterns = [
                rf'({re.escape(term)})[是为]([^。，\n]{{5,50}})',  # "概念是xxx"
                rf'([^。，\n]{{5,50}})(?:称为|叫做)({re.escape(term)})',  # "xxx称为概念"
                rf'({re.escape(term)})[:：]([^。\n]{{5,100}})',  # "概念：定义"
                rf'({re.escape(term)})\s*[（(]([^）)\n]{{5,50}})[）)]',  # "概念（定义）"
            ]
            
            for pattern in context_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # 使用找到的上下文更新定义
                    for match in matches[:1]:  # 只取第一个匹配
                        if isinstance(match, tuple) and len(match) == 2:
                            if term.lower() in match[0].lower():
                                definition = match[1].strip()
                            else:
                                definition = match[0].strip()
                            
                            if definition and len(definition) > len(enhanced_concept.get('definition', '')):
                                enhanced_concept['definition'] = definition
                                enhanced_concept['confidence'] = min(1.0, enhanced_concept['confidence'] + 0.1)
                                enhanced_concept['final_score'] = enhanced_concept['confidence']
                    break
            
            enhanced_concepts.append(enhanced_concept)
        
        return enhanced_concepts
    
    def filter_and_rank_concepts(self, concepts: List[Dict], max_count: int = 10) -> List[Dict[str, Any]]:
        """
        过滤和排序概念
        
        Args:
            concepts: 概念列表
            max_count: 最大返回数量
            
        Returns:
            过滤和排序后的概念列表
        """
        # 去重 - 基于term的相似度
        unique_concepts = []
        seen_terms = set()
        
        for concept in concepts:
            term = concept['term'].lower().strip()
            
            # 检查是否与已有概念过于相似
            is_similar = False
            for seen_term in seen_terms:
                if (term in seen_term or seen_term in term) and abs(len(term) - len(seen_term)) < 3:
                    is_similar = True
                    break
            
            if not is_similar:
                unique_concepts.append(concept)
                seen_terms.add(term)
        
        # 按置信度和定义质量排序
        def concept_score(concept):
            base_score = concept.get('final_score', concept.get('confidence', 0.5))
            
            # 有定义的概念加分
            if concept.get('definition') and len(concept['definition']) > 5:
                base_score += 0.2
            
            # AI提取的概念加分
            if concept.get('source') == 'ai_enhanced':
                base_score += 0.1
            
            # 术语类型加权
            type_weights = {
                'technical_term': 0.1,
                'concept': 0.08,
                'methodology': 0.09,
                'framework': 0.08,
                'principle': 0.07,
                'proper_noun': 0.05,
                'chinese_term': 0.03,
                'acronym': 0.02
            }
            base_score += type_weights.get(concept.get('type', ''), 0)
            
            return min(1.0, base_score)
        
        # 排序并返回
        unique_concepts.sort(key=concept_score, reverse=True)
        return unique_concepts[:max_count]