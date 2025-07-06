"""
结构化构建工作者 - 负责将解析后的内容转换为标准化的知识笔记格式
"""
import re
from typing import Dict, List, Any, Tuple, Set
from agents.base_agent import BaseAgent
from utils.text_processor import TextProcessor, DocumentChunk
import logging

logger = logging.getLogger(__name__)

class StructureBuilder(BaseAgent):
    """结构化构建工作者Agent"""
    
    def __init__(self):
        super().__init__(
            name="结构化构建专家",
            description="将解析后的内容转换为标准化的知识笔记格式"
        )
        self.text_processor = TextProcessor()
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理解析后的内容，生成结构化笔记
        
        Args:
            input_data: {
                "parsed_content": str,      # 解析后的内容
                "content_type": str,        # 内容类型
                "structure": dict,          # 文档结构
                "metadata": dict,           # 元数据
                "chunks": List[str]         # 内容块
            }
        
        Returns:
            {
                "structured_content": str,  # 结构化的Markdown内容
                "concepts": List[dict],     # 提取的概念列表
                "outline": dict,            # 文档大纲
                "tags": List[str],          # 生成的标签
                "metadata": dict            # 更新的元数据
            }
        """
        
        parsed_content = input_data.get("parsed_content", "")
        content_type = input_data.get("content_type", "text")
        structure = input_data.get("structure", {})
        metadata = input_data.get("metadata", {})
        chunks = input_data.get("chunks", [])
        
        # 提取核心概念
        concepts = self._extract_concepts(parsed_content, chunks)
        
        # 生成文档大纲
        outline = self._generate_outline(parsed_content, structure, concepts)
        
        # 构建结构化内容
        structured_content = self._build_structured_content(
            parsed_content, outline, concepts, content_type
        )
        
        # 生成标签
        tags = self._generate_tags(concepts, metadata, content_type)
        
        # 更新元数据
        updated_metadata = self._update_metadata(metadata, concepts, outline)
        
        return {
            "structured_content": structured_content,
            "concepts": concepts,
            "outline": outline,
            "tags": tags,
            "metadata": updated_metadata
        }
    
    def _extract_concepts(self, content: str, chunks: List[str]) -> List[Dict[str, Any]]:
        """提取关键概念"""
        concepts = []
        
        # 1. 基于规则的概念提取
        rule_based_concepts = self._rule_based_concept_extraction(content)
        concepts.extend(rule_based_concepts)
        
        # 2. 基于模式的概念提取
        pattern_based_concepts = self._pattern_based_concept_extraction(content)
        concepts.extend(pattern_based_concepts)
        
        # 3. 基于频率的概念提取
        frequency_based_concepts = self._frequency_based_concept_extraction(content)
        concepts.extend(frequency_based_concepts)
        
        # 4. 🤖 AI增强的概念提取
        ai_concepts = self._ai_enhanced_concept_extraction(content)
        concepts.extend(ai_concepts)
        
        # 去重和评分
        concepts = self._deduplicate_and_score_concepts(concepts)
        
        return concepts
    
    def _rule_based_concept_extraction(self, content: str) -> List[Dict[str, Any]]:
        """基于规则的概念提取"""
        concepts = []
        
        # 定义概念模式
        patterns = [
            # 直接定义模式
            (r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s*[是：]\s*(.{10,100})', 'definition'),
            (r'([^\n]{2,30})\s*[：:]\s*([^\n]{10,200})', 'description'),
            
            # 专业术语模式
            (r'\b([A-Z]{2,})\b', 'acronym'),
            (r'【([^】]+)】', 'term'),
            (r'`([^`]+)`', 'code_term'),
            
            # 中文概念模式
            (r'([一-龟]{2,8})[是指]', 'chinese_concept'),
            (r'所谓([一-龟]{2,15})', 'chinese_definition'),
        ]
        
        for pattern, concept_type in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                if concept_type in ['definition', 'description']:
                    term = match.group(1).strip()
                    definition = match.group(2).strip()
                else:
                    term = match.group(1).strip()
                    definition = ""
                
                if self._is_valid_concept(term):
                    concepts.append({
                        'term': term,
                        'definition': definition,
                        'type': concept_type,
                        'confidence': 0.8 if definition else 0.6,
                        'source': 'rule_based'
                    })
        
        return concepts
    
    def _pattern_based_concept_extraction(self, content: str) -> List[Dict[str, Any]]:
        """基于模式的概念提取"""
        concepts = []
        
        # 提取列表项中的概念
        list_patterns = [
            r'[-*+]\s+\*\*([^*]+)\*\*[：:]\s*([^\n]+)',  # **概念**: 定义
            r'[-*+]\s+([^：:\n]{2,20})[：:]\s*([^\n]+)',  # 概念: 定义
            r'^\d+\.\s+([^：:\n]{2,20})[：:]\s*([^\n]+)', # 1. 概念: 定义
        ]
        
        for pattern in list_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                
                if self._is_valid_concept(term):
                    concepts.append({
                        'term': term,
                        'definition': definition,
                        'type': 'list_item',
                        'confidence': 0.7,
                        'source': 'pattern_based'
                    })
        
        # 提取标题中的概念
        heading_pattern = r'^#{1,6}\s+(.+)$'
        matches = re.finditer(heading_pattern, content, re.MULTILINE)
        for match in matches:
            title = match.group(1).strip()
            # 清理标题中的标记符号
            clean_title = re.sub(r'[【】\[\]()（）]', '', title)
            
            if self._is_valid_concept(clean_title):
                concepts.append({
                    'term': clean_title,
                    'definition': '',
                    'type': 'heading',
                    'confidence': 0.6,
                    'source': 'pattern_based'
                })
        
        return concepts
    
    def _frequency_based_concept_extraction(self, content: str) -> List[Dict[str, Any]]:
        """基于频率的概念提取"""
        concepts = []
        
        # 提取候选词汇
        # 中文词汇
        chinese_terms = re.findall(r'[一-龟]{2,8}', content)
        # 英文词汇
        english_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        # 混合词汇
        mixed_terms = re.findall(r'[A-Za-z]+[一-龟]+|[一-龟]+[A-Za-z]+', content)
        
        all_terms = chinese_terms + english_terms + mixed_terms
        
        # 统计频率
        term_freq = {}
        for term in all_terms:
            term = term.strip()
            if self._is_valid_concept(term):
                term_freq[term] = term_freq.get(term, 0) + 1
        
        # 选择高频词汇作为概念
        min_freq = max(2, len(content) // 2000)  # 动态调整最小频率
        for term, freq in term_freq.items():
            if freq >= min_freq:
                confidence = min(0.9, 0.3 + freq * 0.1)
                concepts.append({
                    'term': term,
                    'definition': '',
                    'type': 'high_frequency',
                    'confidence': confidence,
                    'frequency': freq,
                    'source': 'frequency_based'
                })
        
        return concepts
    
    def _is_valid_concept(self, term: str) -> bool:
        """验证是否为有效概念"""
        term = term.strip()
        
        # 基本过滤条件
        if len(term) < 2 or len(term) > 50:
            return False
        
        # 排除常见停用词
        stop_words = {
            '的', '了', '在', '是', '我', '你', '他', '她', '它', '们', '这', '那', '就', '都', '要', '可以', '没有',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            '如何', '什么', '怎么', '为什么', '哪里', '什么时候', '怎样', '多少'
        }
        
        if term.lower() in stop_words:
            return False
        
        # 排除纯数字、标点符号
        if term.isdigit() or not re.search(r'[a-zA-Z一-龟]', term):
            return False
        
        # 排除过于通用的词汇
        generic_terms = {
            '方法', '系统', '问题', '内容', '信息', '数据', '结果', '过程', '功能', '技术',
            'method', 'system', 'problem', 'content', 'information', 'data', 'result', 'process'
        }
        
        if term.lower() in generic_terms:
            return False
        
        return True
    
    def _ai_enhanced_concept_extraction(self, content: str) -> List[Dict[str, Any]]:
        """🤖 AI增强的概念提取"""
        concepts = []
        
        try:
            logger.info("开始AI增强概念提取...")
            
            # 构建提示词
            messages = [
                {
                    "role": "system",
                    "content": self.get_system_prompt()
                },
                {
                    "role": "user", 
                    "content": f"""请分析以下内容，提取其中的核心概念和重要术语：

{content[:2000]}  # 限制内容长度

请以JSON格式返回概念列表，每个概念包含以下字段：
- term: 概念名称
- definition: 概念定义或解释（如果内容中包含）
- type: 概念类型（如：technical_term, concept, methodology等）
- confidence: 置信度(0-1)

示例：
[
  {{"term": "机器学习", "definition": "让计算机系统自动改进性能的方法", "type": "technical_term", "confidence": 0.9}},
  {{"term": "神经网络", "definition": "", "type": "technical_term", "confidence": 0.8}}
]

只返回JSON数组，不要其他文字。"""
                }
            ]
            
            # 调用AI模型
            response = self.call_llm(messages, max_tokens=1000)
            
            if response:
                logger.info(f"AI概念提取响应: {response[:200]}...")
                
                # 解析AI响应
                try:
                    import json
                    ai_concepts_data = json.loads(response)
                    
                    for concept_data in ai_concepts_data:
                        if isinstance(concept_data, dict) and 'term' in concept_data:
                            concept = {
                                'term': concept_data.get('term', '').strip(),
                                'definition': concept_data.get('definition', '').strip(),
                                'type': concept_data.get('type', 'ai_extracted'),
                                'confidence': float(concept_data.get('confidence', 0.7)),
                                'source': 'ai_enhanced'
                            }
                            
                            # 验证概念有效性
                            if self._is_valid_concept(concept['term']):
                                concepts.append(concept)
                                
                    logger.info(f"AI成功提取了 {len(concepts)} 个概念")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"AI响应JSON解析失败: {e}")
                    # 尝试简单文本解析
                    concepts.extend(self._parse_ai_response_fallback(response))
                    
        except Exception as e:
            logger.error(f"AI概念提取失败: {e}")
            
        return concepts
    
    def _parse_ai_response_fallback(self, response: str) -> List[Dict[str, Any]]:
        """AI响应的备用解析方法"""
        concepts = []
        
        # 尝试提取类似概念的模式
        patterns = [
            r'"term":\s*"([^"]+)"',
            r'概念[:：]\s*([^\n,，]{2,20})',
            r'术语[:：]\s*([^\n,，]{2,20})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                term = match.strip()
                if self._is_valid_concept(term):
                    concepts.append({
                        'term': term,
                        'definition': '',
                        'type': 'ai_extracted_fallback',
                        'confidence': 0.6,
                        'source': 'ai_enhanced_fallback'
                    })
        
        return concepts
    
    def _deduplicate_and_score_concepts(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重和评分概念"""
        # 按词汇合并相似概念
        concept_groups = {}
        
        for concept in concepts:
            term = concept['term']
            key = term.lower().strip()
            
            if key not in concept_groups:
                concept_groups[key] = []
            concept_groups[key].append(concept)
        
        # 合并同组概念
        merged_concepts = []
        for group in concept_groups.values():
            # 选择最好的概念
            best_concept = max(group, key=lambda x: x['confidence'])
            
            # 合并定义
            definitions = [c['definition'] for c in group if c['definition']]
            if definitions:
                best_concept['definition'] = max(definitions, key=len)
            
            # 合并类型
            types = list(set(c['type'] for c in group))
            best_concept['types'] = types
            
            # 计算最终分数
            final_score = best_concept['confidence']
            if len(definitions) > 1:
                final_score += 0.1  # 多定义加分
            if len(types) > 1:
                final_score += 0.1  # 多类型加分
                
            best_concept['final_score'] = min(1.0, final_score)
            merged_concepts.append(best_concept)
        
        # 按分数排序，取前20个
        merged_concepts.sort(key=lambda x: x['final_score'], reverse=True)
        return merged_concepts[:20]
    
    def _generate_outline(self, content: str, structure: Dict[str, Any], concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成文档大纲"""
        outline = {
            "title": "",
            "sections": [],
            "main_concepts": [],
            "complexity": "medium"
        }
        
        # 提取标题
        title = self._extract_main_title(content)
        outline["title"] = title
        
        # 基于现有结构生成章节
        if structure.get("headings"):
            outline["sections"] = self._structure_to_sections(structure["headings"], content)
        else:
            # 基于内容生成章节
            outline["sections"] = self._generate_sections_from_content(content, concepts)
        
        # 选择主要概念
        main_concepts = [c for c in concepts if c['final_score'] > 0.7][:8]
        outline["main_concepts"] = [c['term'] for c in main_concepts]
        
        # 评估复杂度
        outline["complexity"] = self._assess_complexity(content, concepts, outline["sections"])
        
        return outline
    
    def _extract_main_title(self, content: str) -> str:
        """提取主标题"""
        lines = content.split('\n')
        
        # 查找第一个标题
        for line in lines[:10]:  # 只查看前10行
            line = line.strip()
            
            # Markdown一级标题
            if line.startswith('# '):
                return line[2:].strip()
            
            # 其他可能的标题格式
            if len(line) > 5 and len(line) < 100:
                if (line.isupper() or 
                    line.endswith('：') or 
                    line.endswith(':') or
                    re.match(r'^[一二三四五六七八九十\d]+[、．.]', line)):
                    return line.strip('：:')
        
        # 使用第一个非空行
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) > 3:
                return line[:50]  # 限制长度
        
        return "知识笔记"
    
    def _structure_to_sections(self, headings: List[Dict[str, Any]], content: str) -> List[Dict[str, Any]]:
        """将现有结构转换为章节"""
        sections = []
        
        for heading in headings:
            if heading['level'] <= 3:  # 只处理前三级标题
                sections.append({
                    'title': heading['title'],
                    'level': heading['level'],
                    'type': 'heading_based'
                })
        
        return sections
    
    def _generate_sections_from_content(self, content: str, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从内容生成章节"""
        sections = []
        
        # 基于概念密度分割
        high_score_concepts = [c for c in concepts if c['final_score'] > 0.6]
        
        if len(high_score_concepts) >= 3:
            # 生成基于概念的章节
            sections.append({'title': '核心概念', 'level': 2, 'type': 'concept_based'})
            sections.append({'title': '详细说明', 'level': 2, 'type': 'content_based'})
            sections.append({'title': '相关链接', 'level': 2, 'type': 'link_based'})
        else:
            # 生成通用章节
            sections.append({'title': '主要内容', 'level': 2, 'type': 'content_based'})
        
        return sections
    
    def _assess_complexity(self, content: str, concepts: List[Dict[str, Any]], sections: List[Dict[str, Any]]) -> str:
        """评估内容复杂度"""
        factors = {
            'length': len(content),
            'concepts': len(concepts),
            'sections': len(sections),
            'technical_terms': len([c for c in concepts if c['type'] in ['acronym', 'code_term']])
        }
        
        score = 0
        if factors['length'] > 5000: score += 1
        if factors['concepts'] > 10: score += 1
        if factors['sections'] > 5: score += 1
        if factors['technical_terms'] > 3: score += 1
        
        if score >= 3:
            return "advanced"
        elif score >= 1:
            return "intermediate"
        else:
            return "beginner"
    
    def _build_structured_content(self, content: str, outline: Dict[str, Any], 
                                concepts: List[Dict[str, Any]], content_type: str) -> str:
        """构建结构化内容"""
        
        # 根据内容类型选择模板
        if content_type == "conversation":
            return self._build_conversation_format(content, outline, concepts)
        elif content_type == "url":
            return self._build_article_format(content, outline, concepts)
        else:
            return self._build_general_format(content, outline, concepts)
    
    def _build_conversation_format(self, content: str, outline: Dict[str, Any], 
                                 concepts: List[Dict[str, Any]]) -> str:
        """构建对话格式的结构化内容"""
        sections = []
        
        # 标题
        title = outline.get('title', '对话记录整理')
        sections.append(f"# {title}")
        sections.append("")
        
        # 元数据
        sections.append("## 对话信息")
        sections.append(f"- **类型**: 对话记录")
        sections.append(f"- **复杂度**: {outline.get('complexity', 'medium')}")
        sections.append(f"- **主要概念数**: {len(concepts)}")
        sections.append("")
        
        # 核心概念
        if concepts:
            sections.append("## 核心概念")
            sections.append("")
            for concept in concepts[:8]:
                term = concept['term']
                definition = concept.get('definition', '')
                if definition:
                    sections.append(f"- **[[{term}]]**: {definition}")
                else:
                    sections.append(f"- **[[{term}]]**")
            sections.append("")
        
        # 对话内容
        sections.append("## 对话内容")
        sections.append("")
        
        # 处理对话内容，添加概念链接
        processed_content = self._add_concept_links(content, concepts)
        sections.append(processed_content)
        sections.append("")
        
        # 知识关联
        if len(concepts) > 3:
            sections.append("## 知识链")
            sections.append("")
            concept_terms = [f"[[{c['term']}]]" for c in concepts[:6]]
            sections.append(" → ".join(concept_terms))
            sections.append("")
        
        return "\n".join(sections)
    
    def _build_article_format(self, content: str, outline: Dict[str, Any], 
                            concepts: List[Dict[str, Any]]) -> str:
        """构建文章格式的结构化内容"""
        sections = []
        
        # 标题
        title = outline.get('title', '文章整理')
        sections.append(f"# {title}")
        sections.append("")
        
        # 元数据
        sections.append("## 文章信息")
        sections.append(f"- **类型**: 文章整理")
        sections.append(f"- **复杂度**: {outline.get('complexity', 'medium')}")
        sections.append(f"- **主要概念**: {len(concepts)}个")
        sections.append("")
        
        # 核心概念定义
        if concepts:
            sections.append("## 核心概念")
            sections.append("")
            for concept in concepts[:10]:
                term = concept['term']
                definition = concept.get('definition', '')
                if definition and len(definition) > 10:
                    sections.append(f"### [[{term}]]")
                    sections.append(definition)
                    sections.append("")
                else:
                    sections.append(f"- **[[{term}]]**")
            sections.append("")
        
        # 主要内容
        sections.append("## 主要内容")
        sections.append("")
        
        # 添加概念链接并分段
        processed_content = self._add_concept_links(content, concepts)
        formatted_content = self._format_content_sections(processed_content, outline['sections'])
        sections.append(formatted_content)
        sections.append("")
        
        # 知识关联
        if len(concepts) > 2:
            sections.append("## 知识链接")
            sections.append("")
            
            # 主要概念链
            main_concepts = [c['term'] for c in concepts[:5] if c['final_score'] > 0.7]
            if main_concepts:
                sections.append("### 主要概念链")
                concept_links = [f"[[{term}]]" for term in main_concepts]
                sections.append(" → ".join(concept_links))
                sections.append("")
            
            # 相关概念
            related_concepts = [c['term'] for c in concepts[5:10]]
            if related_concepts:
                sections.append("### 相关概念")
                for term in related_concepts:
                    sections.append(f"- [[{term}]]")
                sections.append("")
        
        return "\n".join(sections)
    
    def _build_general_format(self, content: str, outline: Dict[str, Any], 
                            concepts: List[Dict[str, Any]]) -> str:
        """构建通用格式的结构化内容"""
        sections = []
        
        # 标题
        title = outline.get('title', '知识整理')
        sections.append(f"# {title}")
        sections.append("")
        
        # 概念定义
        if concepts:
            sections.append("## 核心概念")
            sections.append("")
            for concept in concepts[:8]:
                term = concept['term']
                definition = concept.get('definition', '')
                if definition:
                    sections.append(f"- **[[{term}]]**: {definition}")
                else:
                    sections.append(f"- **[[{term}]]**")
            sections.append("")
        
        # 主要内容
        sections.append("## 详细内容")
        sections.append("")
        processed_content = self._add_concept_links(content, concepts)
        sections.append(processed_content)
        sections.append("")
        
        return "\n".join(sections)
    
    def _add_concept_links(self, content: str, concepts: List[Dict[str, Any]]) -> str:
        """在内容中添加概念链接"""
        processed_content = content
        
        # 按长度排序，先处理长概念避免被短概念覆盖
        sorted_concepts = sorted(concepts, key=lambda x: len(x['term']), reverse=True)
        
        for concept in sorted_concepts[:15]:  # 限制处理数量
            term = concept['term']
            
            # 避免在已有链接中添加链接
            if f"[[{term}]]" not in processed_content:
                # 使用词边界匹配，避免部分匹配
                pattern = r'\b' + re.escape(term) + r'\b'
                replacement = f"[[{term}]]"
                
                # 只替换第一次出现，避免过度链接
                processed_content = re.sub(pattern, replacement, processed_content, count=1)
        
        return processed_content
    
    def _format_content_sections(self, content: str, sections: List[Dict[str, Any]]) -> str:
        """格式化内容章节"""
        if not sections:
            return content
        
        # 如果已有明确章节结构，保持原样
        if any('heading_based' == s.get('type') for s in sections):
            return content
        
        # 否则，尝试智能分段
        paragraphs = content.split('\n\n')
        if len(paragraphs) <= 3:
            return content
        
        # 按段落分组
        formatted_sections = []
        paragraphs_per_section = max(1, len(paragraphs) // len(sections))
        
        for i, section in enumerate(sections):
            start_idx = i * paragraphs_per_section
            end_idx = start_idx + paragraphs_per_section
            if i == len(sections) - 1:  # 最后一个section包含剩余所有段落
                end_idx = len(paragraphs)
            
            section_content = '\n\n'.join(paragraphs[start_idx:end_idx])
            if section_content.strip():
                formatted_sections.append(f"### {section['title']}")
                formatted_sections.append("")
                formatted_sections.append(section_content)
                formatted_sections.append("")
        
        return '\n'.join(formatted_sections)
    
    def _generate_tags(self, concepts: List[Dict[str, Any]], metadata: Dict[str, Any], content_type: str) -> List[str]:
        """生成标签"""
        tags = set()
        
        # 基于内容类型的标签
        type_tags = {
            'conversation': ['#对话', '#AI问答'],
            'url': ['#文章', '#外部链接'],
            'markdown': ['#文档', '#整理'],
            'text': ['#文本', '#笔记']
        }
        
        tags.update(type_tags.get(content_type, ['#知识']))
        
        # 基于复杂度的标签
        complexity = metadata.get('difficulty', 'intermediate')
        complexity_tags = {
            'beginner': '#初级',
            'intermediate': '#中级', 
            'advanced': '#高级'
        }
        tags.add(complexity_tags.get(complexity, '#中级'))
        
        # 基于概念的标签
        technical_concepts = [c for c in concepts if c['type'] in ['acronym', 'code_term']]
        if technical_concepts:
            tags.add('#技术')
        
        ai_related_terms = ['AI', '人工智能', '机器学习', '深度学习', 'LLM', 'GPT', 'RAG']
        if any(term in str(concepts) for term in ai_related_terms):
            tags.add('#AI')
        
        # 基于主题的标签（从metadata的topics中获取）
        topics = metadata.get('topics', [])
        for topic in topics[:3]:  # 最多3个主题标签
            if topic and len(topic) < 10:
                tags.add(f'#{topic}')
        
        return sorted(list(tags))
    
    def _update_metadata(self, metadata: Dict[str, Any], concepts: List[Dict[str, Any]], 
                        outline: Dict[str, Any]) -> Dict[str, Any]:
        """更新元数据"""
        updated_metadata = metadata.copy()
        
        updated_metadata.update({
            'processed_time': None,  # 实际实现时添加时间戳
            'concept_count': len(concepts),
            'main_concepts': [c['term'] for c in concepts[:5]],
            'outline_sections': len(outline.get('sections', [])),
            'complexity_assessed': outline.get('complexity', 'medium'),
            'processing_version': '1.0'
        })
        
        return updated_metadata