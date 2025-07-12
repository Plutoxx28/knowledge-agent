"""
结构构建器 - 统一的内容结构化工具
"""
import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)


class StructureBuilder:
    """统一的内容结构化工具"""
    
    def __init__(self, ai_client: Optional[OpenAI] = None):
        """
        初始化结构构建器
        
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
                        "X-Title": "Knowledge Agent - Structure Builder",
                    }
                )
                logger.info("结构构建器AI客户端初始化成功")
            except Exception as e:
                logger.error(f"结构构建器AI客户端初始化失败: {e}")
                self.ai_client = None
        else:
            self.ai_client = ai_client
    
    async def build_structure(self, content: str, concepts: List[Dict] = None, 
                            analysis: Dict[str, Any] = None, method: str = "ai_enhanced") -> str:
        """
        构建结构化内容
        
        Args:
            content: 原始内容
            concepts: 概念列表
            analysis: 内容分析结果
            method: 构建方法 ("ai_enhanced", "fallback", "hybrid")
            
        Returns:
            结构化的Markdown内容
        """
        if concepts is None:
            concepts = []
        if analysis is None:
            analysis = {}
            
        if method == "ai_enhanced" and self.ai_client:
            return await self._ai_build_structure(content, concepts, analysis)
        elif method == "hybrid":
            # 尝试AI构建，失败时降级到fallback
            try:
                return await self._ai_build_structure(content, concepts, analysis)
            except Exception as e:
                logger.warning(f"AI结构构建失败，使用fallback方法: {e}")
                return self._fallback_structure_building(content, concepts, analysis)
        else:
            return self._fallback_structure_building(content, concepts, analysis)
    
    async def _ai_build_structure(self, content: str, concepts: List[Dict], 
                                analysis: Dict[str, Any]) -> str:
        """AI增强的结构构建"""
        if not self.ai_client:
            raise Exception("AI客户端未初始化")
            
        try:
            logger.info(f"开始AI结构构建... 使用模型: {settings.model_name}")
            
            # 准备概念信息
            concept_text = self._format_concepts_for_ai(concepts)
            analysis_text = self._format_analysis_for_ai(analysis)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ai_client.chat.completions.create(
                    model=settings.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": """你是一个专业的知识整理专家。请将内容重新组织为高质量的结构化Markdown格式。

要求：
1. 生成清晰的知识笔记结构
2. 概念使用普通文本格式，不要使用[[]]双链
3. 完整保留原始内容
4. 生成相关反向链接
5. 提取详细具体的扩展知识点
6. 保持逻辑清晰和内容完整

标准输出格式：
# 标题

## 相关反向链接
- [[相关概念1]] - 关联说明
- [[相关概念2]] - 关联说明

## 相关概念
- **概念A**：定义和解释
- **概念B**：定义和解释

## 原始内容
原始输入内容（完全保留，不做任何修改）

## 扩展知识
- 相关技术概念1
- 相关技术概念2
- 相关技术概念3

注意事项：
- 标题应该简洁明确，体现内容核心主题
- 反向链接要基于核心概念生成相关主题
- 概念定义要准确清晰，使用加粗格式突出概念名称
- 原始内容部分必须完全保留原始输入
- 扩展知识直接列出具体的技术概念、术语或相关主题，不要包含描述性文字"""
                        },
                        {
                            "role": "user",
                            "content": f"""请将以下内容重新整理为结构化的知识笔记：

原始内容：
{content}

内容分析信息：
{analysis_text}

提取的概念：
{concept_text}

请严格按照标准格式生成高质量的结构化文档，确保：
1. 标题准确反映内容主题
2. 反向链接基于核心概念生成相关主题
3. 概念部分使用**加粗**格式突出概念名称，不要使用[[]]双链
4. 原始内容部分完全保留原始输入
5. 扩展知识直接列出具体的概念名称，如技术术语、相关概念等

特别注意：
- 原始内容部分必须原样保留，不能有任何修改、删减或重新组织
- 扩展知识只列出概念名称，不要包含"的应用"、"相关学习"等描述性文字"""
                        }
                    ],
                    max_tokens=65535,
                    temperature=0.3
                )
            )
            
            structured = response.choices[0].message.content
            logger.info(f"AI结构构建完成，长度: {len(structured)}")
            
            # 验证AI输出的完整性
            validation_result = self._validate_ai_structure(structured, content)
            if not validation_result['valid']:
                logger.warning(f"AI结构化验证失败: {validation_result['reason']}")
                return self._fallback_structure_building(content, concepts, analysis)
            
            return structured
            
        except Exception as e:
            logger.error(f"AI结构构建失败: {e}")
            raise e
    
    def _fallback_structure_building(self, content: str, concepts: List[Dict], 
                                   analysis: Dict[str, Any]) -> str:
        """备用结构构建方法 - 基于模板的构建"""
        logger.info("使用备用模板方法构建结构")
        
        # 提取或生成标题
        title = self._extract_title(content, analysis)
        
        # 构建五部分结构
        sections = []
        sections.append(f"# {title}")
        
        # 1. 相关反向链接
        sections.append("\n## 相关反向链接\n")
        backlinks = self._generate_backlinks(concepts, analysis)
        if backlinks:
            sections.extend(backlinks)
        else:
            sections.append("- 暂无相关链接")
        
        # 2. 相关概念
        sections.append("\n## 相关概念\n")
        concept_lines = self._format_concepts(concepts)
        if concept_lines:
            sections.extend(concept_lines)
        else:
            sections.append("- 暂无提取到有效概念")
        
        # 3. 原始内容（完全保留原始输入）
        sections.append(f"\n## 原始内容\n\n{content}")
        
        # 4. 扩展知识
        sections.append("\n## 扩展知识\n")
        extensions = self._generate_extensions(concepts, analysis, content)
        if extensions:
            sections.extend(extensions)
        else:
            sections.append("- 相关领域的深入学习")
        
        # 添加处理信息
        processing_info = self._generate_processing_info(analysis)
        sections.append(f"\n---\n{processing_info}")
        
        return '\n'.join(sections)
    
    def _format_concepts_for_ai(self, concepts: List[Dict]) -> str:
        """为AI格式化概念信息"""
        if not concepts:
            return "暂无概念信息"
        
        concept_lines = []
        for i, concept in enumerate(concepts[:10], 1):
            term = concept.get('term', '')
            definition = concept.get('definition', '')
            concept_type = concept.get('type', '')
            confidence = concept.get('confidence', 0.0)
            
            line = f"{i}. {term}"
            if definition:
                line += f" - {definition}"
            if concept_type:
                line += f" ({concept_type})"
            line += f" [置信度: {confidence:.2f}]"
            
            concept_lines.append(line)
        
        return '\n'.join(concept_lines)
    
    def _format_analysis_for_ai(self, analysis: Dict[str, Any]) -> str:
        """为AI格式化分析信息"""
        if not analysis:
            return "暂无分析信息"
        
        info_lines = []
        key_mapping = {
            'main_topic': '主要话题',
            'complexity': '复杂度',
            'content_type': '内容类型',
            'key_themes': '关键主题',
            'domain': '领域',
            'audience_level': '受众水平',
            'information_density': '信息密度',
            'language_style': '语言风格'
        }
        
        for key, label in key_mapping.items():
            value = analysis.get(key)
            if value:
                if isinstance(value, list):
                    value = ', '.join(map(str, value))
                info_lines.append(f"- {label}: {value}")
        
        return '\n'.join(info_lines) if info_lines else "暂无分析信息"
    
    def _validate_ai_structure(self, structured: str, original_content: str) -> Dict[str, Any]:
        """验证AI生成的结构化内容"""
        validation = {'valid': True, 'reason': ''}
        
        # 检查必需的部分
        required_sections = ["## 相关反向链接", "## 相关概念", "## 原始内容", "## 扩展知识"]
        missing_sections = []
        
        for section in required_sections:
            if section not in structured:
                missing_sections.append(section)
        
        if missing_sections:
            validation['valid'] = False
            validation['reason'] = f"缺少必需部分: {', '.join(missing_sections)}"
            return validation
        
        # 检查原始内容是否被保留
        if original_content.strip() not in structured:
            validation['valid'] = False
            validation['reason'] = "原始内容未完整保留"
            return validation
        
        # 检查结构完整性
        if not structured.startswith('#'):
            validation['valid'] = False
            validation['reason'] = "缺少文档标题"
            return validation
        
        return validation
    
    def _extract_title(self, content: str, analysis: Dict[str, Any]) -> str:
        """提取或生成标题"""
        # 优先使用分析结果中的主题
        main_topic = analysis.get('main_topic', '')
        if main_topic and main_topic not in ['内容分析', '分析失败', '未知主题']:
            return main_topic
        
        # 尝试从内容中提取标题
        lines = content.strip().split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line.startswith('#'):
                return line.lstrip('#').strip()
            elif len(line) > 5 and len(line) < 100 and not line.startswith((' ', '\t')):
                # 简单的标题识别逻辑
                if any(char in line for char in '：:。.？?！!'):
                    continue
                return line
        
        # 基于内容类型生成默认标题
        content_type = analysis.get('content_type', 'general')
        type_titles = {
            'technical': '技术文档',
            'educational': '学习笔记',
            'academic': '学术资料',
            'conversational': '对话记录',
            'documentation': '说明文档'
        }
        
        return type_titles.get(content_type, '知识整理')
    
    def _generate_backlinks(self, concepts: List[Dict], analysis: Dict[str, Any]) -> List[str]:
        """生成反向链接"""
        backlinks = []
        
        # 基于概念生成反向链接
        for concept in concepts[:5]:
            term = concept.get('term', '').strip()
            concept_type = concept.get('type', 'general')
            
            if term and len(term) > 1:
                # 根据概念类型生成不同的关联说明
                type_descriptions = {
                    'technical_term': '技术相关主题',
                    'concept': '概念相关内容',
                    'methodology': '方法论主题',
                    'framework': '框架相关内容',
                    'principle': '原则相关主题',
                    'proper_noun': '相关实体',
                    'chinese_term': '相关术语',
                    'acronym': '缩写相关内容'
                }
                
                description = type_descriptions.get(concept_type, '相关主题')
                backlinks.append(f"- [[{term}]] - {description}")
        
        # 基于分析结果生成额外的反向链接
        domain = analysis.get('domain', '')
        if domain and domain != '通用':
            backlinks.append(f"- [[{domain}]] - 领域相关内容")
        
        key_themes = analysis.get('key_themes', [])
        for theme in key_themes[:2]:
            if theme and theme not in [c.get('term', '') for c in concepts]:
                backlinks.append(f"- [[{theme}]] - 主题相关内容")
        
        return backlinks[:8]  # 最多8个反向链接
    
    def _format_concepts(self, concepts: List[Dict]) -> List[str]:
        """格式化概念列表"""
        concept_lines = []
        
        for concept in concepts[:8]:  # 最多8个概念
            term = concept.get('term', '').strip()
            definition = concept.get('definition', '').strip()
            
            if term and len(term) > 1:
                if definition and len(definition) > 3:
                    concept_lines.append(f"- **{term}**: {definition}")
                else:
                    concept_lines.append(f"- **{term}**")
        
        return concept_lines
    
    def _generate_extensions(self, concepts: List[Dict], analysis: Dict[str, Any], 
                           content: str) -> List[str]:
        """基于原文内容生成扩展知识概念列表"""
        extensions = []
        existing_terms = set([c.get('term', '').lower() for c in concepts])
        
        # 1. 从原文直接提取技术术语和专业词汇
        import re
        
        # 提取英文技术术语
        tech_patterns = [
            r'\b([A-Z]{2,})\b',  # 缩写词：API, HTTP, JSON
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b',  # 驼峰命名：WebSocket, JavaScript
            r'\b([a-z]+(?:-[a-z]+)+)\b',  # 连字符术语：client-server
        ]
        
        tech_terms = []
        for pattern in tech_patterns:
            matches = re.findall(pattern, content)
            tech_terms.extend(matches)
        
        # 提取中文专业术语（3-6字）
        chinese_terms = re.findall(r'[一-龟]{3,6}', content)
        
        # 过滤并添加新发现的术语
        all_terms = tech_terms + chinese_terms
        for term in all_terms:
            if (term.lower() not in existing_terms and 
                len(term) > 2 and len(term) < 20 and
                not term.isdigit() and
                term not in ['内容', '处理', '分析', '系统', '方法', '工具']):
                extensions.append(f"- {term}")
                existing_terms.add(term.lower())
                if len(extensions) >= 3:
                    break
        
        # 2. 基于内容主题和关键词生成相关概念
        content_lower = content.lower()
        main_topic = analysis.get('main_topic', '')
        
        # 技术领域相关概念
        if any(keyword in content_lower for keyword in ['api', 'protocol', 'mcp', 'rpc']):
            related_concepts = ['JSON-RPC', 'gRPC', 'REST API', 'GraphQL', 'Protocol Buffer']
            extensions.extend([f"- {concept}" for concept in related_concepts[:2] 
                             if concept.lower() not in existing_terms])
        
        elif any(keyword in content_lower for keyword in ['数据库', 'database', 'sql']):
            related_concepts = ['索引优化', '查询计划', '事务管理', 'ORM框架', '数据迁移']
            extensions.extend([f"- {concept}" for concept in related_concepts[:2]
                             if concept.lower() not in existing_terms])
        
        elif any(keyword in content_lower for keyword in ['agent', '代理', '智能', 'ai']):
            related_concepts = ['机器学习', '自然语言处理', '知识图谱', '推理引擎', '决策树']
            extensions.extend([f"- {concept}" for concept in related_concepts[:2]
                             if concept.lower() not in existing_terms])
        
        elif any(keyword in content_lower for keyword in ['web', 'http', '网络', 'socket']):
            related_concepts = ['负载均衡', '反向代理', 'CDN', 'WebSocket', '消息队列']
            extensions.extend([f"- {concept}" for concept in related_concepts[:2]
                             if concept.lower() not in existing_terms])
        
        # 3. 基于内容类型添加领域概念
        content_type = analysis.get('content_type', 'general')
        domain = analysis.get('domain', '')
        
        if content_type == 'technical':
            if '框架' in content or 'framework' in content_lower:
                tech_concepts = ['微服务架构', '容器化', '持续集成', '服务网格']
            elif '算法' in content or 'algorithm' in content_lower:
                tech_concepts = ['时间复杂度', '空间复杂度', '动态规划', '贪心算法']
            else:
                tech_concepts = ['设计模式', '代码重构', '单元测试', '性能优化']
            
            extensions.extend([f"- {concept}" for concept in tech_concepts[:2]
                             if concept.lower() not in existing_terms])
        
        # 4. 从关键主题提取扩展概念
        key_themes = analysis.get('key_themes', [])
        for theme in key_themes[:2]:
            if (theme and theme.lower() not in existing_terms and 
                len(theme) > 2 and theme not in main_topic):
                extensions.append(f"- {theme}")
                existing_terms.add(theme.lower())
        
        # 5. 基于文档结构提取概念
        if '配置' in content or 'config' in content_lower:
            config_concepts = ['环境变量', '配置文件', '参数验证']
            extensions.extend([f"- {concept}" for concept in config_concepts[:1]
                             if concept.lower() not in existing_terms])
        
        if '测试' in content or 'test' in content_lower:
            test_concepts = ['集成测试', '端到端测试', '测试覆盖率']
            extensions.extend([f"- {concept}" for concept in test_concepts[:1]
                             if concept.lower() not in existing_terms])
        
        # 6. 如果扩展概念不够，添加通用相关概念
        if len(extensions) < 3:
            general_concepts = []
            
            if domain == '计算机科学':
                general_concepts = ['数据结构', '编程范式', '软件工程', '系统架构']
            elif domain == '人工智能':
                general_concepts = ['深度学习', '强化学习', '计算机视觉', '语音识别']
            else:
                # 基于主题生成通用概念
                if main_topic:
                    general_concepts = [f'{main_topic}原理', f'{main_topic}实现', f'{main_topic}应用']
                else:
                    general_concepts = ['技术架构', '实现方案', '最佳实践']
            
            extensions.extend([f"- {concept}" for concept in general_concepts[:3-len(extensions)]
                             if concept.lower() not in existing_terms])
        
        return extensions[:5]  # 最多5个扩展概念
    
    def _generate_processing_info(self, analysis: Dict[str, Any]) -> str:
        """生成处理信息"""
        complexity = analysis.get('complexity', 'medium')
        content_type = analysis.get('content_type', 'general')
        source = analysis.get('source', 'fallback_analysis')
        
        processing_method = "AI增强" if source == 'ai_enhanced' else "基础模板"
        
        info_parts = [
            f"本文档复杂度: {complexity}",
            f"内容类型: {content_type}",
            f"处理方式: {processing_method}"
        ]
        
        return f"*{' | '.join(info_parts)}*"
    
    def generate_outline(self, content: str) -> Dict[str, Any]:
        """生成内容大纲"""
        lines = content.split('\n')
        outline = {
            'title': '',
            'sections': [],
            'structure_type': 'flat'
        }
        
        current_level = 0
        sections = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # 计算标题级别
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                
                if not outline['title'] and level == 1:
                    outline['title'] = title
                    continue
                
                sections.append({
                    'level': level,
                    'title': title,
                    'line': line
                })
                
                if level > current_level + 1:
                    outline['structure_type'] = 'hierarchical'
                current_level = max(current_level, level)
        
        outline['sections'] = sections
        
        if not sections:
            outline['structure_type'] = 'flat'
        elif any(s['level'] > 1 for s in sections):
            outline['structure_type'] = 'hierarchical'
        
        return outline
    
    def extract_metadata(self, content: str, analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """提取内容元数据"""
        metadata = {
            'word_count': len(content.split()),
            'character_count': len(content),
            'line_count': len(content.split('\n')),
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
            'has_code_blocks': bool(re.search(r'```|`[^`]+`', content)),
            'has_links': bool(re.search(r'https?://|www\.', content)),
            'has_images': bool(re.search(r'!\[.*?\]\(.*?\)', content)),
            'has_tables': bool(re.search(r'\|.*\|', content)),
            'has_lists': bool(re.search(r'^\s*[-*+]\s|^\s*\d+\.\s', content, re.MULTILINE)),
            'language_detected': 'mixed' if re.search(r'[a-zA-Z]', content) and re.search(r'[\u4e00-\u9fff]', content) else 'chinese' if re.search(r'[\u4e00-\u9fff]', content) else 'english'
        }
        
        if analysis:
            metadata.update({
                'complexity': analysis.get('complexity', 'medium'),
                'content_type': analysis.get('content_type', 'general'),
                'domain': analysis.get('domain', '通用'),
                'estimated_reading_time': analysis.get('estimated_reading_time', 1)
            })
        
        return metadata