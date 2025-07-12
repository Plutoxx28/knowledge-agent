"""
内容分析器 - 统一的内容分析工具
"""
import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """统一的内容分析工具"""
    
    def __init__(self, ai_client: Optional[OpenAI] = None):
        """
        初始化内容分析器
        
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
                        "X-Title": "Knowledge Agent - Content Analyzer",
                    }
                )
                logger.info("内容分析器AI客户端初始化成功")
            except Exception as e:
                logger.error(f"内容分析器AI客户端初始化失败: {e}")
                self.ai_client = None
        else:
            self.ai_client = ai_client
    
    async def analyze_content(self, content: str, method: str = "ai_enhanced") -> Dict[str, Any]:
        """
        分析内容
        
        Args:
            content: 要分析的内容
            method: 分析方法 ("ai_enhanced", "fallback", "hybrid")
            
        Returns:
            分析结果，包含主题、复杂度、类型、关键主题等信息
        """
        if method == "ai_enhanced" and self.ai_client:
            return await self._ai_analyze_content(content)
        elif method == "hybrid":
            # 尝试AI分析，失败时降级到fallback
            try:
                return await self._ai_analyze_content(content)
            except Exception as e:
                logger.warning(f"AI内容分析失败，使用fallback方法: {e}")
                return self._fallback_content_analysis(content)
        else:
            return self._fallback_content_analysis(content)
    
    async def _ai_analyze_content(self, content: str) -> Dict[str, Any]:
        """AI增强的内容分析"""
        if not self.ai_client:
            raise Exception("AI客户端未初始化")
            
        try:
            logger.info(f"开始AI内容分析... 使用模型: {settings.model_name}")
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.ai_client.chat.completions.create(
                    model=settings.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的内容分析专家。请深度分析给定内容的主题、复杂度、类型和结构特征。"
                        },
                        {
                            "role": "user",
                            "content": f"""请深度分析以下内容：

{content[:2000]}

请以JSON格式返回详细的分析结果：
{{
  "main_topic": "主要话题",
  "complexity": "simple|medium|complex",
  "content_type": "technical|educational|general|academic|conversational|documentation",
  "key_themes": ["主题1", "主题2", "主题3"],
  "structure_type": "linear|hierarchical|conversational|fragmented",
  "domain": "技术领域|学科领域",
  "audience_level": "beginner|intermediate|advanced|expert",
  "information_density": "low|medium|high",
  "actionable_content": true/false,
  "requires_context": true/false,
  "language_style": "formal|informal|technical|conversational",
  "word_count": 估计字数,
  "estimated_reading_time": 估计阅读时间(分钟)
}}

分析说明：
- complexity: 基于概念深度、术语复杂度、逻辑层次判断
- content_type: 基于内容性质和用途分类
- structure_type: 基于内容组织方式分类
- domain: 识别内容所属的专业领域
- audience_level: 目标读者的专业水平
- information_density: 信息密度和概念集中度
- actionable_content: 是否包含可执行的指导或方法
- requires_context: 是否需要额外背景知识理解"""
                        }
                    ],
                    max_tokens=65535,
                    temperature=0.1
                )
            )
            
            ai_response = response.choices[0].message.content
            logger.info(f"AI内容分析响应: {ai_response[:200]}...")
            
            try:
                # 清理响应中的markdown标记
                clean_response = ai_response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                analysis = json.loads(clean_response)
                
                # 验证和标准化分析结果
                standardized_analysis = self._standardize_analysis(analysis)
                standardized_analysis['source'] = 'ai_enhanced'
                
                logger.info(f"AI内容分析完成，主题: {standardized_analysis.get('main_topic', '未知')}")
                return standardized_analysis
                
            except json.JSONDecodeError as e:
                logger.warning(f"AI内容分析JSON解析失败: {e}")
                raise Exception(f"JSON解析失败: {e}")
                
        except Exception as e:
            logger.error(f"AI内容分析失败: {e}")
            raise e
    
    def _fallback_content_analysis(self, content: str) -> Dict[str, Any]:
        """备用内容分析方法 - 基于规则的分析"""
        analysis = {
            'main_topic': self._extract_main_topic(content),
            'complexity': self._assess_complexity(content),
            'content_type': self._detect_content_type(content),
            'key_themes': self._extract_key_themes(content),
            'structure_type': self._analyze_structure(content),
            'domain': self._identify_domain(content),
            'audience_level': self._assess_audience_level(content),
            'information_density': self._assess_information_density(content),
            'actionable_content': self._has_actionable_content(content),
            'requires_context': self._requires_context(content),
            'language_style': self._analyze_language_style(content),
            'word_count': len(content.split()),
            'estimated_reading_time': max(1, len(content.split()) // 200),
            'source': 'fallback_analysis'
        }
        
        return analysis
    
    def _standardize_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """标准化分析结果"""
        standardized = {}
        
        # 标准化各个字段
        standardized['main_topic'] = str(analysis.get('main_topic', '内容分析')).strip()
        
        # 复杂度标准化
        complexity = analysis.get('complexity', 'medium').lower()
        if complexity in ['simple', 'easy', 'basic']:
            standardized['complexity'] = 'simple'
        elif complexity in ['complex', 'hard', 'difficult', 'advanced']:
            standardized['complexity'] = 'complex'
        else:
            standardized['complexity'] = 'medium'
        
        # 内容类型标准化
        content_type = analysis.get('content_type', 'general').lower()
        valid_types = ['technical', 'educational', 'general', 'academic', 'conversational', 'documentation']
        standardized['content_type'] = content_type if content_type in valid_types else 'general'
        
        # 主题列表
        themes = analysis.get('key_themes', [])
        if isinstance(themes, list):
            standardized['key_themes'] = [str(theme).strip() for theme in themes[:5]]
        else:
            standardized['key_themes'] = []
        
        # 结构类型
        structure = analysis.get('structure_type', 'linear').lower()
        valid_structures = ['linear', 'hierarchical', 'conversational', 'fragmented']
        standardized['structure_type'] = structure if structure in valid_structures else 'linear'
        
        # 其他字段
        standardized['domain'] = str(analysis.get('domain', '通用')).strip()
        standardized['audience_level'] = analysis.get('audience_level', 'intermediate')
        standardized['information_density'] = analysis.get('information_density', 'medium')
        standardized['actionable_content'] = bool(analysis.get('actionable_content', False))
        standardized['requires_context'] = bool(analysis.get('requires_context', False))
        standardized['language_style'] = analysis.get('language_style', 'formal')
        standardized['word_count'] = int(analysis.get('word_count', 0))
        standardized['estimated_reading_time'] = max(1, int(analysis.get('estimated_reading_time', 1)))
        
        return standardized
    
    def _extract_main_topic(self, content: str) -> str:
        """提取主要话题"""
        # 尝试从标题提取
        lines = content.strip().split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line.startswith('#'):
                return line.lstrip('#').strip()
            elif len(line) > 5 and len(line) < 100 and not line.startswith((' ', '\t')):
                return line
        
        # 从内容中提取关键词
        words = re.findall(r'[一-龟]{2,}|[A-Za-z]{3,}', content)
        if words:
            # 简单统计词频，返回最常见的词作为主题
            from collections import Counter
            word_freq = Counter(words)
            most_common = word_freq.most_common(3)
            return ' '.join([word for word, _ in most_common])
        
        return "内容分析"
    
    def _assess_complexity(self, content: str) -> str:
        """评估内容复杂度"""
        complexity_score = 0
        
        # 技术术语密度
        tech_terms = len(re.findall(r'[A-Z]{2,}|API|SDK|HTTP|JSON|XML|SQL', content))
        complexity_score += min(tech_terms / 10, 2)
        
        # 句子长度
        sentences = re.split(r'[。！？.!?]', content)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if avg_sentence_length > 20:
            complexity_score += 1
        
        # 专业词汇
        complex_patterns = [
            r'[\u4e00-\u9fff]{4,}',  # 长中文词汇
            r'[A-Za-z]{8,}',  # 长英文单词
            r'\d+\.\d+',  # 数字
            r'[（(][^）)]{5,}[）)]'  # 长括号内容
        ]
        
        for pattern in complex_patterns:
            matches = len(re.findall(pattern, content))
            complexity_score += matches / 20
        
        if complexity_score < 1:
            return 'simple'
        elif complexity_score < 3:
            return 'medium'
        else:
            return 'complex'
    
    def _detect_content_type(self, content: str) -> str:
        """检测内容类型"""
        content_lower = content.lower()
        
        # 技术内容指标
        tech_indicators = ['api', 'code', '代码', '函数', '方法', '算法', '数据库', '编程', 'python', 'javascript']
        tech_score = sum(1 for indicator in tech_indicators if indicator in content_lower)
        
        # 教育内容指标
        edu_indicators = ['学习', '教程', '课程', '练习', '作业', '考试', '知识', '理解', '掌握']
        edu_score = sum(1 for indicator in edu_indicators if indicator in content_lower)
        
        # 学术内容指标
        academic_indicators = ['研究', '论文', '实验', '分析', '假设', '结论', '参考文献', '摘要']
        academic_score = sum(1 for indicator in academic_indicators if indicator in content_lower)
        
        # 对话内容指标
        conv_indicators = ['你好', 'hello', '请问', '回答', '问题', '解释一下', '能否']
        conv_score = sum(1 for indicator in conv_indicators if indicator in content_lower)
        
        # 文档内容指标
        doc_indicators = ['使用说明', '安装', '配置', '步骤', '注意事项', '要求', '规范']
        doc_score = sum(1 for indicator in doc_indicators if indicator in content_lower)
        
        scores = {
            'technical': tech_score,
            'educational': edu_score,
            'academic': academic_score,
            'conversational': conv_score,
            'documentation': doc_score
        }
        
        max_type = max(scores, key=scores.get)
        return max_type if scores[max_type] > 0 else 'general'
    
    def _extract_key_themes(self, content: str) -> List[str]:
        """提取关键主题"""
        # 提取高频词汇作为主题
        import re
        from collections import Counter
        
        # 提取有意义的词汇
        words = re.findall(r'[一-龟]{2,}|[A-Za-z]{3,}', content)
        
        # 过滤停用词
        stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '而', '及', '以及', 'the', 'and', 'or', 'but', 'with', 'for', 'to', 'of', 'in', 'on', 'at', 'by'}
        words = [word for word in words if word.lower() not in stop_words]
        
        # 统计词频并返回前5个
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(5)]
    
    def _analyze_structure(self, content: str) -> str:
        """分析内容结构"""
        lines = content.split('\n')
        
        # 检查是否有层次结构（标题）
        has_headers = any(line.strip().startswith('#') for line in lines)
        if has_headers:
            return 'hierarchical'
        
        # 检查是否为对话格式
        dialogue_patterns = [r'^[A-Za-z\u4e00-\u9fff]+[:：]', r'^> ', r'^\d+\.', r'^- ']
        dialogue_count = sum(1 for line in lines if any(re.match(pattern, line.strip()) for pattern in dialogue_patterns))
        
        if dialogue_count > len(lines) * 0.3:
            return 'conversational'
        
        # 检查是否为碎片化内容
        short_lines = sum(1 for line in lines if len(line.strip()) < 50)
        if short_lines > len(lines) * 0.6:
            return 'fragmented'
        
        return 'linear'
    
    def _identify_domain(self, content: str) -> str:
        """识别内容领域"""
        content_lower = content.lower()
        
        domains = {
            '计算机科学': ['编程', '算法', '数据结构', 'python', 'java', 'javascript', '代码', '函数'],
            '人工智能': ['机器学习', '深度学习', '神经网络', 'ai', 'ml', 'dl', '模型', '训练'],
            '数据科学': ['数据分析', '统计', '可视化', '数据挖掘', 'sql', '数据库', '图表'],
            '商业管理': ['管理', '营销', '战略', '业务', '客户', '市场', '销售', '团队'],
            '教育': ['学习', '教学', '课程', '教育', '培训', '知识', '技能', '学生'],
            '健康医疗': ['健康', '医疗', '疾病', '治疗', '药物', '症状', '医生', '患者'],
            '科学研究': ['研究', '实验', '理论', '假设', '数据', '结果', '分析', '论文']
        }
        
        domain_scores = {}
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return '通用'
    
    def _assess_audience_level(self, content: str) -> str:
        """评估目标受众水平"""
        # 基于词汇复杂度和概念深度判断
        complexity = self._assess_complexity(content)
        content_type = self._detect_content_type(content)
        
        if complexity == 'simple' or '入门' in content or '基础' in content:
            return 'beginner'
        elif complexity == 'complex' or content_type == 'academic':
            return 'expert'
        elif '高级' in content or '进阶' in content:
            return 'advanced'
        else:
            return 'intermediate'
    
    def _assess_information_density(self, content: str) -> str:
        """评估信息密度"""
        words = content.split()
        sentences = re.split(r'[。！？.!?]', content)
        
        # 计算平均句子长度
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # 计算概念密度（技术术语、专有名词等）
        concept_patterns = [r'[A-Z]{2,}', r'[一-龟]{3,}', r'\d+\.\d+', r'[A-Za-z]{6,}']
        concept_count = sum(len(re.findall(pattern, content)) for pattern in concept_patterns)
        concept_density = concept_count / max(len(words), 1)
        
        if avg_sentence_length > 15 and concept_density > 0.1:
            return 'high'
        elif avg_sentence_length < 8 and concept_density < 0.05:
            return 'low'
        else:
            return 'medium'
    
    def _has_actionable_content(self, content: str) -> bool:
        """检查是否包含可执行内容"""
        actionable_indicators = [
            '步骤', '方法', '如何', '怎么', '教程', '指南', '操作', '实践',
            'step', 'how to', 'tutorial', 'guide', 'method', 'process'
        ]
        
        return any(indicator in content.lower() for indicator in actionable_indicators)
    
    def _requires_context(self, content: str) -> bool:
        """检查是否需要背景知识"""
        context_indicators = [
            '前面提到', '如前所述', '之前', '上文', '参考', '基于',
            'as mentioned', 'previously', 'refer to', 'based on'
        ]
        
        return any(indicator in content.lower() for indicator in context_indicators)
    
    def _analyze_language_style(self, content: str) -> str:
        """分析语言风格"""
        # 正式性指标
        formal_indicators = ['因此', '然而', '此外', '综上所述', '基于', '根据']
        formal_score = sum(1 for indicator in formal_indicators if indicator in content)
        
        # 非正式性指标
        informal_indicators = ['哈哈', '嗯', '呃', '好的', '谢谢', '不错']
        informal_score = sum(1 for indicator in informal_indicators if indicator in content)
        
        # 技术性指标
        technical_indicators = ['函数', '变量', '参数', '返回值', '调用', 'function', 'parameter']
        technical_score = sum(1 for indicator in technical_indicators if indicator in content)
        
        # 对话性指标
        conversational_indicators = ['你', '我', '我们', '请', '能否', '可以', 'you', 'we', 'can']
        conversational_score = sum(1 for indicator in conversational_indicators if indicator in content)
        
        scores = {
            'formal': formal_score,
            'informal': informal_score,
            'technical': technical_score,
            'conversational': conversational_score
        }
        
        max_style = max(scores, key=scores.get)
        return max_style if scores[max_style] > 0 else 'formal'