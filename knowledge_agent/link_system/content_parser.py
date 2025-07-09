"""
内容解析器 - 处理文档内容解析和概念提取
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
from .data_models import ConceptLink

logger = logging.getLogger(__name__)


class ContentParser:
    """内容解析器，负责文档内容的解析和概念提取"""
    
    def __init__(self):
        self.concept_pattern = re.compile(r'\[\[([^\]]+)\]\]')
    
    def extract_title(self, content: str) -> str:
        """提取文档标题"""
        lines = content.split('\n')
        
        # 方法1：查找第一个 # 标题
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        # 方法2：查找文件开头的非空行作为标题
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('---'):
                if len(line) < 100:  # 标题不应该太长
                    return line
        
        return "无标题"
    
    def extract_concept_links(self, content: str, doc_path: str) -> List[ConceptLink]:
        """提取文档中的概念链接"""
        links = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            matches = self.concept_pattern.findall(line)
            for concept_name in matches:
                # 清理概念名称
                clean_name = self.clean_concept_name(concept_name)
                if clean_name:
                    # 提取上下文
                    context = self.extract_context(line, concept_name)
                    
                    link = ConceptLink(
                        concept_name=clean_name,
                        source_doc=doc_path,
                        target_doc=None,  # 将在后续解析中填充
                        line_number=line_num,
                        context=context,
                        created_at=datetime.now().isoformat()
                    )
                    links.append(link)
        
        return links
    
    def extract_defined_concepts(self, content: str) -> List[str]:
        """提取文档中定义的概念"""
        concepts = []
        
        # 方法1：从核心概念部分提取
        core_concepts = self.extract_core_concepts_section(content)
        concepts.extend(core_concepts)
        
        # 方法2：从标题提取主要概念
        title_concept = self.extract_title_concept(content)
        if title_concept:
            concepts.append(title_concept)
        
        # 方法3：从相关概念部分提取
        related_concepts = self.extract_related_concepts_section(content)
        concepts.extend(related_concepts)
        
        # 去重并返回
        return list(set(concepts))
    
    def extract_core_concepts_section(self, content: str) -> List[str]:
        """从核心概念部分提取概念列表"""
        concepts = []
        
        # 匹配核心概念部分
        patterns = [
            r'## 核心概念\n(.*?)(?=\n## |\n# |$)',
            r'## 相关概念\n(.*?)(?=\n## |\n# |$)',
            r'## 概念定义\n(.*?)(?=\n## |\n# |$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                concept_section = match.group(1)
                # 提取概念名称
                concept_matches = re.findall(r'-\s*\*?\*?\[\[([^\]]+)\]\]', concept_section)
                for concept in concept_matches:
                    clean_concept = self.clean_concept_name(concept)
                    if clean_concept:
                        concepts.append(clean_concept)
        
        return concepts
    
    def extract_related_concepts_section(self, content: str) -> List[str]:
        """从相关概念部分提取概念"""
        concepts = []
        
        # 匹配相关概念部分
        pattern = r'## 相关概念\n(.*?)(?=\n## |\n# |$)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            concept_section = match.group(1)
            # 提取所有 [[概念]] 格式的内容
            concept_matches = self.concept_pattern.findall(concept_section)
            for concept in concept_matches:
                clean_concept = self.clean_concept_name(concept)
                if clean_concept:
                    concepts.append(clean_concept)
        
        return concepts
    
    def extract_title_concept(self, content: str) -> Optional[str]:
        """从标题提取主要概念"""
        title = self.extract_title(content)
        
        # 清理标题，移除常见的修饰词
        title = re.sub(r'^\d+[\.\)]\s*', '', title)  # 移除数字编号
        title = re.sub(r'^第\d+章\s*', '', title)     # 移除章节号
        title = re.sub(r'[：:]\s*.*$', '', title)      # 移除冒号后的内容
        title = title.strip()
        
        # 如果标题是有效的概念名称，返回它
        if title and len(title) > 1 and len(title) < 50:
            return title
        
        return None
    
    def clean_concept_name(self, concept_name: str) -> str:
        """清理概念名称，用于匹配"""
        # 移除前后空格
        name = concept_name.strip()
        
        # 移除特殊字符
        name = re.sub(r'[^\w\s\u4e00-\u9fff\-]', '', name)
        
        # 移除多余的空格
        name = re.sub(r'\s+', ' ', name)
        
        return name.strip()
    
    def extract_context(self, line: str, concept_name: str) -> str:
        """提取概念的上下文"""
        # 查找概念在行中的位置
        concept_pattern = f'[[{concept_name}]]'
        index = line.find(concept_pattern)
        
        if index == -1:
            return line.strip()
        
        # 提取概念前后的文本作为上下文
        start = max(0, index - 30)
        end = min(len(line), index + len(concept_pattern) + 30)
        context = line[start:end].strip()
        
        # 如果上下文太短，使用整行
        if len(context) < 10:
            context = line.strip()
        
        return context
    
    def extract_word_count(self, content: str) -> int:
        """估算文档字数"""
        # 简单的字数统计
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        english_words = len(re.findall(r'[a-zA-Z]+', content))
        
        # 中文字符按字计算，英文按词计算
        return chinese_chars + english_words
    
    def extract_markdown_structure(self, content: str) -> Dict:
        """提取Markdown文档结构"""
        structure = {
            'title': self.extract_title(content),
            'headings': [],
            'sections': {},
            'has_toc': False,
            'has_code_blocks': False,
            'has_tables': False,
            'has_images': False
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # 检测标题
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                heading = {
                    'level': level,
                    'title': title,
                    'line_number': line_num
                }
                structure['headings'].append(heading)
                current_section = title
                structure['sections'][current_section] = []
            
            # 检测各种Markdown元素
            elif line.startswith('```'):
                structure['has_code_blocks'] = True
            elif '|' in line and line.count('|') >= 2:
                structure['has_tables'] = True
            elif line.startswith('!['):
                structure['has_images'] = True
            elif line.lower().startswith('- [目录]') or line.lower().startswith('## 目录'):
                structure['has_toc'] = True
            
            # 将内容归类到当前章节
            elif current_section and line:
                structure['sections'][current_section].append(line)
        
        return structure
    
    def validate_markdown_syntax(self, content: str) -> List[str]:
        """验证Markdown语法，返回问题列表"""
        issues = []
        lines = content.split('\n')
        
        in_code_block = False
        code_block_lang = None
        
        for line_num, line in enumerate(lines, 1):
            # 检查代码块
            if line.strip().startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    code_block_lang = line.strip()[3:].strip()
                else:
                    in_code_block = False
                    code_block_lang = None
            
            # 检查链接格式
            if not in_code_block:
                # 检查损坏的链接
                broken_links = re.findall(r'\[([^\]]*)\]\([^)]*\)', line)
                for link_text in broken_links:
                    if not link_text.strip():
                        issues.append(f"第{line_num}行：空链接文本")
                
                # 检查未闭合的概念链接
                unclosed_concepts = re.findall(r'\[\[([^\]]*)\](?!\])', line)
                if unclosed_concepts:
                    issues.append(f"第{line_num}行：未闭合的概念链接")
                
                # 检查标题格式
                if line.strip().startswith('#'):
                    if not re.match(r'^#+\s+.+', line.strip()):
                        issues.append(f"第{line_num}行：标题格式错误")
        
        # 检查未闭合的代码块
        if in_code_block:
            issues.append("文档结尾存在未闭合的代码块")
        
        return issues
    
    def extract_metadata(self, content: str) -> Dict:
        """提取文档元数据"""
        metadata = {
            'title': self.extract_title(content),
            'word_count': self.extract_word_count(content),
            'concept_count': len(self.concept_pattern.findall(content)),
            'line_count': len(content.split('\n')),
            'character_count': len(content),
            'structure': self.extract_markdown_structure(content),
            'syntax_issues': self.validate_markdown_syntax(content)
        }
        
        return metadata