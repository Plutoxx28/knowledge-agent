"""
长文本处理模块 - 实现层次化和流式处理策略
"""
import re
from typing import List, Dict, Any, Tuple, Generator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """文档块数据类"""
    content: str
    chunk_id: int
    level: int  # 层级：1=章节, 2=段落, 3=句子
    metadata: Dict[str, Any]
    start_pos: int
    end_pos: int

@dataclass
class DocumentStructure:
    """文档结构数据类"""
    headings: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]
    total_length: int
    complexity_score: float

class TextProcessor:
    """文本处理器 - 实现智能分块和结构分析"""
    
    def __init__(self, max_chunk_size: int = 3000, overlap_size: int = 500):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
    def analyze_document_structure(self, content: str) -> DocumentStructure:
        """分析文档结构"""
        headings = self._extract_headings(content)
        sections = self._identify_sections(content, headings)
        complexity_score = self._calculate_complexity(content, headings)
        
        return DocumentStructure(
            headings=headings,
            sections=sections,
            total_length=len(content),
            complexity_score=complexity_score
        )
    
    def choose_processing_strategy(self, content: str) -> str:
        """选择最优处理策略"""
        structure = self.analyze_document_structure(content)
        
        # 基于文档特征选择策略
        if len(content) < self.max_chunk_size:
            return "direct"  # 直接处理
        elif len(structure.headings) >= 3 and structure.complexity_score > 0.3:
            return "hierarchical"  # 层次化处理
        elif self._is_sequential_content(content):
            return "streaming"  # 流式处理
        else:
            return "hybrid"  # 混合策略
    
    def hierarchical_processing(self, content: str) -> List[DocumentChunk]:
        """层次化处理"""
        structure = self.analyze_document_structure(content)
        chunks = []
        
        # Level 1: 按主要章节分割
        for i, section in enumerate(structure.sections):
            section_content = content[section['start']:section['end']]
            
            if len(section_content) <= self.max_chunk_size:
                # 章节内容适中，直接作为一个块
                chunks.append(DocumentChunk(
                    content=section_content,
                    chunk_id=len(chunks),
                    level=1,
                    metadata={"section_title": section.get('title', f'Section {i+1}')},
                    start_pos=section['start'],
                    end_pos=section['end']
                ))
            else:
                # 章节过长，进一步分割
                sub_chunks = self._split_section_hierarchically(
                    section_content, section['start'], section.get('title', f'Section {i+1}')
                )
                chunks.extend(sub_chunks)
        
        return chunks
    
    def streaming_processing(self, content: str) -> Generator[DocumentChunk, None, None]:
        """流式处理生成器"""
        position = 0
        chunk_id = 0
        
        while position < len(content):
            # 计算当前块的结束位置
            end_pos = min(position + self.max_chunk_size, len(content))
            
            # 寻找合适的分割点
            if end_pos < len(content):
                end_pos = self._find_split_point(content, position, end_pos)
            
            # 提取块内容
            chunk_content = content[position:end_pos].strip()
            
            if chunk_content:
                yield DocumentChunk(
                    content=chunk_content,
                    chunk_id=chunk_id,
                    level=2,
                    metadata={"stream_position": position},
                    start_pos=position,
                    end_pos=end_pos
                )
                chunk_id += 1
            
            # 更新位置，考虑重叠
            position = max(position + 1, end_pos - self.overlap_size)
    
    def hybrid_processing(self, content: str) -> List[DocumentChunk]:
        """混合策略处理"""
        structure = self.analyze_document_structure(content)
        chunks = []
        
        # 对结构化部分使用层次化处理
        structured_sections = [s for s in structure.sections if s.get('has_structure', False)]
        sequential_parts = []
        
        last_end = 0
        for section in structured_sections:
            # 处理结构化部分之前的内容
            if section['start'] > last_end:
                sequential_content = content[last_end:section['start']].strip()
                if sequential_content:
                    sequential_parts.append((last_end, section['start'], sequential_content))
            
            # 处理结构化部分
            section_content = content[section['start']:section['end']]
            section_chunks = self._split_section_hierarchically(
                section_content, section['start'], section.get('title', 'Section')
            )
            chunks.extend(section_chunks)
            
            last_end = section['end']
        
        # 处理最后的顺序内容
        if last_end < len(content):
            sequential_content = content[last_end:].strip()
            if sequential_content:
                sequential_parts.append((last_end, len(content), sequential_content))
        
        # 对顺序部分使用流式处理
        for start, end, seq_content in sequential_parts:
            seq_chunks = list(self._streaming_split(seq_content, start))
            chunks.extend(seq_chunks)
        
        # 按位置排序
        chunks.sort(key=lambda x: x.start_pos)
        
        # 重新分配chunk_id
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = i
        
        return chunks
    
    def _extract_headings(self, content: str) -> List[Dict[str, Any]]:
        """提取标题"""
        headings = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Markdown标题
            md_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if md_match:
                level = len(md_match.group(1))
                title = md_match.group(2).strip()
                headings.append({
                    'level': level,
                    'title': title,
                    'line_number': i,
                    'type': 'markdown'
                })
                continue
            
            # 检测其他标题格式
            if self._is_likely_heading(line):
                # 根据格式推断层级
                level = self._infer_heading_level(line)
                headings.append({
                    'level': level,
                    'title': line.strip(),
                    'line_number': i,
                    'type': 'inferred'
                })
        
        return headings
    
    def _identify_sections(self, content: str, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别章节"""
        lines = content.split('\n')
        sections = []
        
        for i, heading in enumerate(headings):
            start_line = heading['line_number']
            
            # 找到下一个同级或更高级标题
            end_line = len(lines)
            for j in range(i + 1, len(headings)):
                if headings[j]['level'] <= heading['level']:
                    end_line = headings[j]['line_number']
                    break
            
            # 计算字符位置
            start_pos = sum(len(lines[k]) + 1 for k in range(start_line))
            end_pos = sum(len(lines[k]) + 1 for k in range(end_line))
            
            sections.append({
                'title': heading['title'],
                'level': heading['level'],
                'start': start_pos,
                'end': min(end_pos, len(content)),
                'start_line': start_line,
                'end_line': end_line,
                'has_structure': heading['level'] <= 3
            })
        
        return sections
    
    def _calculate_complexity(self, content: str, headings: List[Dict[str, Any]]) -> float:
        """计算文档复杂度"""
        factors = {
            'length': min(len(content) / 10000, 1.0),  # 长度因子
            'headings': min(len(headings) / 20, 1.0),   # 标题数量因子
            'nested_depth': 0,                          # 嵌套深度因子
            'list_density': 0,                          # 列表密度因子
            'code_blocks': 0                            # 代码块因子
        }
        
        # 计算嵌套深度
        if headings:
            max_level = max(h['level'] for h in headings)
            min_level = min(h['level'] for h in headings)
            factors['nested_depth'] = min((max_level - min_level) / 5, 1.0)
        
        # 计算列表密度
        list_lines = len(re.findall(r'^\s*[-*+]\s+', content, re.MULTILINE))
        total_lines = len(content.split('\n'))
        if total_lines > 0:
            factors['list_density'] = min(list_lines / total_lines, 1.0)
        
        # 计算代码块密度
        code_blocks = len(re.findall(r'```.*?```', content, re.DOTALL))
        factors['code_blocks'] = min(code_blocks / 10, 1.0)
        
        # 加权平均
        weights = {'length': 0.3, 'headings': 0.25, 'nested_depth': 0.2, 
                  'list_density': 0.15, 'code_blocks': 0.1}
        
        complexity = sum(factors[k] * weights[k] for k in factors)
        return complexity
    
    def _is_sequential_content(self, content: str) -> bool:
        """判断是否为顺序内容（如对话、日志）"""
        patterns = [
            r'用户[:：]\s*',
            r'助手[:：]\s*',
            r'\d{4}-\d{2}-\d{2}',  # 日期格式
            r'\d{2}:\d{2}:\d{2}',  # 时间格式
            r'^\d+\.\s+',          # 有序列表
        ]
        
        matches = 0
        for pattern in patterns:
            if re.search(pattern, content[:2000], re.MULTILINE):
                matches += 1
        
        return matches >= 2
    
    def _split_section_hierarchically(self, content: str, start_offset: int, section_title: str) -> List[DocumentChunk]:
        """层次化分割章节"""
        chunks = []
        
        if len(content) <= self.max_chunk_size:
            return [DocumentChunk(
                content=content,
                chunk_id=0,
                level=2,
                metadata={"section_title": section_title},
                start_pos=start_offset,
                end_pos=start_offset + len(content)
            )]
        
        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', content)
        current_chunk = ""
        chunk_start = start_offset
        chunk_id = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= self.max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content=current_chunk,
                        chunk_id=chunk_id,
                        level=2,
                        metadata={"section_title": section_title, "chunk_type": "paragraph_group"},
                        start_pos=chunk_start,
                        end_pos=chunk_start + len(current_chunk)
                    ))
                    chunk_id += 1
                    chunk_start += len(current_chunk) + 2
                
                # 开始新块
                current_chunk = para
        
        # 添加最后一块
        if current_chunk:
            chunks.append(DocumentChunk(
                content=current_chunk,
                chunk_id=chunk_id,
                level=2,
                metadata={"section_title": section_title, "chunk_type": "paragraph_group"},
                start_pos=chunk_start,
                end_pos=chunk_start + len(current_chunk)
            ))
        
        return chunks
    
    def _streaming_split(self, content: str, start_offset: int) -> Generator[DocumentChunk, None, None]:
        """流式分割内容"""
        position = 0
        chunk_id = 0
        
        while position < len(content):
            end_pos = min(position + self.max_chunk_size, len(content))
            
            if end_pos < len(content):
                end_pos = self._find_split_point(content, position, end_pos)
            
            chunk_content = content[position:end_pos].strip()
            
            if chunk_content:
                yield DocumentChunk(
                    content=chunk_content,
                    chunk_id=chunk_id,
                    level=3,
                    metadata={"chunk_type": "streaming"},
                    start_pos=start_offset + position,
                    end_pos=start_offset + end_pos
                )
                chunk_id += 1
            
            position = max(position + 1, end_pos - self.overlap_size)
    
    def _find_split_point(self, content: str, start: int, max_end: int) -> int:
        """寻找最佳分割点"""
        # 优先级：段落 > 句子 > 词语
        
        # 1. 寻找段落边界
        for i in range(max_end - 1, start + self.max_chunk_size // 2, -1):
            if content[i:i+2] == '\n\n':
                return i + 2
        
        # 2. 寻找句子边界
        sentence_endings = '.。!！?？'
        for i in range(max_end - 1, start + self.max_chunk_size // 2, -1):
            if content[i] in sentence_endings and i + 1 < len(content):
                if content[i + 1].isspace():
                    return i + 1
        
        # 3. 寻找词语边界
        for i in range(max_end - 1, start + self.max_chunk_size // 2, -1):
            if content[i].isspace():
                return i + 1
        
        # 4. 硬分割
        return max_end
    
    def _is_likely_heading(self, line: str) -> bool:
        """判断是否可能是标题"""
        line = line.strip()
        
        # 排除明显不是标题的情况
        if len(line) > 100 or len(line) < 3:
            return False
        
        if line.endswith(('.', '。', '!', '！', '?', '？')):
            return False
        
        # 检查是否包含标题特征
        heading_indicators = [
            line.isupper(),  # 全大写
            re.match(r'^[一二三四五六七八九十\d]+[、．.]', line),  # 数字编号
            line.endswith('：') or line.endswith(':'),  # 冒号结尾
            len(line.split()) <= 8,  # 词数较少
        ]
        
        return sum(heading_indicators) >= 2
    
    def _infer_heading_level(self, line: str) -> int:
        """推断标题级别"""
        line = line.strip()
        
        # 根据编号推断
        if re.match(r'^[一二三四五六七八九十]+[、．.]', line):
            return 1
        elif re.match(r'^\d+[、．.]', line):
            return 2
        elif re.match(r'^\(\d+\)', line):
            return 3
        
        # 根据长度和格式推断
        if len(line) < 15 and (line.isupper() or line.endswith(':')):
            return 1
        elif len(line) < 30:
            return 2
        else:
            return 3