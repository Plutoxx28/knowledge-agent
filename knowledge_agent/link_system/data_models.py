"""
数据模型 - 链接系统的核心数据结构定义
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConceptLink:
    """概念链接信息"""
    concept_name: str           # 概念名称
    source_doc: str            # 来源文档路径
    target_doc: Optional[str]   # 目标文档路径（如果找到对应文档）
    line_number: int           # 在源文档中的行号
    context: str               # 上下文（概念周围的文本）
    created_at: str            # 创建时间


@dataclass
class DocumentMeta:
    """文档元数据"""
    doc_path: str              # 文档路径
    title: str                 # 文档标题
    concepts: List[str]        # 文档包含的概念
    outbound_links: List[str]  # 出站链接（引用的概念）
    inbound_links: List[str]   # 入站链接（被哪些文档引用）
    last_updated: str          # 最后更新时间
    file_hash: str            # 文件内容哈希