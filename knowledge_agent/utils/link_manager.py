"""
链接管理器 - 负责双向链接系统的实现
处理概念与文档之间的映射关系，支持链接跳转和反向链接
"""

import os
import re
import json
import hashlib
import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)


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


class LinkManager:
    """双向链接管理器"""
    
    def __init__(self, knowledge_base_path: str, db_path: str = None):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.db_path = db_path or str(self.knowledge_base_path / ".link_manager.db")
        self.concept_pattern = re.compile(r'\[\[([^\]]+)\]\]')
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        # 设置数据库连接超时和模式
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        
        try:
            cursor = conn.cursor()
            
            # 文档元数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    doc_path TEXT PRIMARY KEY,
                    title TEXT,
                    concepts TEXT,  -- JSON array
                    outbound_links TEXT,  -- JSON array
                    inbound_links TEXT,   -- JSON array
                    last_updated TEXT,
                    file_hash TEXT
                )
            ''')
            
            # 概念链接表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS concept_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_name TEXT,
                    source_doc TEXT,
                    target_doc TEXT,
                    line_number INTEGER,
                    context TEXT,
                    created_at TEXT,
                    FOREIGN KEY (source_doc) REFERENCES documents (doc_path)
                )
            ''')
            
            # 概念-文档映射表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS concept_documents (
                    concept_name TEXT,
                    doc_path TEXT,
                    is_primary BOOLEAN,  -- 是否是该概念的主文档
                    PRIMARY KEY (concept_name, doc_path)
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concept_name ON concept_links(concept_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_doc ON concept_links(source_doc)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concept_docs ON concept_documents(concept_name)')
            
            conn.commit()
        finally:
            conn.close()
    
    def scan_knowledge_base(self) -> Dict[str, int]:
        """扫描整个知识库，更新链接数据库
        
        Returns:
            Dict[str, int]: 统计信息 {'scanned_files': count, 'total_concepts': count, 'total_links': count}
        """
        stats = {'scanned_files': 0, 'total_concepts': 0, 'total_links': 0}
        
        # 扫描所有Markdown文件
        for md_file in self.knowledge_base_path.rglob("*.md"):
            if self._should_process_file(md_file):
                self._process_document(md_file)
                stats['scanned_files'] += 1
        
        # 解析所有链接
        self._resolve_all_links()
        
        # 计算统计信息
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM concept_documents')
            stats['total_concepts'] = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM concept_links')
            stats['total_links'] = cursor.fetchone()[0]
        
        return stats
    
    def scan_knowledge_base_simple(self) -> Dict[str, int]:
        """简化的知识库扫描，只更新文档信息，不处理复杂的链接关系"""
        stats = {'scanned_files': 0, 'updated_files': 0}
        
        # 扫描所有Markdown文件
        for md_file in self.knowledge_base_path.rglob("*.md"):
            if self._should_process_file(md_file):
                updated = self._process_document_simple(md_file)
                stats['scanned_files'] += 1
                if updated:
                    stats['updated_files'] += 1
        
        return stats
    
    def _process_document_simple(self, doc_path: Path) -> bool:
        """简化的文档处理，只更新基本信息"""
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 计算文件哈希
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # 检查是否需要更新
            if self._is_file_up_to_date(str(doc_path), file_hash):
                return False
            
            # 提取文档标题
            title = self._extract_title(content)
            
            # 更新数据库中的文档信息
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO documents 
                    (doc_path, title, concepts, outbound_links, inbound_links, last_updated, file_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (str(doc_path), title, json.dumps([]), json.dumps([]), 
                      json.dumps([]), datetime.now().isoformat(), file_hash))
                conn.commit()
            
            return True
            
        except Exception as e:
            print(f"处理文档 {doc_path} 时出错: {e}")
            return False
    
    def _should_process_file(self, file_path: Path) -> bool:
        """判断是否应该处理该文件"""
        # 跳过隐藏文件和特殊目录
        if any(part.startswith('.') for part in file_path.parts):
            return False
        
        # 只处理Markdown文件
        return file_path.suffix.lower() == '.md'
    
    def _process_document(self, doc_path: Path):
        """处理单个文档，提取概念和链接"""
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 计算文件哈希
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # 检查是否需要更新
            if self._is_file_up_to_date(str(doc_path), file_hash):
                return
            
            # 提取文档标题
            title = self._extract_title(content)
            
            # 提取所有概念链接
            concept_links = self._extract_concept_links(content, str(doc_path))
            
            # 提取文档定义的概念（通过特定模式识别）
            defined_concepts = self._extract_defined_concepts(content)
            
            # 更新数据库
            self._update_document_in_db(
                doc_path=str(doc_path),
                title=title,
                defined_concepts=defined_concepts,
                concept_links=concept_links,
                file_hash=file_hash
            )
            
        except Exception as e:
            print(f"处理文档 {doc_path} 时出错: {e}")
    
    def _is_file_up_to_date(self, doc_path: str, file_hash: str) -> bool:
        """检查文件是否已是最新版本"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT file_hash FROM documents WHERE doc_path = ?', (doc_path,))
            result = cursor.fetchone()
            return result and result[0] == file_hash
    
    def _extract_title(self, content: str) -> str:
        """提取文档标题"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return "无标题"
    
    def _extract_concept_links(self, content: str, doc_path: str) -> List[ConceptLink]:
        """提取文档中的所有概念链接"""
        links = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            matches = self.concept_pattern.finditer(line)
            for match in matches:
                concept_name = match.group(1).strip()
                if concept_name:  # 非空概念名
                    # 提取上下文（概念周围的文本）
                    context = self._extract_context(line, match.start(), match.end())
                    
                    link = ConceptLink(
                        concept_name=concept_name,
                        source_doc=doc_path,
                        target_doc=None,  # 稍后解析
                        line_number=line_num,
                        context=context,
                        created_at=datetime.now().isoformat()
                    )
                    links.append(link)
        
        return links
    
    def _extract_defined_concepts(self, content: str) -> List[str]:
        """提取文档中定义的概念（主要概念）"""
        defined_concepts = []
        
        # 方法1：从"核心概念"部分提取
        core_concepts_section = self._extract_core_concepts_section(content)
        if core_concepts_section:
            # 查找 - **[[概念名]]**: 定义 的模式
            pattern = r'-\s*\*\*\[\[([^\]]+)\]\]\*\*:\s*(.+)'
            matches = re.findall(pattern, core_concepts_section)
            for concept_name, definition in matches:
                defined_concepts.append(concept_name.strip())
        
        # 方法2：从标题推测主要概念
        title_concept = self._extract_title_concept(content)
        if title_concept and title_concept not in defined_concepts:
            defined_concepts.append(title_concept)
        
        return defined_concepts
    
    def _extract_core_concepts_section(self, content: str) -> Optional[str]:
        """提取核心概念部分的内容"""
        lines = content.split('\n')
        in_core_concepts = False
        core_concepts_lines = []
        
        for line in lines:
            if line.strip().startswith('## 核心概念'):
                in_core_concepts = True
                continue
            elif in_core_concepts and line.strip().startswith('## '):
                break
            elif in_core_concepts:
                core_concepts_lines.append(line)
        
        return '\n'.join(core_concepts_lines) if core_concepts_lines else None
    
    def _extract_title_concept(self, content: str) -> Optional[str]:
        """从标题中提取主要概念"""
        title = self._extract_title(content)
        
        # 清理标题，移除标点和修饰词
        # 如果标题包含冒号，取冒号前的部分
        if '：' in title:
            concept = title.split('：')[0].strip()
        elif ':' in title:
            concept = title.split(':')[0].strip()
        else:
            concept = title.strip()
        
        # 清理常见的修饰词
        cleanup_patterns = [
            r'^关于',
            r'^什么是',
            r'技术详解$',
            r'指南$',
            r'概述$',
            r'介绍$',
            r'详解$',
            r'教程$',
            r'手册$'
        ]
        
        for pattern in cleanup_patterns:
            concept = re.sub(pattern, '', concept).strip()
        
        return concept if concept else None
    
    def _extract_context(self, line: str, start: int, end: int, window: int = 30) -> str:
        """提取概念周围的上下文"""
        context_start = max(0, start - window)
        context_end = min(len(line), end + window)
        return line[context_start:context_end].strip()
    
    def _update_document_in_db(self, doc_path: str, title: str, defined_concepts: List[str], 
                              concept_links: List[ConceptLink], file_hash: str):
        """更新数据库中的文档信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 删除旧的链接记录
            cursor.execute('DELETE FROM concept_links WHERE source_doc = ?', (doc_path,))
            cursor.execute('DELETE FROM concept_documents WHERE doc_path = ?', (doc_path,))
            
            # 插入新的链接记录
            for link in concept_links:
                cursor.execute('''
                    INSERT INTO concept_links 
                    (concept_name, source_doc, target_doc, line_number, context, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (link.concept_name, link.source_doc, link.target_doc, 
                      link.line_number, link.context, link.created_at))
            
            # 插入概念-文档映射
            outbound_concepts = [link.concept_name for link in concept_links]
            for concept in defined_concepts:
                cursor.execute('''
                    INSERT OR REPLACE INTO concept_documents (concept_name, doc_path, is_primary)
                    VALUES (?, ?, ?)
                ''', (concept, doc_path, True))
            
            for concept in outbound_concepts:
                if concept not in defined_concepts:
                    cursor.execute('''
                        INSERT OR REPLACE INTO concept_documents (concept_name, doc_path, is_primary)
                        VALUES (?, ?, ?)
                    ''', (concept, doc_path, False))
            
            # 更新文档元数据
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (doc_path, title, concepts, outbound_links, inbound_links, last_updated, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (doc_path, title, json.dumps(defined_concepts), 
                  json.dumps(outbound_concepts), json.dumps([]), 
                  datetime.now().isoformat(), file_hash))
            
            conn.commit()
    
    def _resolve_all_links(self):
        """解析所有概念链接，找到对应的目标文档"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 获取所有未解析的链接
            cursor.execute('SELECT id, concept_name FROM concept_links WHERE target_doc IS NULL')
            unresolved_links = cursor.fetchall()
            
            for link_id, concept_name in unresolved_links:
                target_doc = self._find_target_document(concept_name)
                if target_doc:
                    cursor.execute('UPDATE concept_links SET target_doc = ? WHERE id = ?', 
                                 (target_doc, link_id))
            
            # 更新反向链接
            self._update_inbound_links()
            
            conn.commit()
    
    def _find_target_document(self, concept_name: str) -> Optional[str]:
        """为概念找到对应的目标文档 - 只匹配标题相关的文档"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 方法1：精确匹配 - 查找标题概念完全匹配的文档
            cursor.execute('''
                SELECT doc_path FROM concept_documents 
                WHERE concept_name = ? AND is_primary = 1
                LIMIT 1
            ''', (concept_name,))
            
            result = cursor.fetchone()
            if result:
                return result[0]
            
            # 方法2：严格的标题匹配 - 检查文档标题是否真的对应该概念
            cursor.execute('''
                SELECT d.doc_path, d.title FROM documents d
            ''')
            
            all_docs = cursor.fetchall()
            
            # 对每个文档，重新提取标题概念并严格匹配
            for doc_path, title in all_docs:
                # 重新计算该文档的主要概念
                if os.path.exists(doc_path):
                    try:
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # 提取该文档的标题概念
                        doc_title_concept = self._extract_title_concept(content)
                        
                        # 严格匹配：只有当文档的标题概念与查找概念完全一致时才匹配
                        if doc_title_concept and doc_title_concept == concept_name:
                            return doc_path
                            
                    except Exception:
                        continue
            
            return None
    
    def _clean_concept_name(self, name: str) -> str:
        """清理概念名称，用于匹配"""
        if not name:
            return ""
        
        # 转换为小写
        clean_name = name.lower()
        
        # 移除常见标点符号
        import string
        clean_name = clean_name.translate(str.maketrans('', '', string.punctuation))
        
        # 移除空格
        clean_name = ''.join(clean_name.split())
        
        return clean_name
    
    def _update_inbound_links(self):
        """更新所有文档的反向链接信息"""
        # 使用单独的连接来避免锁定问题
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        
        try:
            cursor = conn.cursor()
            
            # 获取所有文档
            cursor.execute('SELECT doc_path FROM documents')
            all_docs = [row[0] for row in cursor.fetchall()]
            
            for doc_path in all_docs:
                # 查找指向该文档的所有链接
                cursor.execute('''
                    SELECT DISTINCT source_doc FROM concept_links 
                    WHERE target_doc = ? AND source_doc != ?
                ''', (doc_path, doc_path))
                
                inbound_docs = [row[0] for row in cursor.fetchall()]
                
                # 更新文档的反向链接
                cursor.execute('''
                    UPDATE documents SET inbound_links = ? WHERE doc_path = ?
                ''', (json.dumps(inbound_docs), doc_path))
            
            conn.commit()
        finally:
            conn.close()
    
    def get_concept_links(self, concept_name: str) -> List[ConceptLink]:
        """获取概念的所有链接信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT concept_name, source_doc, target_doc, line_number, context, created_at
                FROM concept_links WHERE concept_name = ?
            ''', (concept_name,))
            
            links = []
            for row in cursor.fetchall():
                links.append(ConceptLink(
                    concept_name=row[0],
                    source_doc=row[1],
                    target_doc=row[2],
                    line_number=row[3],
                    context=row[4],
                    created_at=row[5]
                ))
            
            return links
    
    def get_document_links(self, doc_path: str) -> Dict[str, List[str]]:
        """获取文档的链接信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT outbound_links, inbound_links FROM documents WHERE doc_path = ?
            ''', (doc_path,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'outbound': json.loads(result[0] or '[]'),
                    'inbound': json.loads(result[1] or '[]')
                }
            
            return {'outbound': [], 'inbound': []}
    
    def find_concept_target(self, concept_name: str) -> Optional[str]:
        """查找概念对应的目标文档路径"""
        return self._find_target_document(concept_name)
    
    def get_all_concepts(self, limit: int = 100, offset: int = 0) -> List[Dict[str, any]]:
        """获取所有概念及其统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT concept_name, 
                       COUNT(*) as reference_count,
                       MAX(CASE WHEN is_primary = 1 THEN doc_path END) as primary_doc
                FROM concept_documents 
                GROUP BY concept_name
                ORDER BY reference_count DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            concepts = []
            for row in cursor.fetchall():
                concepts.append({
                    'name': row[0],
                    'reference_count': row[1],
                    'primary_doc': row[2],
                    'has_target': row[2] is not None
                })
            
            return concepts
    
    def generate_link_report(self) -> Dict[str, any]:
        """生成链接系统的分析报告"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 总体统计
            cursor.execute('SELECT COUNT(*) FROM documents')
            total_docs = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT concept_name) FROM concept_documents')
            total_concepts = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM concept_links')
            total_links = cursor.fetchone()[0]
            
            # 已解析链接数
            cursor.execute('SELECT COUNT(*) FROM concept_links WHERE target_doc IS NOT NULL')
            resolved_links = cursor.fetchone()[0]
            
            # 孤立概念（没有目标文档的概念）
            cursor.execute('''
                SELECT concept_name FROM concept_documents 
                WHERE concept_name NOT IN (
                    SELECT concept_name FROM concept_documents WHERE is_primary = 1
                )
                GROUP BY concept_name
            ''')
            orphaned_concepts = [row[0] for row in cursor.fetchall()]
            
            return {
                'total_documents': total_docs,
                'total_concepts': total_concepts,
                'total_links': total_links,
                'resolved_links': resolved_links,
                'resolution_rate': resolved_links / total_links if total_links > 0 else 0,
                'orphaned_concepts': orphaned_concepts,
                'orphaned_count': len(orphaned_concepts)
            }
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, any]]:
        """获取所有文档及其元数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT doc_path, title, concepts, outbound_links, 
                       inbound_links, last_updated
                FROM documents 
                ORDER BY last_updated DESC
            ''')
            
            documents = []
            current_knowledge_base = str(self.knowledge_base_path)
            processed_count = 0
            
            for row in cursor.fetchall():
                doc_path, title, concepts_json, outbound_json, inbound_json, last_updated = row
                
                # 解析实际的文档路径
                resolved_path = self._resolve_document_path(doc_path)
                
                # 只包含当前知识库路径下的文档，且文件确实存在
                if not resolved_path.startswith(current_knowledge_base) or not os.path.exists(resolved_path):
                    continue
                
                # 只处理md文件
                if not resolved_path.lower().endswith('.md'):
                    continue
                
                # 跳过offset数量的文档
                if processed_count < offset:
                    processed_count += 1
                    continue
                
                # 达到limit数量时停止
                if len(documents) >= limit:
                    break
                
                # 解析JSON字段
                try:
                    concepts = json.loads(concepts_json) if concepts_json else []
                    outbound_links = json.loads(outbound_json) if outbound_json else []
                    inbound_links = json.loads(inbound_json) if inbound_json else []
                except json.JSONDecodeError:
                    concepts = []
                    outbound_links = []
                    inbound_links = []
                
                documents.append({
                    'id': hashlib.md5(doc_path.encode()).hexdigest()[:12],
                    'title': title or os.path.basename(resolved_path),
                    'doc_path': resolved_path,  # 使用解析后的路径
                    'concepts': concepts,
                    'concept_count': len(concepts),
                    'outbound_links': outbound_links,
                    'inbound_links': inbound_links,
                    'created_at': last_updated or datetime.now().isoformat(),
                    'word_count': self._estimate_word_count(resolved_path)
                })
                processed_count += 1
            
            return documents
    
    def _resolve_document_path(self, doc_path: str) -> str:
        """解析文档路径，处理旧路径到新路径的转换"""
        # 如果文件已经存在，直接返回
        if os.path.exists(doc_path):
            return doc_path
        
        # 处理路径转换
        # 将 /Users/yanglulu10/knowledge-agent/知识库/ 转换为 /Users/pluto/Desktop/知识库/知识库/
        old_base = "/Users/yanglulu10/knowledge-agent/知识库/"
        new_base = str(self.knowledge_base_path) + "/"
        
        if doc_path.startswith(old_base):
            relative_path = doc_path[len(old_base):]
            new_path = os.path.join(new_base, relative_path)
            
            # 检查新路径是否存在
            if os.path.exists(new_path):
                return new_path
        
        # 如果都不存在，尝试在知识库中查找相同的文件名
        filename = os.path.basename(doc_path)
        knowledge_base_path = str(self.knowledge_base_path)
        
        for root, dirs, files in os.walk(knowledge_base_path):
            if filename in files:
                potential_path = os.path.join(root, filename)
                if os.path.exists(potential_path):
                    return potential_path
        
        # 如果找不到，返回原路径（会导致过滤掉）
        return doc_path
    
    def get_document_info(self, doc_id: str) -> Optional[Dict[str, any]]:
        """获取特定文档的详细信息"""
        documents = self.get_all_documents(limit=1000)
        for doc in documents:
            if doc['id'] == doc_id:
                return doc
        return None
    
    def get_concept_info(self, concept_name: str) -> Optional[Dict[str, any]]:
        """获取特定概念的详细信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 获取概念的基本信息
            cursor.execute('''
                SELECT doc_path, is_primary 
                FROM concept_documents 
                WHERE concept_name = ?
            ''', (concept_name,))
            
            concept_docs = cursor.fetchall()
            if not concept_docs:
                return None
            
            # 获取相关链接
            cursor.execute('''
                SELECT source_doc, target_doc, line_number, context 
                FROM concept_links 
                WHERE concept_name = ?
            ''', (concept_name,))
            
            links = cursor.fetchall()
            
            return {
                'term': concept_name,
                'documents': [{'doc_path': doc[0], 'is_primary': bool(doc[1])} for doc in concept_docs],
                'links': [{'source_doc': link[0], 'target_doc': link[1], 'line_number': link[2], 'context': link[3]} for link in links],
                'reference_count': len(concept_docs)
            }
    
    def get_stats(self) -> Dict[str, any]:
        """获取链接系统统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 文档数量
            cursor.execute('SELECT COUNT(*) FROM documents')
            doc_count = cursor.fetchone()[0]
            
            # 概念数量
            cursor.execute('SELECT COUNT(DISTINCT concept_name) FROM concept_documents')
            concept_count = cursor.fetchone()[0]
            
            # 链接数量
            cursor.execute('SELECT COUNT(*) FROM concept_links')
            link_count = cursor.fetchone()[0]
            
            return {
                'total_documents': doc_count,
                'total_concepts': concept_count,
                'total_links': link_count,
                'last_updated': datetime.now().isoformat()
            }
    
    def discover_links_for_document(self, doc_id: str, threshold: float = 0.7) -> List[Dict[str, any]]:
        """为特定文档发现链接"""
        doc_info = self.get_document_info(doc_id)
        if not doc_info:
            return []
        
        links = []
        for concept in doc_info.get('concepts', []):
            concept_info = self.get_concept_info(concept)
            if concept_info:
                links.append({
                    'source': concept,
                    'target': concept_info.get('term'),
                    'type': 'concept_link',
                    'strength': threshold,
                    'confidence': 0.9
                })
        
        return links
    
    def _estimate_word_count(self, doc_path: str) -> int:
        """估算文档字数"""
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
                chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
                english_words = len(re.findall(r'\b[a-zA-Z]+\b', content))
                return chinese_chars + english_words
        except:
            return 0
    
    def remove_document(self, doc_path: str) -> bool:
        """删除文档及其相关链接和概念
        
        Args:
            doc_path: 要删除的文档路径
            
        Returns:
            bool: 删除是否成功
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 删除相关的概念链接
                cursor.execute('DELETE FROM concept_links WHERE source_doc = ?', (doc_path,))
                deleted_links = cursor.rowcount
                
                # 删除概念-文档映射
                cursor.execute('DELETE FROM concept_documents WHERE doc_path = ?', (doc_path,))
                deleted_concepts = cursor.rowcount
                
                # 删除文档记录
                cursor.execute('DELETE FROM documents WHERE doc_path = ?', (doc_path,))
                deleted_docs = cursor.rowcount
                
                conn.commit()
                
                # 更新反向链接
                self._update_inbound_links()
                
                logger.info(f"删除文档成功: {doc_path} (链接:{deleted_links}, 概念:{deleted_concepts}, 文档:{deleted_docs})")
                return True
                
        except Exception as e:
            logger.error(f"删除文档失败 {doc_path}: {e}")
            return False
    
    def update_document_incremental(self, doc_path: Path) -> bool:
        """增量更新单个文档
        
        Args:
            doc_path: 文档路径
            
        Returns:
            bool: 更新是否成功
        """
        try:
            # 检查文件是否存在
            if not doc_path.exists():
                logger.warning(f"文件不存在: {doc_path}")
                return False
            
            # 读取文件内容
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 计算文件哈希
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # 检查是否需要更新
            if self._is_file_up_to_date(str(doc_path), file_hash):
                logger.debug(f"文件无变化，跳过更新: {doc_path}")
                return True
            
            # 提取文档信息
            title = self._extract_title(content)
            concept_links = self._extract_concept_links(content, str(doc_path))
            defined_concepts = self._extract_defined_concepts(content)
            
            # 更新数据库
            self._update_document_in_db(
                doc_path=str(doc_path),
                title=title,
                defined_concepts=defined_concepts,
                concept_links=concept_links,
                file_hash=file_hash
            )
            
            # 重新解析链接关系
            self._resolve_all_links()
            
            logger.info(f"更新文档成功: {doc_path} (标题: {title})")
            return True
            
        except Exception as e:
            logger.error(f"更新文档失败 {doc_path}: {e}")
            return False


def main():
    """命令行工具入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='知识库链接管理器')
    parser.add_argument('knowledge_base', help='知识库路径')
    parser.add_argument('--scan', action='store_true', help='扫描知识库并更新链接')
    parser.add_argument('--report', action='store_true', help='生成链接报告')
    parser.add_argument('--concept', help='查询特定概念的链接信息')
    
    args = parser.parse_args()
    
    manager = LinkManager(args.knowledge_base)
    
    if args.scan:
        print("扫描知识库中...")
        stats = manager.scan_knowledge_base()
        print(f"扫描完成: {stats}")
    
    if args.report:
        report = manager.generate_link_report()
        print("=== 链接系统报告 ===")
        print(f"文档总数: {report['total_documents']}")
        print(f"概念总数: {report['total_concepts']}")
        print(f"链接总数: {report['total_links']}")
        print(f"已解析链接: {report['resolved_links']}")
        print(f"解析率: {report['resolution_rate']:.1%}")
        print(f"孤立概念数: {report['orphaned_count']}")
        
        if report['orphaned_concepts']:
            print("\n孤立概念（前10个）:")
            for concept in report['orphaned_concepts'][:10]:
                print(f"  - {concept}")
    
    if args.concept:
        links = manager.get_concept_links(args.concept)
        print(f"=== 概念 '{args.concept}' 的链接信息 ===")
        for link in links:
            print(f"来源: {link.source_doc}:{link.line_number}")
            print(f"目标: {link.target_doc or '未找到'}")
            print(f"上下文: {link.context}")
            print()



if __name__ == '__main__':
    main()