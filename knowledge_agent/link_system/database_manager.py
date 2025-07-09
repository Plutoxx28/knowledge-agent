"""
数据库管理器 - 处理链接系统的数据库操作
"""

import os
import json
import hashlib
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理器，负责所有数据库操作"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
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
    
    def update_document_in_db(self, doc_path: str, title: str, defined_concepts: List[str], 
                              concept_links: List, file_hash: str):
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
    
    def remove_document(self, doc_path: str) -> bool:
        """删除文档及其相关链接和概念"""
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
                self.update_inbound_links()
                
                logger.info(f"删除文档成功: {doc_path} (链接:{deleted_links}, 概念:{deleted_concepts}, 文档:{deleted_docs})")
                return True
                
        except Exception as e:
            logger.error(f"删除文档失败 {doc_path}: {e}")
            return False
    
    def is_file_up_to_date(self, doc_path: str, file_hash: str) -> bool:
        """检查文件是否已是最新版本"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT file_hash FROM documents WHERE doc_path = ?', (doc_path,))
            result = cursor.fetchone()
            return result and result[0] == file_hash
    
    def resolve_all_links(self, find_target_func):
        """解析所有概念链接，找到对应的目标文档"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 获取所有未解析的链接
            cursor.execute('SELECT id, concept_name FROM concept_links WHERE target_doc IS NULL')
            unresolved_links = cursor.fetchall()
            
            for link_id, concept_name in unresolved_links:
                target_doc = find_target_func(concept_name)
                if target_doc:
                    cursor.execute('UPDATE concept_links SET target_doc = ? WHERE id = ?', 
                                 (target_doc, link_id))
            
            # 更新反向链接
            self.update_inbound_links()
            
            conn.commit()
    
    def find_target_document(self, concept_name: str, extract_title_concept_func) -> Optional[str]:
        """为概念找到对应的目标文档"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 方法1：精确匹配
            cursor.execute('''
                SELECT doc_path FROM concept_documents 
                WHERE concept_name = ? AND is_primary = 1
                LIMIT 1
            ''', (concept_name,))
            
            result = cursor.fetchone()
            if result:
                return result[0]
            
            # 方法2：严格的标题匹配
            cursor.execute('SELECT d.doc_path, d.title FROM documents d')
            all_docs = cursor.fetchall()
            
            for doc_path, title in all_docs:
                if os.path.exists(doc_path):
                    try:
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        doc_title_concept = extract_title_concept_func(content)
                        
                        if doc_title_concept and doc_title_concept == concept_name:
                            return doc_path
                            
                    except Exception:
                        continue
            
            return None
    
    def update_inbound_links(self):
        """更新所有文档的反向链接信息"""
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
    
    def get_concept_links(self, concept_name: str) -> List[Dict]:
        """获取概念的所有链接信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT concept_name, source_doc, target_doc, line_number, context, created_at
                FROM concept_links WHERE concept_name = ?
            ''', (concept_name,))
            
            links = []
            for row in cursor.fetchall():
                links.append({
                    'concept_name': row[0],
                    'source_doc': row[1],
                    'target_doc': row[2],
                    'line_number': row[3],
                    'context': row[4],
                    'created_at': row[5]
                })
            
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
    
    def get_all_concepts(self, limit: int = 100, offset: int = 0) -> List[Dict]:
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
    
    def generate_link_report(self) -> Dict:
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
            
            # 孤立概念
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
    
    def get_all_documents(self, limit: int = 100, offset: int = 0, 
                         knowledge_base_path: str = None, 
                         resolve_path_func=None,
                         estimate_word_count_func=None) -> List[Dict]:
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
            processed_count = 0
            
            for row in cursor.fetchall():
                doc_path, title, concepts_json, outbound_json, inbound_json, last_updated = row
                
                # 解析实际的文档路径
                if resolve_path_func:
                    resolved_path = resolve_path_func(doc_path)
                else:
                    resolved_path = doc_path
                
                # 过滤条件
                if knowledge_base_path and not resolved_path.startswith(knowledge_base_path):
                    continue
                if not os.path.exists(resolved_path):
                    continue
                if not resolved_path.lower().endswith('.md'):
                    continue
                
                # 分页处理
                if processed_count < offset:
                    processed_count += 1
                    continue
                
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
                    'doc_path': resolved_path,
                    'concepts': concepts,
                    'concept_count': len(concepts),
                    'outbound_links': outbound_links,
                    'inbound_links': inbound_links,
                    'created_at': last_updated or datetime.now().isoformat(),
                    'word_count': estimate_word_count_func(resolved_path) if estimate_word_count_func else 0
                })
                processed_count += 1
            
            return documents
    
    def get_document_info(self, doc_id: str, get_all_documents_func) -> Optional[Dict]:
        """获取特定文档的详细信息"""
        documents = get_all_documents_func(limit=1000)
        for doc in documents:
            if doc['id'] == doc_id:
                return doc
        return None
    
    def get_concept_info(self, concept_name: str) -> Optional[Dict]:
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
    
    def get_stats(self) -> Dict:
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
    
    def process_document_simple(self, doc_path: Path, extract_title_func) -> bool:
        """简化的文档处理，只更新基本信息"""
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 计算文件哈希
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # 检查是否需要更新
            if self.is_file_up_to_date(str(doc_path), file_hash):
                return False
            
            # 提取文档标题
            title = extract_title_func(content)
            
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
            logger.error(f"处理文档 {doc_path} 时出错: {e}")
            return False