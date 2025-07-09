"""
查询服务 - 提供统一的查询接口
"""

import logging
from typing import Dict, List, Optional
from .data_models import ConceptLink, DocumentMeta

logger = logging.getLogger(__name__)


class QueryService:
    """查询服务，提供统一的查询接口"""
    
    def __init__(self, database_manager, path_utils):
        self.db = database_manager
        self.path_utils = path_utils
    
    def get_concept_links(self, concept_name: str) -> List[ConceptLink]:
        """获取概念的所有链接信息"""
        links_data = self.db.get_concept_links(concept_name)
        links = []
        
        for link_data in links_data:
            link = ConceptLink(
                concept_name=link_data['concept_name'],
                source_doc=link_data['source_doc'],
                target_doc=link_data['target_doc'],
                line_number=link_data['line_number'],
                context=link_data['context'],
                created_at=link_data['created_at']
            )
            links.append(link)
        
        return links
    
    def get_document_links(self, doc_path: str) -> Dict[str, List[str]]:
        """获取文档的链接信息"""
        return self.db.get_document_links(doc_path)
    
    def get_all_concepts(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """获取所有概念及其统计信息"""
        return self.db.get_all_concepts(limit, offset)
    
    def get_all_documents(self, limit: int = 100, offset: int = 0, 
                         knowledge_base_path: str = None) -> List[Dict]:
        """获取所有文档及其元数据"""
        return self.db.get_all_documents(
            limit=limit,
            offset=offset,
            knowledge_base_path=knowledge_base_path,
            resolve_path_func=lambda p: self.path_utils.resolve_document_path(p, knowledge_base_path),
            estimate_word_count_func=self._estimate_word_count
        )
    
    def get_document_info(self, doc_id: str) -> Optional[Dict]:
        """获取特定文档的详细信息"""
        return self.db.get_document_info(doc_id, self.get_all_documents)
    
    def get_concept_info(self, concept_name: str) -> Optional[Dict]:
        """获取特定概念的详细信息"""
        return self.db.get_concept_info(concept_name)
    
    def get_stats(self) -> Dict:
        """获取链接系统统计信息"""
        return self.db.get_stats()
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索概念"""
        all_concepts = self.db.get_all_concepts(limit=1000)
        results = []
        
        query_lower = query.lower()
        
        for concept in all_concepts:
            name = concept['name']
            name_lower = name.lower()
            
            # 计算匹配度
            if query_lower == name_lower:
                score = 1.0
            elif query_lower in name_lower:
                score = 0.8
            elif any(word in name_lower for word in query_lower.split()):
                score = 0.6
            else:
                continue
            
            results.append({
                'concept': name,
                'score': score,
                'reference_count': concept['reference_count'],
                'has_target': concept['has_target']
            })
        
        # 按相关度排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索文档"""
        all_documents = self.get_all_documents(limit=1000)
        results = []
        
        query_lower = query.lower()
        
        for doc in all_documents:
            title = doc['title']
            title_lower = title.lower()
            
            # 计算匹配度
            if query_lower == title_lower:
                score = 1.0
            elif query_lower in title_lower:
                score = 0.8
            elif any(word in title_lower for word in query_lower.split()):
                score = 0.6
            else:
                continue
            
            results.append({
                'document': doc,
                'score': score
            })
        
        # 按相关度排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return [r['document'] for r in results[:limit]]
    
    def discover_links_for_document(self, doc_path: str, threshold: float = 0.7) -> List[Dict]:
        """为文档发现潜在的链接"""
        links = []
        
        # 获取文档的出站链接
        doc_links = self.get_document_links(doc_path)
        outbound_links = doc_links.get('outbound', [])
        
        # 为每个出站链接查找目标文档
        for concept_name in outbound_links:
            concept_info = self.get_concept_info(concept_name)
            if concept_info:
                target_docs = [doc for doc in concept_info['documents'] if doc['is_primary']]
                if target_docs:
                    links.append({
                        'concept': concept_name,
                        'target_doc': target_docs[0]['doc_path'],
                        'confidence': 0.9
                    })
        
        return links
    
    def get_concept_network(self, concept_name: str, depth: int = 2) -> Dict:
        """获取概念的网络关系"""
        network = {
            'center': concept_name,
            'nodes': [],
            'edges': []
        }
        
        visited = set()
        to_visit = [(concept_name, 0)]
        
        while to_visit:
            current_concept, current_depth = to_visit.pop(0)
            
            if current_concept in visited or current_depth > depth:
                continue
            
            visited.add(current_concept)
            concept_info = self.get_concept_info(current_concept)
            
            if concept_info:
                network['nodes'].append({
                    'id': current_concept,
                    'label': current_concept,
                    'reference_count': concept_info['reference_count'],
                    'depth': current_depth
                })
                
                # 查找相关概念
                if current_depth < depth:
                    for doc_info in concept_info['documents']:
                        doc_path = doc_info['doc_path']
                        doc_links = self.get_document_links(doc_path)
                        
                        for related_concept in doc_links.get('outbound', []):
                            if related_concept not in visited:
                                to_visit.append((related_concept, current_depth + 1))
                                
                                # 添加边
                                network['edges'].append({
                                    'source': current_concept,
                                    'target': related_concept,
                                    'weight': 1.0
                                })
        
        return network
    
    def _estimate_word_count(self, file_path: str) -> int:
        """估算文档字数"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的字数统计
            import re
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
            english_words = len(re.findall(r'[a-zA-Z]+', content))
            
            return chinese_chars + english_words
        except Exception as e:
            logger.error(f"估算字数失败 {file_path}: {e}")
            return 0
    
    def get_popular_concepts(self, limit: int = 20) -> List[Dict]:
        """获取热门概念"""
        concepts = self.get_all_concepts(limit=limit)
        return sorted(concepts, key=lambda x: x['reference_count'], reverse=True)
    
    def get_recent_documents(self, limit: int = 20) -> List[Dict]:
        """获取最近的文档"""
        return self.get_all_documents(limit=limit)
    
    def get_orphaned_concepts(self, limit: int = 50) -> List[str]:
        """获取孤立概念"""
        report = self.db.generate_link_report()
        return report.get('orphaned_concepts', [])[:limit]