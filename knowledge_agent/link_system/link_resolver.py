"""
链接解析器 - 处理概念链接的解析和目标查找
"""

import logging
from typing import Optional, List, Dict
from .data_models import ConceptLink

logger = logging.getLogger(__name__)


class LinkResolver:
    """链接解析器，负责概念链接的解析和目标查找"""
    
    def __init__(self, database_manager, content_parser):
        self.db = database_manager
        self.parser = content_parser
    
    def resolve_concept_link(self, concept_name: str) -> Optional[str]:
        """解析概念链接，找到对应的目标文档"""
        return self.db.find_target_document(concept_name, self.parser.extract_title_concept)
    
    def resolve_all_links(self):
        """解析所有概念链接"""
        self.db.resolve_all_links(self.resolve_concept_link)
    
    def find_concept_target(self, concept_name: str) -> Optional[str]:
        """查找概念的目标文档"""
        return self.resolve_concept_link(concept_name)
    
    def update_link_targets(self, concept_links: List[ConceptLink]) -> List[ConceptLink]:
        """更新概念链接的目标文档"""
        updated_links = []
        
        for link in concept_links:
            if not link.target_doc:
                target_doc = self.resolve_concept_link(link.concept_name)
                if target_doc:
                    link.target_doc = target_doc
            updated_links.append(link)
        
        return updated_links
    
    def discover_related_concepts(self, concept_name: str) -> List[str]:
        """发现与给定概念相关的其他概念"""
        related_concepts = []
        
        # 获取包含该概念的文档
        concept_links = self.db.get_concept_links(concept_name)
        
        # 从这些文档中提取其他概念
        for link in concept_links:
            doc_links = self.db.get_document_links(link['source_doc'])
            related_concepts.extend(doc_links.get('outbound', []))
        
        # 去重并移除自身
        related_concepts = list(set(related_concepts))
        if concept_name in related_concepts:
            related_concepts.remove(concept_name)
        
        return related_concepts
    
    def calculate_concept_similarity(self, concept1: str, concept2: str) -> float:
        """计算两个概念的相似度"""
        # 简单的字符串相似度计算
        if concept1 == concept2:
            return 1.0
        
        # 检查是否为包含关系
        if concept1 in concept2 or concept2 in concept1:
            return 0.8
        
        # 检查共同词汇
        words1 = set(concept1.split())
        words2 = set(concept2.split())
        
        if words1 & words2:
            return 0.6
        
        return 0.0
    
    def find_similar_concepts(self, concept_name: str, threshold: float = 0.6) -> List[Dict]:
        """查找相似的概念"""
        similar_concepts = []
        all_concepts = self.db.get_all_concepts(limit=1000)
        
        for concept in all_concepts:
            similarity = self.calculate_concept_similarity(concept_name, concept['name'])
            if similarity >= threshold and concept['name'] != concept_name:
                similar_concepts.append({
                    'concept': concept['name'],
                    'similarity': similarity,
                    'reference_count': concept['reference_count']
                })
        
        # 按相似度排序
        similar_concepts.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_concepts
    
    def validate_link_integrity(self) -> Dict:
        """验证链接完整性"""
        report = {
            'total_links': 0,
            'broken_links': 0,
            'resolved_links': 0,
            'orphaned_concepts': [],
            'duplicate_concepts': []
        }
        
        # 获取所有概念
        all_concepts = self.db.get_all_concepts(limit=10000)
        
        for concept in all_concepts:
            concept_name = concept['name']
            links = self.db.get_concept_links(concept_name)
            
            report['total_links'] += len(links)
            
            # 检查链接是否有目标
            for link in links:
                if link.get('target_doc'):
                    report['resolved_links'] += 1
                else:
                    report['broken_links'] += 1
            
            # 检查是否为孤立概念
            if not concept['has_target']:
                report['orphaned_concepts'].append(concept_name)
        
        return report