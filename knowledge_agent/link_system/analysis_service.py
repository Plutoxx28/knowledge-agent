"""
分析服务 - 提供链接系统的分析和报告功能
"""

import logging
from typing import Dict, List, Optional
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


class AnalysisService:
    """分析服务，提供链接系统的分析和报告功能"""
    
    def __init__(self, database_manager, query_service):
        self.db = database_manager
        self.query = query_service
    
    def generate_link_report(self) -> Dict:
        """生成链接系统的分析报告"""
        return self.db.generate_link_report()
    
    def analyze_concept_distribution(self) -> Dict:
        """分析概念分布"""
        concepts = self.query.get_all_concepts(limit=1000)
        
        # 按引用次数分组
        distribution = {
            'high_reference': [],    # 引用次数 > 10
            'medium_reference': [],  # 引用次数 5-10
            'low_reference': [],     # 引用次数 1-4
            'no_reference': []       # 引用次数 0
        }
        
        for concept in concepts:
            ref_count = concept['reference_count']
            if ref_count > 10:
                distribution['high_reference'].append(concept)
            elif ref_count >= 5:
                distribution['medium_reference'].append(concept)
            elif ref_count >= 1:
                distribution['low_reference'].append(concept)
            else:
                distribution['no_reference'].append(concept)
        
        return {
            'total_concepts': len(concepts),
            'distribution': distribution,
            'statistics': {
                'high_reference_count': len(distribution['high_reference']),
                'medium_reference_count': len(distribution['medium_reference']),
                'low_reference_count': len(distribution['low_reference']),
                'no_reference_count': len(distribution['no_reference'])
            }
        }
    
    def analyze_document_connectivity(self) -> Dict:
        """分析文档连通性"""
        documents = self.query.get_all_documents(limit=1000)
        
        connectivity_stats = {
            'total_documents': len(documents),
            'connected_documents': 0,
            'isolated_documents': 0,
            'hub_documents': [],  # 出站链接 > 20
            'authority_documents': []  # 入站链接 > 10
        }
        
        for doc in documents:
            outbound_count = len(doc.get('outbound_links', []))
            inbound_count = len(doc.get('inbound_links', []))
            
            if outbound_count > 0 or inbound_count > 0:
                connectivity_stats['connected_documents'] += 1
            else:
                connectivity_stats['isolated_documents'] += 1
            
            # 识别hub文档（出站链接多）
            if outbound_count > 20:
                connectivity_stats['hub_documents'].append({
                    'title': doc['title'],
                    'doc_path': doc['doc_path'],
                    'outbound_count': outbound_count
                })
            
            # 识别authority文档（入站链接多）
            if inbound_count > 10:
                connectivity_stats['authority_documents'].append({
                    'title': doc['title'],
                    'doc_path': doc['doc_path'],
                    'inbound_count': inbound_count
                })
        
        return connectivity_stats
    
    def analyze_concept_clusters(self) -> Dict:
        """分析概念聚类"""
        concepts = self.query.get_all_concepts(limit=500)
        clusters = {}
        
        for concept in concepts:
            concept_name = concept['name']
            
            # 简单的聚类：基于概念名称的前缀
            if ' ' in concept_name:
                prefix = concept_name.split()[0]
            else:
                prefix = concept_name[:3] if len(concept_name) > 3 else concept_name
            
            if prefix not in clusters:
                clusters[prefix] = []
            
            clusters[prefix].append(concept)
        
        # 只保留有多个成员的聚类
        filtered_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
        
        return {
            'total_clusters': len(filtered_clusters),
            'clusters': filtered_clusters,
            'largest_cluster': max(filtered_clusters.items(), key=lambda x: len(x[1])) if filtered_clusters else None
        }
    
    def analyze_link_quality(self) -> Dict:
        """分析链接质量"""
        report = self.generate_link_report()
        
        quality_metrics = {
            'resolution_rate': report.get('resolution_rate', 0),
            'broken_links': report.get('total_links', 0) - report.get('resolved_links', 0),
            'orphaned_concepts': len(report.get('orphaned_concepts', [])),
            'quality_score': 0
        }
        
        # 计算质量分数 (0-100)
        resolution_rate = quality_metrics['resolution_rate']
        orphaned_ratio = quality_metrics['orphaned_concepts'] / max(report.get('total_concepts', 1), 1)
        
        quality_score = (resolution_rate * 70) + ((1 - orphaned_ratio) * 30)
        quality_metrics['quality_score'] = min(100, max(0, quality_score))
        
        return quality_metrics
    
    def generate_usage_trends(self) -> Dict:
        """生成使用趋势分析"""
        # 这里可以添加基于时间的分析
        # 目前返回基础统计
        concepts = self.query.get_all_concepts(limit=1000)
        documents = self.query.get_all_documents(limit=1000)
        
        return {
            'total_concepts': len(concepts),
            'total_documents': len(documents),
            'most_referenced_concepts': sorted(concepts, key=lambda x: x['reference_count'], reverse=True)[:10],
            'recent_documents': documents[:10],  # 假设已按时间排序
            'generated_at': datetime.now().isoformat()
        }
    
    def identify_knowledge_gaps(self) -> Dict:
        """识别知识缺口"""
        orphaned_concepts = self.query.get_orphaned_concepts(limit=100)
        
        # 分析概念类型
        concept_types = {
            'technical_terms': [],
            'general_concepts': [],
            'proper_nouns': [],
            'unknown': []
        }
        
        for concept in orphaned_concepts:
            # 简单的概念类型分类
            if concept.isupper():
                concept_types['technical_terms'].append(concept)
            elif concept.istitle():
                concept_types['proper_nouns'].append(concept)
            elif len(concept.split()) > 1:
                concept_types['general_concepts'].append(concept)
            else:
                concept_types['unknown'].append(concept)
        
        return {
            'total_gaps': len(orphaned_concepts),
            'gap_types': concept_types,
            'recommendations': self._generate_gap_recommendations(concept_types)
        }
    
    def _generate_gap_recommendations(self, concept_types: Dict) -> List[str]:
        """生成知识缺口建议"""
        recommendations = []
        
        if concept_types['technical_terms']:
            recommendations.append(f"创建技术术语词典，包含 {len(concept_types['technical_terms'])} 个技术术语")
        
        if concept_types['proper_nouns']:
            recommendations.append(f"补充专有名词解释，涉及 {len(concept_types['proper_nouns'])} 个专有名词")
        
        if concept_types['general_concepts']:
            recommendations.append(f"扩展概念定义，完善 {len(concept_types['general_concepts'])} 个一般概念")
        
        return recommendations
    
    def generate_comprehensive_report(self) -> Dict:
        """生成综合分析报告"""
        return {
            'overview': self.generate_link_report(),
            'concept_distribution': self.analyze_concept_distribution(),
            'document_connectivity': self.analyze_document_connectivity(),
            'concept_clusters': self.analyze_concept_clusters(),
            'link_quality': self.analyze_link_quality(),
            'usage_trends': self.generate_usage_trends(),
            'knowledge_gaps': self.identify_knowledge_gaps(),
            'generated_at': datetime.now().isoformat()
        }
    
    def get_concept_graph_data(self, max_concepts: int = 100) -> Dict:
        """获取概念图谱数据"""
        concepts = self.query.get_all_concepts(limit=max_concepts)
        
        nodes = []
        edges = []
        
        for concept in concepts:
            # 创建节点
            node = {
                'id': concept['name'],
                'label': concept['name'],
                'size': min(50, 10 + concept['reference_count'] * 2),
                'color': '#0066cc' if concept['has_target'] else '#ff6666'
            }
            nodes.append(node)
            
            # 创建边（基于文档共现）
            concept_info = self.query.get_concept_info(concept['name'])
            if concept_info:
                for doc_info in concept_info['documents']:
                    doc_path = doc_info['doc_path']
                    doc_links = self.query.get_document_links(doc_path)
                    
                    for related_concept in doc_links.get('outbound', []):
                        if related_concept != concept['name'] and related_concept in [c['name'] for c in concepts]:
                            edge = {
                                'source': concept['name'],
                                'target': related_concept,
                                'weight': 1.0
                            }
                            edges.append(edge)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'total_concepts': len(nodes),
            'total_connections': len(edges)
        }