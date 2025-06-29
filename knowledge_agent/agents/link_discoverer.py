"""
链接发现工作者 - 负责发现概念间的关系并建立链接
"""
import re
from typing import Dict, List, Any, Tuple, Set
from agents.base_agent import BaseAgent
from utils.vector_db import LocalVectorDB
import logging

logger = logging.getLogger(__name__)

class LinkDiscoverer(BaseAgent):
    """链接发现工作者Agent"""
    
    def __init__(self, vector_db: LocalVectorDB):
        super().__init__(
            name="链接发现专家",
            description="发现概念间的关系并建立双向链接"
        )
        self.vector_db = vector_db
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理概念，发现链接关系
        
        Args:
            input_data: {
                "concepts": List[dict],     # 当前文档的概念
                "structured_content": str, # 结构化内容
                "metadata": dict,          # 元数据
                "doc_id": str             # 文档ID
            }
        
        Returns:
            {
                "concept_links": List[dict],    # 概念间链接
                "existing_links": List[dict],   # 与现有知识库的链接
                "link_suggestions": List[dict], # 建议的链接
                "updated_content": str,         # 更新后的内容
                "relationship_map": dict        # 关系映射
            }
        """
        
        concepts = input_data.get("concepts", [])
        content = input_data.get("structured_content", "")
        metadata = input_data.get("metadata", {})
        doc_id = input_data.get("doc_id", "")
        
        # 1. 发现内部概念链接
        internal_links = self._discover_internal_links(concepts, content)
        
        # 2. 发现与现有知识库的链接
        external_links = self._discover_external_links(concepts, doc_id)
        
        # 3. 生成链接建议
        link_suggestions = self._generate_link_suggestions(concepts, internal_links, external_links)
        
        # 4. 构建关系映射
        relationship_map = self._build_relationship_map(concepts, internal_links, external_links)
        
        # 5. 更新内容中的链接
        updated_content = self._update_content_links(content, internal_links, external_links)
        
        return {
            "concept_links": internal_links,
            "existing_links": external_links,
            "link_suggestions": link_suggestions,
            "updated_content": updated_content,
            "relationship_map": relationship_map
        }
    
    def _discover_internal_links(self, concepts: List[Dict[str, Any]], content: str) -> List[Dict[str, Any]]:
        """发现内部概念链接"""
        links = []
        
        # 1. 基于语义相似度的链接
        semantic_links = self._find_semantic_links(concepts)
        links.extend(semantic_links)
        
        # 2. 基于内容共现的链接
        cooccurrence_links = self._find_cooccurrence_links(concepts, content)
        links.extend(cooccurrence_links)
        
        # 3. 基于层级关系的链接
        hierarchical_links = self._find_hierarchical_links(concepts, content)
        links.extend(hierarchical_links)
        
        # 4. 基于因果关系的链接
        causal_links = self._find_causal_links(concepts, content)
        links.extend(causal_links)
        
        # 去重和评分
        links = self._deduplicate_and_score_links(links)
        
        return links
    
    def _find_semantic_links(self, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于语义相似度发现链接"""
        links = []
        
        if len(concepts) < 2:
            return links
        
        concept_terms = [c['term'] for c in concepts]
        
        try:
            # 使用向量数据库计算概念间相似度
            similarity_links = self.vector_db.find_concept_links(concept_terms, threshold=0.6)
            
            for concept1, concept2, similarity in similarity_links:
                links.append({
                    'source': concept1,
                    'target': concept2,
                    'type': 'semantic_similarity',
                    'strength': similarity,
                    'bidirectional': True,
                    'confidence': 0.7
                })
                
        except Exception as e:
            logger.warning(f"语义链接发现失败: {e}")
        
        return links
    
    def _find_cooccurrence_links(self, concepts: List[Dict[str, Any]], content: str) -> List[Dict[str, Any]]:
        """基于共现发现链接"""
        links = []
        
        # 分析概念在文本中的共现模式
        concept_positions = {}
        
        for concept in concepts:
            term = concept['term']
            positions = []
            
            # 找到概念在文本中的所有位置
            for match in re.finditer(re.escape(term), content, re.IGNORECASE):
                positions.append(match.start())
            
            concept_positions[term] = positions
        
        # 计算概念间的距离
        window_size = 200  # 共现窗口大小
        
        for term1, positions1 in concept_positions.items():
            for term2, positions2 in concept_positions.items():
                if term1 >= term2:  # 避免重复
                    continue
                
                # 计算最小距离
                min_distance = float('inf')
                cooccurrences = 0
                
                for pos1 in positions1:
                    for pos2 in positions2:
                        distance = abs(pos1 - pos2)
                        if distance <= window_size:
                            cooccurrences += 1
                            min_distance = min(min_distance, distance)
                
                if cooccurrences > 0:
                    # 距离越近，强度越高
                    strength = max(0.1, 1.0 - (min_distance / window_size))
                    confidence = min(0.9, 0.4 + cooccurrences * 0.1)
                    
                    links.append({
                        'source': term1,
                        'target': term2,
                        'type': 'cooccurrence',
                        'strength': strength,
                        'bidirectional': True,
                        'confidence': confidence,
                        'cooccurrence_count': cooccurrences,
                        'min_distance': min_distance
                    })
        
        return links
    
    def _find_hierarchical_links(self, concepts: List[Dict[str, Any]], content: str) -> List[Dict[str, Any]]:
        """发现层级关系链接"""
        links = []
        
        # 定义层级关系模式
        hierarchical_patterns = [
            (r'([^。\n]+)包含([^。\n]+)', 'contains'),
            (r'([^。\n]+)是([^。\n]+)的一种', 'is_type_of'),
            (r'([^。\n]+)属于([^。\n]+)', 'belongs_to'),
            (r'([^。\n]+)分为([^。\n]+)', 'divided_into'),
            (r'([^。\n]+)由([^。\n]+)组成', 'composed_of'),
        ]
        
        concept_terms = [c['term'] for c in concepts]
        
        for pattern, relation_type in hierarchical_patterns:
            matches = re.finditer(pattern, content)
            
            for match in matches:
                text1 = match.group(1).strip()
                text2 = match.group(2).strip()
                
                # 检查是否包含我们的概念
                source_concept = None
                target_concept = None
                
                for term in concept_terms:
                    if term in text1:
                        source_concept = term
                    if term in text2:
                        target_concept = term
                
                if source_concept and target_concept and source_concept != target_concept:
                    links.append({
                        'source': source_concept,
                        'target': target_concept,
                        'type': relation_type,
                        'strength': 0.8,
                        'bidirectional': False,
                        'confidence': 0.7,
                        'context': match.group(0)
                    })
        
        return links
    
    def _find_causal_links(self, concepts: List[Dict[str, Any]], content: str) -> List[Dict[str, Any]]:
        """发现因果关系链接"""
        links = []
        
        # 定义因果关系模式
        causal_patterns = [
            (r'([^。\n]+)导致([^。\n]+)', 'causes'),
            (r'([^。\n]+)引起([^。\n]+)', 'triggers'),
            (r'由于([^。\n]+)[，,]([^。\n]+)', 'caused_by'),
            (r'([^。\n]+)的结果是([^。\n]+)', 'results_in'),
            (r'([^。\n]+)影响([^。\n]+)', 'influences'),
        ]
        
        concept_terms = [c['term'] for c in concepts]
        
        for pattern, relation_type in causal_patterns:
            matches = re.finditer(pattern, content)
            
            for match in matches:
                text1 = match.group(1).strip()
                text2 = match.group(2).strip()
                
                # 检查是否包含我们的概念
                source_concept = None
                target_concept = None
                
                for term in concept_terms:
                    if term in text1:
                        source_concept = term
                    if term in text2:
                        target_concept = term
                
                if source_concept and target_concept and source_concept != target_concept:
                    links.append({
                        'source': source_concept,
                        'target': target_concept,
                        'type': relation_type,
                        'strength': 0.7,
                        'bidirectional': False,
                        'confidence': 0.6,
                        'context': match.group(0)
                    })
        
        return links
    
    def _discover_external_links(self, concepts: List[Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
        """发现与现有知识库的链接"""
        external_links = []
        
        for concept in concepts:
            term = concept['term']
            
            try:
                # 在向量数据库中搜索相关概念
                related_concepts = self.vector_db.search_related_concepts(
                    term, n_results=5, threshold=0.6
                )
                
                for related in related_concepts:
                    # 排除同一文档的概念
                    if related['doc_id'] != doc_id:
                        external_links.append({
                            'source': term,
                            'target': related['term'],
                            'type': 'related_concept',
                            'strength': related['similarity'],
                            'bidirectional': True,
                            'confidence': related['confidence'],
                            'target_doc_id': related['doc_id'],
                            'target_definition': related['definition']
                        })
                
            except Exception as e:
                logger.warning(f"搜索外部链接失败 {term}: {e}")
        
        return external_links
    
    def _generate_link_suggestions(self, concepts: List[Dict[str, Any]], 
                                 internal_links: List[Dict[str, Any]], 
                                 external_links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成链接建议"""
        suggestions = []
        
        # 1. 建议缺失的内部链接
        concept_terms = set(c['term'] for c in concepts)
        linked_pairs = set()
        
        for link in internal_links:
            linked_pairs.add((link['source'], link['target']))
            linked_pairs.add((link['target'], link['source']))
        
        # 检查是否有概念没有任何链接
        for concept in concepts:
            term = concept['term']
            has_links = any(term in pair for pair in linked_pairs)
            
            if not has_links and len(concepts) > 1:
                # 建议与最相关的概念建立链接
                best_candidate = None
                best_score = 0
                
                for other_concept in concepts:
                    if other_concept['term'] != term:
                        # 简单的相似度评估
                        score = self._calculate_simple_similarity(
                            concept, other_concept
                        )
                        if score > best_score:
                            best_score = score
                            best_candidate = other_concept['term']
                
                if best_candidate and best_score > 0.3:
                    suggestions.append({
                        'type': 'missing_internal_link',
                        'source': term,
                        'target': best_candidate,
                        'reason': '孤立概念建议链接',
                        'confidence': best_score
                    })
        
        # 2. 建议高质量的外部链接
        high_quality_external = [
            link for link in external_links 
            if link['strength'] > 0.8 and link['confidence'] > 0.7
        ]
        
        for link in high_quality_external[:3]:  # 最多建议3个
            suggestions.append({
                'type': 'recommended_external_link',
                'source': link['source'],
                'target': link['target'],
                'target_doc_id': link['target_doc_id'],
                'reason': f"高相似度链接 ({link['strength']:.2f})",
                'confidence': link['confidence']
            })
        
        return suggestions
    
    def _calculate_simple_similarity(self, concept1: Dict[str, Any], 
                                   concept2: Dict[str, Any]) -> float:
        """计算简单相似度"""
        # 基于类型和定义的简单相似度
        score = 0.0
        
        # 类型相似度
        if concept1.get('type') == concept2.get('type'):
            score += 0.3
        
        # 定义长度相似度
        def1 = concept1.get('definition', '')
        def2 = concept2.get('definition', '')
        
        if def1 and def2:
            len_diff = abs(len(def1) - len(def2))
            max_len = max(len(def1), len(def2))
            if max_len > 0:
                len_similarity = 1.0 - (len_diff / max_len)
                score += len_similarity * 0.2
        
        # 词汇重叠度
        words1 = set(re.findall(r'\\w+', concept1['term'].lower()))
        words2 = set(re.findall(r'\\w+', concept2['term'].lower()))
        
        if words1 and words2:
            overlap = len(words1 & words2) / len(words1 | words2)
            score += overlap * 0.5
        
        return min(1.0, score)
    
    def _deduplicate_and_score_links(self, links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重和评分链接"""
        # 按源和目标去重
        unique_links = {}
        
        for link in links:
            source = link['source']
            target = link['target']
            
            # 标准化键（确保一致性）
            key = tuple(sorted([source, target]))
            
            if key not in unique_links:
                unique_links[key] = link
            else:
                # 保留置信度更高的链接
                existing = unique_links[key]
                if link['confidence'] > existing['confidence']:
                    unique_links[key] = link
        
        # 按置信度排序
        result_links = list(unique_links.values())
        result_links.sort(key=lambda x: x['confidence'], reverse=True)
        
        return result_links
    
    def _build_relationship_map(self, concepts: List[Dict[str, Any]], 
                               internal_links: List[Dict[str, Any]], 
                               external_links: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建关系映射"""
        relationship_map = {
            'nodes': [],
            'edges': [],
            'clusters': [],
            'statistics': {}
        }
        
        # 添加节点
        for concept in concepts:
            relationship_map['nodes'].append({
                'id': concept['term'],
                'label': concept['term'],
                'type': 'internal_concept',
                'definition': concept.get('definition', ''),
                'confidence': concept.get('final_score', 0.5)
            })
        
        # 添加外部节点
        external_nodes = set()
        for link in external_links:
            if link['target'] not in external_nodes:
                external_nodes.add(link['target'])
                relationship_map['nodes'].append({
                    'id': link['target'],
                    'label': link['target'],
                    'type': 'external_concept',
                    'definition': link.get('target_definition', ''),
                    'doc_id': link.get('target_doc_id', '')
                })
        
        # 添加边
        edge_id = 0
        for link in internal_links + external_links:
            relationship_map['edges'].append({
                'id': edge_id,
                'source': link['source'],
                'target': link['target'],
                'type': link['type'],
                'strength': link['strength'],
                'bidirectional': link.get('bidirectional', True),
                'confidence': link['confidence']
            })
            edge_id += 1
        
        # 计算统计信息
        relationship_map['statistics'] = {
            'total_concepts': len(concepts),
            'internal_links': len(internal_links),
            'external_links': len(external_links),
            'average_confidence': sum(link['confidence'] for link in internal_links + external_links) / max(1, len(internal_links + external_links))
        }
        
        return relationship_map
    
    def _update_content_links(self, content: str, internal_links: List[Dict[str, Any]], 
                            external_links: List[Dict[str, Any]]) -> str:
        """更新内容中的链接"""
        updated_content = content
        
        # 收集所有需要链接的概念
        linked_concepts = set()
        
        for link in internal_links + external_links:
            linked_concepts.add(link['source'])
            linked_concepts.add(link['target'])
        
        # 确保所有链接的概念都使用 [[]] 格式
        for concept in linked_concepts:
            # 只有当概念还没有被链接时才添加链接
            pattern = r'\\b' + re.escape(concept) + r'\\b'
            link_pattern = r'\\[\\[' + re.escape(concept) + r'\\]\\]'
            
            # 检查是否已经有链接
            if not re.search(link_pattern, updated_content):
                # 替换第一次出现的概念为链接格式
                updated_content = re.sub(
                    pattern, 
                    f'[[{concept}]]', 
                    updated_content, 
                    count=1
                )
        
        return updated_content