"""
本地向量数据库模块 - 使用ChromaDB实现语义搜索和相似度匹配
"""
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class LocalVectorDB:
    """本地向量数据库类"""
    
    def __init__(self, db_path: str = "./data/chroma_db", 
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化向量数据库
        
        Args:
            db_path: 数据库存储路径
            model_name: 嵌入模型名称
        """
        self.db_path = db_path
        self.model_name = model_name
        
        # 确保目录存在
        os.makedirs(db_path, exist_ok=True)
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 初始化嵌入模型
        try:
            self.encoder = SentenceTransformer(model_name)
            logger.info(f"成功加载嵌入模型: {model_name}")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            # 使用备用模型
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 获取或创建集合
        self.knowledge_collection = self._get_or_create_collection("knowledge_base")
        self.concept_collection = self._get_or_create_collection("concepts")
        
    def _get_or_create_collection(self, name: str):
        """获取或创建集合"""
        try:
            return self.client.get_collection(name)
        except:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_document(self, content: str, metadata: Dict[str, Any], 
                    doc_id: Optional[str] = None) -> str:
        """
        添加文档到向量数据库
        
        Args:
            content: 文档内容
            metadata: 文档元数据
            doc_id: 文档ID，如果不提供会自动生成
            
        Returns:
            文档ID
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        try:
            # 生成嵌入向量
            embedding = self.encoder.encode(content).tolist()
            
            # 清理metadata，只保留简单数据类型
            clean_metadata = self._clean_metadata(metadata)
            
            # 添加到知识库集合
            self.knowledge_collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[clean_metadata],
                ids=[doc_id]
            )
            
            logger.info(f"成功添加文档: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """清理metadata，只保留ChromaDB支持的数据类型"""
        clean_meta = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean_meta[key] = value
            elif isinstance(value, list):
                # 将列表转换为字符串
                clean_meta[key] = ', '.join(str(item) for item in value)
            elif isinstance(value, dict):
                # 将字典转换为JSON字符串
                clean_meta[key] = str(value)
            else:
                # 其他类型转换为字符串
                clean_meta[key] = str(value)
        
        return clean_meta
    
    def add_concepts(self, concepts: List[Dict[str, Any]], doc_id: str):
        """
        添加概念到概念集合
        
        Args:
            concepts: 概念列表
            doc_id: 关联的文档ID
        """
        try:
            embeddings = []
            documents = []
            metadatas = []
            ids = []
            
            for concept in concepts:
                concept_text = concept.get('term', '')
                definition = concept.get('definition', '')
                
                # 组合概念和定义作为文档
                concept_doc = f"{concept_text}: {definition}" if definition else concept_text
                
                # 生成嵌入
                embedding = self.encoder.encode(concept_doc).tolist()
                
                # 准备数据
                concept_id = f"{doc_id}_{concept_text}_{uuid.uuid4().hex[:8]}"
                
                embeddings.append(embedding)
                documents.append(concept_doc)
                metadatas.append({
                    'term': concept_text,
                    'definition': definition,
                    'doc_id': doc_id,
                    'type': concept.get('type', 'unknown'),
                    'confidence': concept.get('final_score', 0.5)
                })
                ids.append(concept_id)
            
            if embeddings:
                self.concept_collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"成功添加 {len(concepts)} 个概念")
                
        except Exception as e:
            logger.error(f"添加概念失败: {e}")
            raise
    
    def search_similar_documents(self, query: str, n_results: int = 5, 
                                threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            threshold: 相似度阈值
            
        Returns:
            相似文档列表
        """
        try:
            # 生成查询嵌入
            query_embedding = self.encoder.encode(query).tolist()
            
            # 搜索
            results = self.knowledge_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # 处理结果
            similar_docs = []
            
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    # 计算相似度分数 (ChromaDB返回的是距离，需要转换)
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # 余弦相似度
                    
                    if similarity >= threshold:
                        similar_docs.append({
                            'id': results['ids'][0][i],
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity': similarity
                        })
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"搜索相似文档失败: {e}")
            return []
    
    def search_related_concepts(self, query: str, n_results: int = 10,
                               threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        搜索相关概念
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            threshold: 相似度阈值
            
        Returns:
            相关概念列表
        """
        try:
            # 生成查询嵌入
            query_embedding = self.encoder.encode(query).tolist()
            
            # 搜索概念
            results = self.concept_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # 处理结果
            related_concepts = []
            
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    similarity = 1 - distance
                    
                    if similarity >= threshold:
                        metadata = results['metadatas'][0][i]
                        related_concepts.append({
                            'id': results['ids'][0][i],
                            'term': metadata.get('term', ''),
                            'definition': metadata.get('definition', ''),
                            'doc_id': metadata.get('doc_id', ''),
                            'type': metadata.get('type', 'unknown'),
                            'confidence': metadata.get('confidence', 0.5),
                            'similarity': similarity
                        })
            
            return related_concepts
            
        except Exception as e:
            logger.error(f"搜索相关概念失败: {e}")
            return []
    
    def find_concept_links(self, concepts: List[str], threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        找出概念间的链接关系
        
        Args:
            concepts: 概念列表
            threshold: 相似度阈值
            
        Returns:
            概念链接列表 [(concept1, concept2, similarity), ...]
        """
        links = []
        
        try:
            # 获取所有概念的嵌入
            concept_embeddings = {}
            for concept in concepts:
                embedding = self.encoder.encode(concept)
                concept_embeddings[concept] = embedding
            
            # 计算两两相似度
            concept_list = list(concepts)
            for i in range(len(concept_list)):
                for j in range(i + 1, len(concept_list)):
                    concept1 = concept_list[i]
                    concept2 = concept_list[j]
                    
                    # 计算余弦相似度
                    emb1 = concept_embeddings[concept1]
                    emb2 = concept_embeddings[concept2]
                    
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    
                    if similarity >= threshold:
                        links.append((concept1, concept2, float(similarity)))
            
            # 按相似度排序
            links.sort(key=lambda x: x[2], reverse=True)
            
        except Exception as e:
            logger.error(f"计算概念链接失败: {e}")
        
        return links
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取文档"""
        try:
            result = self.knowledge_collection.get(ids=[doc_id])
            
            if result['ids'] and result['ids'][0]:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            
        except Exception as e:
            logger.error(f"获取文档失败: {e}")
            
        return None
    
    def update_document(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """更新文档"""
        try:
            # 生成新的嵌入
            embedding = self.encoder.encode(content).tolist()
            
            # 更新文档
            self.knowledge_collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )
            
            logger.info(f"成功更新文档: {doc_id}")
            
        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            raise
    
    def delete_document(self, doc_id: str):
        """删除文档及其相关概念"""
        try:
            # 删除主文档
            self.knowledge_collection.delete(ids=[doc_id])
            
            # 删除相关概念
            # 由于ChromaDB不支持复杂查询，我们需要先查询再删除
            all_concepts = self.concept_collection.get()
            
            concept_ids_to_delete = []
            if all_concepts['metadatas']:
                for i, metadata in enumerate(all_concepts['metadatas']):
                    if metadata.get('doc_id') == doc_id:
                        concept_ids_to_delete.append(all_concepts['ids'][i])
            
            if concept_ids_to_delete:
                self.concept_collection.delete(ids=concept_ids_to_delete)
            
            logger.info(f"成功删除文档和相关概念: {doc_id}")
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            raise
    
    def get_all_concepts(self) -> List[Dict[str, Any]]:
        """获取所有概念"""
        try:
            results = self.concept_collection.get()
            
            concepts = []
            if results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    concepts.append({
                        'id': results['ids'][i],
                        'term': metadata.get('term', ''),
                        'definition': metadata.get('definition', ''),
                        'doc_id': metadata.get('doc_id', ''),
                        'type': metadata.get('type', 'unknown'),
                        'confidence': metadata.get('confidence', 0.5)
                    })
            
            return concepts
            
        except Exception as e:
            logger.error(f"获取所有概念失败: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            knowledge_count = self.knowledge_collection.count()
            concept_count = self.concept_collection.count()
            
            return {
                'knowledge_documents': knowledge_count,
                'concepts': concept_count,
                'db_path': self.db_path,
                'model_name': self.model_name
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def reset_database(self):
        """重置数据库（删除所有数据）"""
        try:
            self.client.reset()
            
            # 重新创建集合
            self.knowledge_collection = self._get_or_create_collection("knowledge_base")
            self.concept_collection = self._get_or_create_collection("concepts")
            
            logger.info("数据库已重置")
            
        except Exception as e:
            logger.error(f"重置数据库失败: {e}")
            raise