"""
主编排Agent - 协调各个工作者Agent的执行
"""
import uuid
from typing import Dict, List, Any, Optional
from agents.base_agent import BaseAgent
from agents.content_parser import ContentParser
from agents.structure_builder import StructureBuilder
from agents.link_discoverer import LinkDiscoverer
from utils.vector_db import LocalVectorDB
from utils.text_processor import TextProcessor
import logging
import os

logger = logging.getLogger(__name__)

class KnowledgeOrchestrator(BaseAgent):
    """知识整理主编排Agent"""
    
    def __init__(self, knowledge_base_path: str, vector_db_path: str = "./data/chroma_db"):
        super().__init__(
            name="知识整理编排专家",
            description="协调各个工作者Agent完成知识整理任务"
        )
        
        self.knowledge_base_path = knowledge_base_path
        
        # 初始化向量数据库
        self.vector_db = LocalVectorDB(vector_db_path)
        
        # 初始化工作者Agents
        self.content_parser = ContentParser()
        self.structure_builder = StructureBuilder()
        self.link_discoverer = LinkDiscoverer(self.vector_db)
        self.text_processor = TextProcessor()
        
        logger.info("知识整理编排Agent初始化完成")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理知识整理任务
        
        Args:
            input_data: {
                "content": str,          # 输入内容
                "type": str,             # 内容类型 (auto/text/url/conversation/markdown)
                "metadata": dict,        # 元数据
                "operation": str,        # 操作类型 (create/update/analyze)
                "target_file": str,      # 目标文件（更新操作时使用）
                "options": dict          # 处理选项
            }
        
        Returns:
            {
                "success": bool,
                "result": dict,          # 处理结果
                "output_file": str,      # 输出文件路径
                "doc_id": str,          # 文档ID
                "statistics": dict,      # 处理统计
                "errors": List[str]      # 错误信息
            }
        """
        
        content = input_data.get("content", "")
        content_type = input_data.get("type", "auto")
        metadata = input_data.get("metadata", {})
        operation = input_data.get("operation", "create")
        target_file = input_data.get("target_file", "")
        options = input_data.get("options", {})
        
        result = {
            "success": False,
            "result": {},
            "output_file": "",
            "doc_id": "",
            "statistics": {},
            "errors": []
        }
        
        try:
            # 1. 分析任务复杂度并选择策略
            strategy = self._analyze_task_complexity(content, operation, options)
            logger.info(f"选择处理策略: {strategy}")
            
            # 2. 根据策略执行处理
            if operation == "create":
                result = self._create_new_document(content, content_type, metadata, options, strategy)
            elif operation == "update":
                result = self._update_existing_document(content, target_file, metadata, options, strategy)
            elif operation == "analyze":
                result = self._analyze_content(content, content_type, metadata, options)
            else:
                result["errors"].append(f"不支持的操作类型: {operation}")
                return result
            
            result["success"] = len(result["errors"]) == 0
            
        except Exception as e:
            logger.error(f"处理任务失败: {str(e)}")
            result["errors"].append(f"处理失败: {str(e)}")
            
        return result
    
    def _analyze_task_complexity(self, content: str, operation: str, options: Dict[str, Any]) -> str:
        """分析任务复杂度并选择处理策略"""
        # 基本复杂度因素
        factors = {
            'content_length': len(content),
            'operation_type': operation,
            'has_structure': bool(options.get('force_structure', False)),
            'requires_links': bool(options.get('enable_linking', True)),
            'batch_mode': bool(options.get('batch_mode', False))
        }
        
        # 选择策略
        if factors['content_length'] > 10000:
            if operation == "create":
                return "hierarchical_processing"
            else:
                return "streaming_processing"
        elif factors['batch_mode']:
            return "batch_processing"
        elif operation == "update":
            return "incremental_update"
        else:
            return "standard_processing"
    
    def _create_new_document(self, content: str, content_type: str, metadata: Dict[str, Any], 
                           options: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """创建新文档"""
        result = {
            "success": False,
            "result": {},
            "output_file": "",
            "doc_id": "",
            "statistics": {},
            "errors": []
        }
        
        try:
            doc_id = str(uuid.uuid4())
            
            # Step 1: 内容解析
            logger.info("开始内容解析...")
            parse_input = {
                "content": content,
                "type": content_type,
                "metadata": metadata
            }
            
            if strategy == "hierarchical_processing":
                parse_result = self._hierarchical_content_parsing(parse_input)
            elif strategy == "streaming_processing":
                parse_result = self._streaming_content_parsing(parse_input)
            else:
                parse_result = self.content_parser.process(parse_input)
            
            if not parse_result.get("parsed_content"):
                result["errors"].append("内容解析失败")
                return result
            
            # Step 2: 结构化构建
            logger.info("开始结构化构建...")
            structure_input = parse_result.copy()
            structure_result = self.structure_builder.process(structure_input)
            
            if not structure_result.get("structured_content"):
                result["errors"].append("结构化构建失败")
                return result
            
            # Step 3: 链接发现
            if options.get("enable_linking", True):
                logger.info("开始链接发现...")
                link_input = {
                    "concepts": structure_result.get("concepts", []),
                    "structured_content": structure_result.get("structured_content", ""),
                    "metadata": structure_result.get("metadata", {}),
                    "doc_id": doc_id
                }
                link_result = self.link_discoverer.process(link_input)
                
                # 使用更新后的内容
                final_content = link_result.get("updated_content", structure_result["structured_content"])
            else:
                link_result = {"concept_links": [], "existing_links": [], "relationship_map": {}}
                final_content = structure_result["structured_content"]
            
            # Step 4: 保存到向量数据库
            if options.get("enable_vector_db", True):
                logger.info("保存到向量数据库...")
                try:
                    self.vector_db.add_document(
                        content=final_content,
                        metadata={
                            **structure_result.get("metadata", {}),
                            "doc_id": doc_id,
                            "original_type": content_type,
                            "processing_strategy": strategy
                        },
                        doc_id=doc_id
                    )
                    
                    # 添加概念
                    concepts = structure_result.get("concepts", [])
                    if concepts:
                        self.vector_db.add_concepts(concepts, doc_id)
                        
                except Exception as e:
                    logger.warning(f"保存到向量数据库失败: {e}")
                    result["errors"].append(f"向量数据库保存失败: {e}")
            
            # Step 5: 保存文件
            output_file = self._save_to_file(final_content, structure_result, doc_id)
            
            # 组装结果
            result.update({
                "success": True,
                "result": {
                    "content": final_content,
                    "concepts": structure_result.get("concepts", []),
                    "outline": structure_result.get("outline", {}),
                    "tags": structure_result.get("tags", []),
                    "links": link_result.get("concept_links", []),
                    "external_links": link_result.get("existing_links", []),
                    "relationship_map": link_result.get("relationship_map", {})
                },
                "output_file": output_file,
                "doc_id": doc_id,
                "statistics": {
                    "original_length": len(content),
                    "processed_length": len(final_content),
                    "concept_count": len(structure_result.get("concepts", [])),
                    "internal_links": len(link_result.get("concept_links", [])),
                    "external_links": len(link_result.get("existing_links", [])),
                    "processing_strategy": strategy
                }
            })
            
        except Exception as e:
            logger.error(f"创建文档失败: {str(e)}")
            result["errors"].append(f"创建文档失败: {str(e)}")
        
        return result
    
    def _hierarchical_content_parsing(self, parse_input: Dict[str, Any]) -> Dict[str, Any]:
        """层次化内容解析"""
        content = parse_input["content"]
        
        # 使用文本处理器进行层次化处理
        strategy = self.text_processor.choose_processing_strategy(content)
        
        if strategy == "hierarchical":
            chunks = self.text_processor.hierarchical_processing(content)
        else:
            chunks = self.text_processor.hybrid_processing(content)
        
        # 分别处理每个块，然后合并
        all_parsed_content = []
        combined_metadata = {}
        
        for chunk in chunks:
            chunk_input = parse_input.copy()
            chunk_input["content"] = chunk.content
            
            chunk_result = self.content_parser.process(chunk_input)
            all_parsed_content.append(chunk_result.get("parsed_content", ""))
            
            # 合并元数据
            chunk_metadata = chunk_result.get("metadata", {})
            for key, value in chunk_metadata.items():
                if key not in combined_metadata:
                    combined_metadata[key] = value
                elif isinstance(value, list):
                    combined_metadata[key].extend(value)
        
        # 合并结果
        return {
            "parsed_content": "\n\n".join(all_parsed_content),
            "content_type": parse_input.get("type", "text"),
            "structure": {"chunks": len(chunks), "processing": "hierarchical"},
            "metadata": combined_metadata,
            "chunks": [chunk.content for chunk in chunks]
        }
    
    def _streaming_content_parsing(self, parse_input: Dict[str, Any]) -> Dict[str, Any]:
        """流式内容解析"""
        content = parse_input["content"]
        
        # 使用流式处理
        chunk_generator = self.text_processor.streaming_processing(content)
        
        all_parsed_content = []
        combined_metadata = {}
        chunk_count = 0
        
        for chunk in chunk_generator:
            chunk_input = parse_input.copy()
            chunk_input["content"] = chunk.content
            
            chunk_result = self.content_parser.process(chunk_input)
            all_parsed_content.append(chunk_result.get("parsed_content", ""))
            
            # 合并元数据
            chunk_metadata = chunk_result.get("metadata", {})
            for key, value in chunk_metadata.items():
                if key not in combined_metadata:
                    combined_metadata[key] = value
                elif isinstance(value, list):
                    combined_metadata[key].extend(value)
            
            chunk_count += 1
        
        return {
            "parsed_content": "\n\n".join(all_parsed_content),
            "content_type": parse_input.get("type", "text"),
            "structure": {"chunks": chunk_count, "processing": "streaming"},
            "metadata": combined_metadata,
            "chunks": all_parsed_content
        }
    
    def _update_existing_document(self, content: str, target_file: str, metadata: Dict[str, Any], 
                                options: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """更新现有文档"""
        result = {
            "success": False,
            "result": {},
            "output_file": "",
            "doc_id": "",
            "statistics": {},
            "errors": []
        }
        
        try:
            # 1. 读取现有文档
            if not os.path.exists(target_file):
                result["errors"].append(f"目标文件不存在: {target_file}")
                return result
            
            with open(target_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            
            # 2. 查找相关文档ID (从文件名或内容中提取)
            doc_id = self._extract_doc_id(target_file, existing_content)
            
            # 3. 分析新内容与现有内容的关系
            similarity_docs = self.vector_db.search_similar_documents(content, n_results=3)
            
            # 4. 决定更新策略
            if similarity_docs and any(d['similarity'] > 0.8 for d in similarity_docs):
                # 高相似度，进行增量更新
                updated_result = self._incremental_update(content, existing_content, doc_id, metadata)
            else:
                # 低相似度，作为新内容合并
                updated_result = self._merge_new_content(content, existing_content, doc_id, metadata)
            
            result.update(updated_result)
            
        except Exception as e:
            logger.error(f"更新文档失败: {str(e)}")
            result["errors"].append(f"更新文档失败: {str(e)}")
        
        return result
    
    def _analyze_content(self, content: str, content_type: str, metadata: Dict[str, Any], 
                        options: Dict[str, Any]) -> Dict[str, Any]:
        """分析内容但不创建文档"""
        result = {
            "success": False,
            "result": {},
            "output_file": "",
            "doc_id": "",
            "statistics": {},
            "errors": []
        }
        
        try:
            # 只进行内容解析和结构化分析
            parse_input = {
                "content": content,
                "type": content_type,
                "metadata": metadata
            }
            
            parse_result = self.content_parser.process(parse_input)
            structure_result = self.structure_builder.process(parse_result)
            
            # 分析相关内容
            related_docs = self.vector_db.search_similar_documents(content, n_results=5)
            related_concepts = self.vector_db.search_related_concepts(content, n_results=10)
            
            result.update({
                "success": True,
                "result": {
                    "analysis": {
                        "content_type": parse_result.get("content_type"),
                        "complexity": structure_result.get("metadata", {}).get("complexity_assessed"),
                        "main_concepts": structure_result.get("concepts", [])[:5],
                        "estimated_reading_time": parse_result.get("structure", {}).get("estimated_reading_time", 0)
                    },
                    "related_documents": related_docs,
                    "related_concepts": related_concepts,
                    "suggestions": {
                        "recommended_tags": structure_result.get("tags", []),
                        "potential_links": len(related_concepts),
                        "similar_documents": len(related_docs)
                    }
                },
                "statistics": {
                    "original_length": len(content),
                    "concept_count": len(structure_result.get("concepts", [])),
                    "related_doc_count": len(related_docs),
                    "related_concept_count": len(related_concepts)
                }
            })
            
        except Exception as e:
            logger.error(f"分析内容失败: {str(e)}")
            result["errors"].append(f"分析内容失败: {str(e)}")
        
        return result
    
    def _incremental_update(self, new_content: str, existing_content: str, doc_id: str, 
                          metadata: Dict[str, Any]) -> Dict[str, Any]:
        """增量更新现有文档"""
        # 实现增量更新逻辑
        # 这里简化实现，实际可以更复杂
        
        # 在现有内容末尾添加新内容
        separator = "\n\n---\n\n## 补充内容\n\n"
        merged_content = existing_content + separator + new_content
        
        # 重新处理整个文档
        return self._create_new_document(merged_content, "markdown", metadata, {}, "standard_processing")
    
    def _merge_new_content(self, new_content: str, existing_content: str, doc_id: str, 
                         metadata: Dict[str, Any]) -> Dict[str, Any]:
        """合并新内容到现有文档"""
        # 智能合并逻辑
        # 这里简化实现
        
        # 分析新内容的主题
        parse_result = self.content_parser.process({
            "content": new_content,
            "type": "auto",
            "metadata": metadata
        })
        
        structure_result = self.structure_builder.process(parse_result)
        main_topic = structure_result.get("outline", {}).get("title", "新增内容")
        
        # 在适当位置插入新内容
        new_section = f"\n\n## {main_topic}\n\n{new_content}"
        merged_content = existing_content + new_section
        
        return self._create_new_document(merged_content, "markdown", metadata, {}, "standard_processing")
    
    def _save_to_file(self, content: str, structure_result: Dict[str, Any], doc_id: str) -> str:
        """保存内容到文件"""
        # 生成文件名
        title = structure_result.get("outline", {}).get("title", "知识笔记")
        # 清理文件名中的特殊字符
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_')[:50]  # 限制长度
        
        filename = f"{safe_title}_{doc_id[:8]}.md"
        filepath = os.path.join(self.knowledge_base_path, filename)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"文件已保存: {filepath}")
        return filepath
    
    def _extract_doc_id(self, filepath: str, content: str) -> str:
        """从文件路径或内容中提取文档ID"""
        # 尝试从文件名中提取
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        if len(parts) > 1:
            potential_id = parts[-1].replace('.md', '')
            if len(potential_id) == 8:  # UUID的前8位
                return potential_id
        
        # 生成新的ID
        return str(uuid.uuid4())
    
    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的知识整理编排专家，负责协调各个专业Agent完成复杂的知识整理任务。
你的职责包括：
1. 分析任务复杂度并选择最优处理策略
2. 协调内容解析、结构化构建、链接发现等各个环节
3. 确保处理流程的质量和效率
4. 提供详细的处理结果和统计信息

请始终保持专业性和准确性，确保知识整理的质量。"""