"""
主编排Agent - 协调各个工作者Agent的执行
"""
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Callable
from .base_agent import BaseAgent
from .content_parser import ContentParser
from .structure_builder import StructureBuilder
from .link_discoverer import LinkDiscoverer
from ..utils.vector_db import LocalVectorDB
from ..utils.text_processor import TextProcessor
from ..utils.link_manager import LinkManager
from ..utils.progress_websocket import create_progress_callback, ProgressBroadcaster
import logging
import os
from enum import Enum
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    """任务复杂度等级"""
    SIMPLE = "simple_task"
    MEDIUM = "medium_task" 
    COMPLEX = "complex_task"

class ProcessingStage(Enum):
    """处理阶段"""
    ANALYZING = "analyzing"
    GENERATING_WORKERS = "generating_workers"
    WORKER_PROCESSING = "worker_processing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"

@dataclass
class ProcessingProgress:
    """处理进度信息"""
    task_id: str
    complexity: TaskComplexity
    stage: ProcessingStage
    current_step: str
    total_steps: int
    completed_steps: int
    workers: List[str] = None
    error: str = None
    start_time: float = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "complexity": self.complexity.value,
            "stage": self.stage.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "progress_percent": (self.completed_steps / self.total_steps * 100) if self.total_steps > 0 else 0,
            "workers": self.workers or [],
            "error": self.error,
            "elapsed_time": time.time() - self.start_time if self.start_time else 0
        }

class KnowledgeOrchestrator(BaseAgent):
    """知识整理主编排Agent"""
    
    def __init__(self, knowledge_base_path: str, vector_db_path: str = "./data/chroma_db", 
                 progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
                 enable_websocket: bool = True):
        super().__init__(
            name="知识整理编排专家",
            description="协调各个工作者Agent完成知识整理任务"
        )
        
        self.knowledge_base_path = knowledge_base_path
        self.current_progress: Optional[ProcessingProgress] = None
        
        # 设置进度回调
        if progress_callback is None and enable_websocket:
            # 使用默认的WebSocket广播器
            self.progress_callback = create_progress_callback()
        else:
            self.progress_callback = progress_callback
        
        # 初始化向量数据库
        self.vector_db = LocalVectorDB(vector_db_path)
        
        # 初始化链接管理器
        self.link_manager = LinkManager(knowledge_base_path)
        
        # 初始化工作者Agents
        self.content_parser = ContentParser()
        self.structure_builder = StructureBuilder()
        self.link_discoverer = LinkDiscoverer(self.vector_db)
        self.text_processor = TextProcessor()
        
        logger.info("知识整理编排Agent初始化完成")
    
    def _update_progress(self, stage: ProcessingStage, current_step: str, completed_steps: int = None, 
                        workers: List[str] = None, error: str = None):
        """更新处理进度"""
        if self.current_progress:
            self.current_progress.stage = stage
            self.current_progress.current_step = current_step
            if completed_steps is not None:
                self.current_progress.completed_steps = completed_steps
            if workers is not None:
                self.current_progress.workers = workers
            if error is not None:
                self.current_progress.error = error
            
            # 调用进度回调
            if self.progress_callback:
                self.progress_callback(self.current_progress)
                
            logger.info(f"进度更新: {stage.value} - {current_step}")
    
    def _determine_complexity(self, content: str, operation: str, options: Dict[str, Any]) -> TaskComplexity:
        """确定任务复杂度"""
        content_length = len(content)
        
        # 复杂度判断逻辑
        complexity_factors = {
            'length': content_length,
            'has_structure': bool(options.get('force_structure', False)),
            'requires_links': bool(options.get('enable_linking', True)),
            'batch_mode': bool(options.get('batch_mode', False)),
            'update_mode': operation == "update"
        }
        
        # 简单任务：短文本，无特殊要求
        if (content_length < 1000 and not complexity_factors['batch_mode'] 
            and not complexity_factors['update_mode']):
            return TaskComplexity.SIMPLE
        
        # 复杂任务：长文本或多个复杂要求
        elif (content_length > 10000 or 
              sum([complexity_factors['has_structure'], 
                   complexity_factors['batch_mode'], 
                   complexity_factors['update_mode']]) >= 2):
            return TaskComplexity.COMPLEX
        
        # 中等任务：其他情况
        else:
            return TaskComplexity.MEDIUM
    
    def _get_worker_list(self, complexity: TaskComplexity, operation: str) -> List[str]:
        """根据复杂度获取需要的工作者列表"""
        base_workers = ["内容解析器", "结构构建器"]
        
        if complexity == TaskComplexity.SIMPLE:
            # 单Agent独立处理
            return ["内容处理Agent"]
        elif complexity == TaskComplexity.MEDIUM:
            # 生成3-4个工作者
            workers = base_workers + ["链接发现器"]
            if operation == "update":
                workers.append("增量更新器")
            return workers
        else:  # COMPLEX
            # 生成5-6个工作者并行处理
            workers = base_workers + ["链接发现器", "概念提取器", "关系分析器"]
            if operation == "update":
                workers.append("智能合并器")
            return workers
    
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
                "errors": List[str],     # 错误信息
                "progress": dict         # 最终进度信息
            }
        """
        
        content = input_data.get("content", "")
        content_type = input_data.get("type", "auto")
        metadata = input_data.get("metadata", {})
        operation = input_data.get("operation", "create")
        target_file = input_data.get("target_file", "")
        options = input_data.get("options", {})
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        result = {
            "success": False,
            "result": {},
            "output_file": "",
            "doc_id": "",
            "statistics": {},
            "errors": [],
            "task_id": task_id,
            "progress": {}
        }
        
        try:
            # 1. 初始化进度跟踪
            complexity = self._determine_complexity(content, operation, options)
            workers = self._get_worker_list(complexity, operation)
            
            # 根据复杂度设置总步数
            if complexity == TaskComplexity.SIMPLE:
                total_steps = 3  # 分析->处理->完成
            elif complexity == TaskComplexity.MEDIUM:
                total_steps = 5  # 分析->生成工作者->内容解析->结构构建->链接发现->完成
            else:  # COMPLEX
                total_steps = 7  # 分析->生成工作者->并行处理(多步)->完成
            
            self.current_progress = ProcessingProgress(
                task_id=task_id,
                complexity=complexity,
                stage=ProcessingStage.ANALYZING,
                current_step="Agent识别中",
                total_steps=total_steps,
                completed_steps=0,
                workers=[],
                start_time=time.time()
            )
            
            # 更新进度：分析阶段
            self._update_progress(ProcessingStage.ANALYZING, "Agent识别中", 0)
            
            # 2. 分析任务复杂度并选择策略  
            strategy = self._analyze_task_complexity(content, operation, options)
            logger.info(f"选择处理策略: {strategy}")
            
            # 3. 根据复杂度显示不同的进度信息
            if complexity == TaskComplexity.SIMPLE:
                # 简单任务：显示 "Agent处理中"
                self._update_progress(ProcessingStage.WORKER_PROCESSING, "Agent处理中", 1)
            else:
                # 复杂任务：显示工作者生成
                self._update_progress(ProcessingStage.GENERATING_WORKERS, 
                                    f"生成了{len(workers)}个工作者: {', '.join(workers)}", 1, workers)
                
                # 显示工作者处理
                self._update_progress(ProcessingStage.WORKER_PROCESSING, "工作者处理中", 2, workers)
            
            # 4. 根据策略执行处理
            if operation == "create":
                result.update(self._create_new_document(content, content_type, metadata, options, strategy))
            elif operation == "update":
                result.update(self._update_existing_document(content, target_file, metadata, options, strategy))
            elif operation == "analyze":
                result.update(self._analyze_content(content, content_type, metadata, options))
            else:
                result["errors"].append(f"不支持的操作类型: {operation}")
                self._update_progress(ProcessingStage.COMPLETED, "处理失败", total_steps, 
                                    workers, f"不支持的操作类型: {operation}")
                return result
            
            # 5. 完成处理
            if len(result["errors"]) == 0:
                result["success"] = True
                self._update_progress(ProcessingStage.COMPLETED, "处理完成", total_steps, workers)
            else:
                self._update_progress(ProcessingStage.COMPLETED, "处理失败", total_steps, 
                                    workers, "; ".join(result["errors"]))
            
            # 添加最终进度信息到结果中
            result["progress"] = self.current_progress.to_dict() if self.current_progress else {}
            
        except Exception as e:
            logger.error(f"处理任务失败: {str(e)}")
            error_msg = f"处理失败: {str(e)}"
            result["errors"].append(error_msg)
            
            if self.current_progress:
                self._update_progress(ProcessingStage.COMPLETED, "处理异常", 
                                    self.current_progress.total_steps, 
                                    self.current_progress.workers, error_msg)
                result["progress"] = self.current_progress.to_dict()
            
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
            if self.current_progress:
                complexity = self.current_progress.complexity
                if complexity != TaskComplexity.SIMPLE:
                    self._update_progress(ProcessingStage.WORKER_PROCESSING, "内容解析器处理中", 3)
            
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
            if self.current_progress:
                complexity = self.current_progress.complexity
                if complexity == TaskComplexity.MEDIUM:
                    self._update_progress(ProcessingStage.WORKER_PROCESSING, "结构构建器处理中", 4)
                elif complexity == TaskComplexity.COMPLEX:
                    self._update_progress(ProcessingStage.WORKER_PROCESSING, "结构构建器处理中", 4)
            
            structure_input = parse_result.copy()
            structure_result = self.structure_builder.process(structure_input)
            
            if not structure_result.get("structured_content"):
                result["errors"].append("结构化构建失败")
                return result
            
            # Step 3: 链接发现
            if options.get("enable_linking", True):
                logger.info("开始链接发现...")
                if self.current_progress:
                    complexity = self.current_progress.complexity
                    if complexity == TaskComplexity.MEDIUM:
                        self._update_progress(ProcessingStage.WORKER_PROCESSING, "链接发现器处理中", 5)
                    elif complexity == TaskComplexity.COMPLEX:
                        self._update_progress(ProcessingStage.WORKER_PROCESSING, "概念提取器和关系分析器处理中", 5)
                
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
                if self.current_progress:
                    complexity = self.current_progress.complexity
                    if complexity == TaskComplexity.COMPLEX:
                        self._update_progress(ProcessingStage.FINALIZING, "保存到向量数据库", 6)
                
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
            if self.current_progress:
                complexity = self.current_progress.complexity
                if complexity == TaskComplexity.SIMPLE:
                    self._update_progress(ProcessingStage.FINALIZING, "保存文件", 2)
                else:
                    self._update_progress(ProcessingStage.FINALIZING, "保存文件", 
                                        self.current_progress.completed_steps + 1)
            
            output_file = self._save_to_file(final_content, structure_result, doc_id)
            
            # Step 6: 更新链接数据库
            if options.get("enable_linking", True):
                logger.info("更新链接数据库...")
                if self.current_progress:
                    self._update_progress(ProcessingStage.FINALIZING, "更新链接数据库", 
                                        self.current_progress.completed_steps + 1)
                
                try:
                    # 处理单个文档的链接更新
                    self.link_manager._process_document(output_file)
                    # 重新解析所有链接
                    self.link_manager._resolve_all_links()
                except Exception as e:
                    logger.warning(f"更新链接数据库失败: {e}")
                    result["errors"].append(f"链接数据库更新失败: {e}")
            
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