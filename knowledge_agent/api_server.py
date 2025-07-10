#!/usr/bin/env python3
"""
Knowledge Agent API Server
使用 FastAPI 为 Knowledge Agent 提供 HTTP API 接口
"""

import asyncio
import json
import logging
import os
import traceback
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 添加当前目录到Python路径
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入知识库 Agent 组件
try:
    from agents.orchestrator import KnowledgeOrchestrator
    from utils.vector_db import LocalVectorDB
    from link_system import LinkManager
    from utils.file_watcher import create_file_watcher
    from utils.link_renderer import ConceptGraphGenerator
    from config import Settings
    from simple_processor import SimpleKnowledgeProcessor
    from core.strategy_history import StrategyHistoryDB
    from core.strategy_optimizer import StrategyOptimizer
    from core.strategy_evaluator import StrategyEvaluator
    from core.strategy_learner import StrategyLearner
    from core.history_analyzer import HistoryAnalyzer
    FULL_MODE = True
except ImportError as e:
    print(f"警告：无法导入完整的Agent组件: {e}")
    print("使用简化模式运行...")
    FULL_MODE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="Knowledge Agent API",
    description="智能知识库管理系统 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
orchestrator = None
vector_db = None
link_manager = None
file_watcher = None
simple_processor = None

# 策略优化相关组件
strategy_history_db = None
strategy_optimizer = None
strategy_evaluator = None
strategy_learner = None
history_analyzer = None

# 简化的WebSocket连接管理
active_websocket_connections = set()

# 全局任务管理
active_tasks = {}

# 自定义异常
class ProcessingStoppedException(Exception):
    """处理被用户停止的异常"""
    pass

# Pydantic 模型定义
class ProcessingOptions(BaseModel):
    strategy: str = Field(default="standard", description="处理策略")
    enableLinking: bool = Field(default=True, description="启用链接发现")
    generateSummary: bool = Field(default=True, description="生成摘要")
    extractConcepts: bool = Field(default=True, description="提取概念")
    enable_vector_db: bool = Field(default=True, description="启用向量数据库")
    force_structure: bool = Field(default=False, description="强制结构化")

class ProcessingRequest(BaseModel):
    content: str = Field(..., description="要处理的内容")
    type: str = Field(default="text", description="内容类型")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    options: ProcessingOptions = Field(default_factory=ProcessingOptions, description="处理选项")

class ProcessingResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    output_file: Optional[str] = None
    doc_id: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)
    message: Optional[str] = None

class ConceptSearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    limit: int = Field(default=10, description="返回结果数量")
    threshold: float = Field(default=0.6, description="相似度阈值")

class ConceptSearchResponse(BaseModel):
    concepts: List[Dict[str, Any]]
    total: int
    query: str

class DocumentSearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    limit: int = Field(default=10, description="返回结果数量")
    threshold: float = Field(default=0.6, description="相似度阈值")

class DocumentSearchResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total: int
    query: str

class LinkDiscoveryRequest(BaseModel):
    doc_id: str = Field(..., description="文档ID")
    threshold: float = Field(default=0.7, description="链接发现阈值")

class LinkDiscoveryResponse(BaseModel):
    links: List[Dict[str, Any]]
    total: int
    doc_id: str

class StatsResponse(BaseModel):
    documents: int
    concepts: int
    links: int
    last_updated: str

class ConceptGraphRequest(BaseModel):
    max_concepts: int = Field(default=100, description="最大概念数量", ge=1, le=500)
    include_documents: bool = Field(default=False, description="是否包含文档节点")

class ConceptGraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    total_concepts: int
    total_links: int

# 简化的进度广播函数
async def broadcast_progress(progress_data: Dict[str, Any]):
    """向所有WebSocket连接广播进度更新"""
    global active_websocket_connections
    
    if not active_websocket_connections:
        logger.info("没有活跃的WebSocket连接，跳过广播")
        return
    
    message = {
        "type": "progress_update",
        "data": progress_data,
        "timestamp": time.time()
    }
    
    logger.info(f"📡 广播进度更新给 {len(active_websocket_connections)} 个客户端: {progress_data.get('current_step', 'unknown')}")
    
    # 并发发送给所有客户端
    disconnected_connections = set()
    for websocket in active_websocket_connections.copy():
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
            logger.info(f"✅ 成功发送消息到WebSocket")
        except Exception as e:
            logger.error(f"❌ 发送WebSocket消息失败: {e}")
            disconnected_connections.add(websocket)
    
    # 清理断开的连接
    active_websocket_connections -= disconnected_connections
    logger.info(f"📊 WebSocket广播完成，剩余连接: {len(active_websocket_connections)}")

# 简化的进度回调类
class SimpleProgressCallback:
    """简化的进度回调"""
    
    def __call__(self, progress):
        """进度回调函数"""
        try:
            logger.info(f"🎯 SimpleProgressCallback被调用！")
            progress_data = progress.to_dict()
            
            # 记录进度数据用于调试
            logger.info(f"📊 [{progress.task_id[:8]}] 发送进度更新: {progress.current_step} "
                      f"({progress.completed_steps}/{progress.total_steps}) - 客户端数量: {len(active_websocket_connections)}")
            logger.info(f"📊 进度数据: {progress_data}")
            
            # 同步广播进度 - 使用asyncio.create_task在事件循环中运行
            try:
                loop = asyncio.get_event_loop()
                logger.info(f"🔄 获取事件循环成功，正在运行: {loop.is_running()}")
                if loop.is_running():
                    task = asyncio.create_task(broadcast_progress(progress_data))
                    logger.info(f"📤 创建广播任务: {task}")
                else:
                    logger.info(f"📤 事件循环未运行，直接运行广播")
                    loop.run_until_complete(broadcast_progress(progress_data))
            except RuntimeError as e:
                logger.warning(f"⚠️ 事件循环异常: {e}，尝试创建新的")
                # 如果没有事件循环，尝试创建新的
                asyncio.run(broadcast_progress(progress_data))
            
            # 记录关键进度点
            if progress.stage.value in ["analyzing", "generating_workers", "completed"]:
                logger.info(f"🎯 [{progress.task_id[:8]}] {progress.current_step} "
                          f"({progress.completed_steps}/{progress.total_steps})")
            
        except Exception as e:
            logger.error(f"❌ 广播进度失败: {e}")
            import traceback
            logger.error(f"❌ 错误堆栈: {traceback.format_exc()}")

# 启动时初始化
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化组件"""
    global orchestrator, vector_db, link_manager, file_watcher, simple_processor
    global strategy_history_db, strategy_optimizer, strategy_evaluator, strategy_learner, history_analyzer
    
    logger.info("正在初始化 Knowledge Agent API 服务器...")
    
    if not FULL_MODE:
        logger.warning("运行在简化模式下，某些功能可能不可用")
        return
    
    try:
        # 初始化配置
        settings = Settings()
        
        # 初始化简化处理器 - 新的主要处理器
        simple_processor = SimpleKnowledgeProcessor(broadcast_progress)
        logger.info("简化知识处理器初始化完成")
        
        # 初始化向量数据库
        vector_db = LocalVectorDB()
        logger.info("向量数据库初始化完成")
        
        # 初始化链接管理器
        link_manager = LinkManager(
            knowledge_base_path=settings.knowledge_base_path
        )
        logger.info("链接管理器初始化完成")
        
        # 初始化知识编排器 - 作为备用
        progress_callback = SimpleProgressCallback()
        
        orchestrator = KnowledgeOrchestrator(
            knowledge_base_path=settings.knowledge_base_path,
            progress_callback=progress_callback
        )
        logger.info("知识编排器初始化完成")
        
        # 初始化文件监控器
        def file_change_callback(change_info):
            """文件变化回调函数"""
            logger.info(f"文件变化通知: {change_info}")
            # 可以通过WebSocket通知前端
            if active_websocket_connections:
                asyncio.create_task(broadcast_progress(change_info))
        
        file_watcher = create_file_watcher(
            knowledge_base_path=settings.knowledge_base_path,
            link_manager=link_manager,
            callback=file_change_callback
        )
        file_watcher.start()
        logger.info("文件监控器启动完成")
        
        # 初始化策略优化组件
        try:
            strategy_history_db = StrategyHistoryDB()
            logger.info("策略历史数据库初始化完成")
            
            strategy_optimizer = StrategyOptimizer(strategy_history_db)
            logger.info("策略优化器初始化完成")
            
            strategy_evaluator = StrategyEvaluator(strategy_history_db)
            logger.info("策略评估器初始化完成")
            
            history_analyzer = HistoryAnalyzer(strategy_history_db)
            logger.info("历史分析器初始化完成")
            
            strategy_learner = StrategyLearner(strategy_history_db, strategy_optimizer)
            logger.info("策略学习器初始化完成")
            
            logger.info("✅ 策略优化系统初始化成功")
            
        except Exception as e:
            logger.error(f"❌ 策略优化系统初始化失败: {e}")
            logger.warning("系统将在不使用策略优化功能的情况下运行")
        
        logger.info("Knowledge Agent API 服务器启动成功!")
        
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# 关闭时清理
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global file_watcher
    
    logger.info("正在关闭 Knowledge Agent API 服务器...")
    
    if file_watcher:
        file_watcher.stop()
        logger.info("文件监控器已停止")
    
    # 关闭所有WebSocket连接
    for websocket in active_websocket_connections.copy():
        try:
            await websocket.close()
        except:
            pass
    active_websocket_connections.clear()
    
    logger.info("Knowledge Agent API 服务器已关闭")

# API 路由

@app.get("/", tags=["General"])
async def root():
    """根路径"""
    return {
        "message": "Knowledge Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health", tags=["General"])
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "simple_processor": simple_processor is not None,
            "orchestrator": orchestrator is not None,
            "vector_db": vector_db is not None,
            "link_manager": link_manager is not None,
            "file_watcher": file_watcher is not None and file_watcher.is_running if file_watcher else False
        }
    }

@app.get("/file-watcher/status", tags=["General"])
async def get_file_watcher_status():
    """获取文件监控器状态"""
    if not file_watcher:
        return {
            "enabled": False,
            "status": "未启用",
            "message": "文件监控器未初始化"
        }
    
    status = file_watcher.get_status()
    return {
        "enabled": True,
        "status": status,
        "message": "文件监控器运行正常" if status['is_running'] else "文件监控器已停止"
    }

@app.get("/stats", response_model=StatsResponse, tags=["General"])
async def get_stats():
    """获取系统统计信息"""
    try:
        # 获取向量数据库统计
        db_stats = vector_db.get_collection_stats() if vector_db else {}
        
        # 获取链接管理器统计
        link_stats = link_manager.get_stats() if link_manager else {}
        
        return StatsResponse(
            documents=db_stats.get("documents", 0),
            concepts=db_stats.get("concepts", 0),
            links=link_stats.get("total_links", 0),
            last_updated=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rescan", tags=["General"])
async def rescan_knowledge_base():
    """重新扫描知识库"""
    try:
        if not link_manager:
            raise HTTPException(status_code=500, detail="链接管理器未初始化")
        
        # 重新扫描知识库
        stats = await asyncio.get_event_loop().run_in_executor(
            None, link_manager.scan_knowledge_base_simple
        )
        
        return {
            "success": True,
            "message": "知识库重新扫描完成",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"重新扫描知识库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=ProcessingResponse, tags=["Processing"])
async def process_content(request: ProcessingRequest):
    """处理内容"""
    try:
        if not simple_processor:
            raise HTTPException(status_code=500, detail="简化处理器未初始化")
        
        import uuid
        task_id = str(uuid.uuid4())
        
        logger.info(f"开始处理内容，任务ID: {task_id}，长度: {len(request.content)} 字符")
        
        # 注册任务状态
        active_tasks[task_id] = "running"
        
        # 使用AI编排处理器（默认启用）
        result = await simple_processor.process_content(
            content=request.content,
            content_type=request.type,
            metadata=request.metadata,
            options={
                "enable_linking": request.options.enableLinking,
                "enable_vector_db": request.options.enable_vector_db,
                "force_structure": request.options.force_structure,
                "batch_mode": False,
                "enable_ai_orchestration": True,  # 启用AI编排
                "task_id": task_id  # 传递任务ID
            }
        )
        
        logger.info(f"✅ 处理完成: {'成功' if result.get('success') else '失败'}")
        
        return ProcessingResponse(
            success=result.get("success", False),
            result=result.get("result"),
            output_file="",  # 简化处理器暂不保存文件
            doc_id=result.get("doc_id"),
            statistics=result.get("result", {}).get("statistics", {}),
            errors=[result.get("error")] if result.get("error") else [],
            message="处理完成"
        )
        
    except ProcessingStoppedException:
        logger.info(f"任务 {task_id} 被用户停止")
        return ProcessingResponse(
            success=False,
            errors=["处理被用户停止"],
            message="处理已停止"
        )
    except Exception as e:
        logger.error(f"处理内容失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理任务状态
        active_tasks.pop(task_id, None)

@app.post("/upload", response_model=ProcessingResponse, tags=["Processing"])
async def upload_file(file: UploadFile = File(...)):
    """上传文件处理"""
    try:
        if not orchestrator:
            raise HTTPException(status_code=500, detail="编排器未初始化")
        
        # 读取文件内容
        content = await file.read()
        
        # 检查文件类型
        allowed_types = ['.md', '.txt', '.doc', '.docx']
        file_extension = '.' + file.filename.split('.')[-1].lower()
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型: {file_extension}. 支持的类型: {', '.join(allowed_types)}"
            )
        
        # 解码文件内容
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="文件编码不支持，请使用 UTF-8 编码")
        
        # 构建处理参数
        input_data = {
            "content": text_content,
            "type": "text",
            "metadata": {
                "source": file.filename,
                "upload_time": datetime.now().isoformat()
            },
            "operation": "create",
            "options": {
                "enable_linking": True,
                "enable_vector_db": True,
                "force_structure": False,
                "batch_mode": False
            }
        }
        
        # 执行处理
        result = await asyncio.get_event_loop().run_in_executor(
            None, orchestrator.process, input_data
        )
        
        return ProcessingResponse(
            success=result.get("success", False),
            result=result.get("result"),
            output_file=result.get("output_file"),
            doc_id=result.get("doc_id"),
            statistics=result.get("statistics"),
            errors=result.get("errors", []),
            message=f"文件 {file.filename} 处理完成"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传文件处理失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/concepts", response_model=ConceptSearchResponse, tags=["Search"])
async def search_concepts(request: ConceptSearchRequest):
    """搜索概念"""
    try:
        if not vector_db:
            raise HTTPException(status_code=500, detail="向量数据库未初始化")
        
        results = vector_db.search_related_concepts(
            query=request.query,
            n_results=request.limit,
            threshold=request.threshold
        )
        
        return ConceptSearchResponse(
            concepts=results,
            total=len(results),
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"搜索概念失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/documents", response_model=DocumentSearchResponse, tags=["Search"])
async def search_documents(request: DocumentSearchRequest):
    """搜索文档"""
    try:
        if not vector_db:
            raise HTTPException(status_code=500, detail="向量数据库未初始化")
        
        results = vector_db.search_similar_documents(
            query=request.query,
            n_results=request.limit,
            threshold=request.threshold
        )
        
        return DocumentSearchResponse(
            documents=results,
            total=len(results),
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"搜索文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/discover/links", response_model=LinkDiscoveryResponse, tags=["Links"])
async def discover_links(request: LinkDiscoveryRequest):
    """发现链接"""
    try:
        if not link_manager:
            raise HTTPException(status_code=500, detail="链接管理器未初始化")
        
        links = link_manager.discover_links_for_document(
            doc_id=request.doc_id,
            threshold=request.threshold
        )
        
        return LinkDiscoveryResponse(
            links=links,
            total=len(links),
            doc_id=request.doc_id
        )
        
    except Exception as e:
        logger.error(f"发现链接失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/concepts", tags=["Concepts"])
async def list_concepts(limit: int = 100, offset: int = 0):
    """列出所有概念"""
    try:
        if not link_manager:
            raise HTTPException(status_code=500, detail="链接管理器未初始化")
        
        concepts = link_manager.get_all_concepts(limit=limit, offset=offset)
        
        return {
            "concepts": concepts,
            "total": len(concepts),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"列出概念失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/concepts/{concept_name}", tags=["Concepts"])
async def get_concept(concept_name: str):
    """获取特定概念的详细信息"""
    try:
        if not link_manager:
            raise HTTPException(status_code=500, detail="链接管理器未初始化")
        
        concept_info = link_manager.get_concept_info(concept_name)
        
        if not concept_info:
            raise HTTPException(status_code=404, detail=f"概念 '{concept_name}' 不存在")
        
        return concept_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取概念信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/concept-graph", response_model=ConceptGraphResponse, tags=["Concepts"])
async def get_concept_graph(max_concepts: int = 100, include_documents: bool = False):
    """获取概念图谱数据"""
    try:
        if not link_manager:
            raise HTTPException(status_code=500, detail="链接管理器未初始化")
        
        # 创建概念图生成器
        graph_generator = ConceptGraphGenerator(link_manager)
        
        # 生成图谱数据
        graph_data = await asyncio.get_event_loop().run_in_executor(
            None, graph_generator.generate_graph_data, max_concepts
        )
        
        # 转换数据格式以符合前端预期
        nodes = []
        for node in graph_data['nodes']:
            nodes.append({
                'id': str(node['id']),
                'label': node['label'],
                'type': 'concept',
                'size': node['size'],
                'color': node['color'],
                'metadata': {
                    'referenceCount': node.get('size', 10) - 10,  # 从size反推引用次数
                    'hasDocument': node['color'] == '#0066cc',  # 蓝色表示有文档
                    'category': 'concept'
                }
            })
        
        # 转换链接数据
        links = []
        for link in graph_data['links']:
            links.append({
                'source': str(link['source']),
                'target': str(link['target']),
                'weight': link['weight'],
                'type': 'concept-link'
            })
        
        return ConceptGraphResponse(
            nodes=nodes,
            links=links,
            total_concepts=len(nodes),
            total_links=len(links)
        )
        
    except Exception as e:
        logger.error(f"获取概念图谱失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", tags=["Documents"])
async def list_documents(limit: int = 100, offset: int = 0):
    """列出所有文档"""
    try:
        if not link_manager:
            raise HTTPException(status_code=500, detail="链接管理器未初始化")
        
        documents = link_manager.get_all_documents(limit=limit, offset=offset)
        
        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"列出文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}", tags=["Documents"])
async def get_document(doc_id: str):
    """获取特定文档的详细信息"""
    try:
        if not link_manager:
            raise HTTPException(status_code=500, detail="链接管理器未初始化")
        
        doc_info = link_manager.get_document_info(doc_id)
        
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"文档 '{doc_id}' 不存在")
        
        return doc_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """删除文档"""
    try:
        if not link_manager:
            raise HTTPException(status_code=500, detail="链接管理器未初始化")
        
        # 获取文档信息
        doc_info = link_manager.get_document_info(doc_id)
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"文档 '{doc_id}' 不存在")
        
        doc_path = doc_info.get('doc_path')
        if not doc_path:
            raise HTTPException(status_code=404, detail="文档路径不存在")
        
        # 删除物理文件
        resolved_path = link_manager._resolve_document_path(doc_path)
        try:
            if os.path.exists(resolved_path):
                os.remove(resolved_path)
                logger.info(f"删除物理文件: {resolved_path}")
        except Exception as e:
            logger.warning(f"删除物理文件失败: {e}")
            # 即使文件删除失败，也继续删除数据库记录
        
        # 从数据库中删除文档记录
        success = link_manager.remove_document(doc_path)
        
        if success:
            return {
                "success": True,
                "message": f"文档 '{doc_info.get('title', doc_id)}' 删除成功",
                "doc_id": doc_id
            }
        else:
            raise HTTPException(status_code=500, detail="删除文档记录失败")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}/content", tags=["Documents"])
async def get_document_content(doc_id: str):
    """获取文档内容"""
    try:
        if not link_manager:
            raise HTTPException(status_code=500, detail="链接管理器未初始化")
        
        doc_info = link_manager.get_document_info(doc_id)
        
        if not doc_info:
            raise HTTPException(status_code=404, detail=f"文档 '{doc_id}' 不存在")
        
        doc_path = doc_info.get('doc_path')
        if not doc_path:
            raise HTTPException(status_code=404, detail="文档路径不存在")
        
        # 处理路径转换 - 将旧路径转换为新路径
        resolved_path = link_manager._resolve_document_path(doc_path)
        
        try:
            with open(resolved_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"content": content}
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"文档文件不存在: {resolved_path}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取文档失败: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档内容失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket 路由
@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """WebSocket 进度推送"""
    await websocket.accept()
    logger.info(f"新的WebSocket连接已建立，当前连接数: {len(active_websocket_connections) + 1}")
    
    # 添加到活跃连接集合
    active_websocket_connections.add(websocket)
    
    try:
        # 发送连接确认消息
        await websocket.send_text(json.dumps({
            "type": "connection_confirmed",
            "message": "WebSocket连接已建立",
            "timestamp": time.time()
        }))
        
        # 保持连接直到断开
        while True:
            try:
                message = await websocket.receive_text()
                # 处理客户端发送的消息
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": time.time()
                        }))
                    elif data.get("type") == "stop_processing":
                        task_id = data.get("task_id")
                        if task_id and task_id in active_tasks:
                            active_tasks[task_id] = "stopped"
                            logger.info(f"任务 {task_id} 被标记为停止")
                            await websocket.send_text(json.dumps({
                                "type": "processing_stopped",
                                "task_id": task_id,
                                "message": "处理已停止",
                                "timestamp": time.time()
                            }))
                        else:
                            logger.warning(f"尝试停止不存在的任务: {task_id}")
                except json.JSONDecodeError:
                    pass
            except WebSocketDisconnect:
                logger.info("WebSocket连接断开")
                break
                
    except Exception as e:
        logger.error(f"WebSocket 错误: {str(e)}")
    finally:
        # 从活跃连接中移除
        active_websocket_connections.discard(websocket)
        logger.info(f"WebSocket连接已移除，当前连接数: {len(active_websocket_connections)}")


# ===== 策略优化监控API端点 =====

@app.get("/strategy/performance")
async def get_strategy_performance(strategy_name: Optional[str] = None, time_window: str = "30d"):
    """获取策略性能数据"""
    try:
        if not strategy_evaluator:
            raise HTTPException(status_code=503, detail="策略评估器未初始化")
        
        if strategy_name:
            # 获取特定策略的性能
            evaluation = strategy_evaluator.evaluate_strategy(strategy_name, time_window)
            return {
                "strategy_name": strategy_name,
                "evaluation": evaluation.__dict__ if hasattr(evaluation, '__dict__') else evaluation,
                "time_window": time_window
            }
        else:
            # 获取所有策略的性能
            evaluations = strategy_evaluator.evaluate_all_strategies(time_window)
            return {
                "all_strategies": {
                    name: eval.__dict__ if hasattr(eval, '__dict__') else eval 
                    for name, eval in evaluations.items()
                },
                "time_window": time_window,
                "total_strategies": len(evaluations)
            }
    
    except Exception as e:
        logger.error(f"获取策略性能失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略性能失败: {str(e)}")


@app.get("/strategy/history")
async def get_strategy_history(limit: int = 100, strategy_name: Optional[str] = None):
    """获取策略执行历史"""
    try:
        if not strategy_history_db:
            raise HTTPException(status_code=503, detail="策略历史数据库未初始化")
        
        history = strategy_history_db.get_execution_history(limit, strategy_name)
        
        return {
            "history": history,
            "total_records": len(history),
            "strategy_filter": strategy_name,
            "limit": limit
        }
    
    except Exception as e:
        logger.error(f"获取策略历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略历史失败: {str(e)}")


@app.get("/strategy/rankings")
async def get_strategy_rankings(content_type: Optional[str] = None, time_window: str = "7d"):
    """获取策略排名"""
    try:
        if not strategy_history_db:
            raise HTTPException(status_code=503, detail="策略历史数据库未初始化")
        
        rankings = strategy_history_db.get_strategy_rankings(content_type, time_window)
        
        return {
            "rankings": rankings,
            "content_type_filter": content_type,
            "time_window": time_window,
            "total_strategies": len(rankings)
        }
    
    except Exception as e:
        logger.error(f"获取策略排名失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略排名失败: {str(e)}")


@app.get("/strategy/analysis/trends")
async def get_performance_trends(time_window: str = "30d"):
    """获取性能趋势分析"""
    try:
        if not history_analyzer:
            raise HTTPException(status_code=503, detail="历史分析器未初始化")
        
        trends = history_analyzer.analyze_strategy_performance_trends(time_window)
        
        return trends
    
    except Exception as e:
        logger.error(f"获取性能趋势失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能趋势失败: {str(e)}")


@app.get("/strategy/analysis/failures")
async def get_failure_analysis(time_window: str = "14d"):
    """获取失败模式分析"""
    try:
        if not history_analyzer:
            raise HTTPException(status_code=503, detail="历史分析器未初始化")
        
        failure_analysis = history_analyzer.identify_failure_patterns(time_window)
        
        return failure_analysis
    
    except Exception as e:
        logger.error(f"获取失败分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败分析失败: {str(e)}")


@app.post("/strategy/recommend")
async def get_strategy_recommendation(content_features: Dict[str, Any]):
    """获取策略推荐"""
    try:
        if not strategy_optimizer:
            raise HTTPException(status_code=503, detail="策略优化器未初始化")
        
        recommendation = strategy_optimizer.select_optimal_strategy(content_features)
        
        return {
            "recommendation": recommendation.__dict__ if hasattr(recommendation, '__dict__') else recommendation,
            "content_features": content_features
        }
    
    except Exception as e:
        logger.error(f"获取策略推荐失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取策略推荐失败: {str(e)}")


@app.get("/strategy/learning/status")
async def get_learning_status():
    """获取学习状态"""
    try:
        if not strategy_learner:
            raise HTTPException(status_code=503, detail="策略学习器未初始化")
        
        status = strategy_learner.get_learning_status()
        
        return status
    
    except Exception as e:
        logger.error(f"获取学习状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取学习状态失败: {str(e)}")


@app.post("/strategy/learning/trigger")
async def trigger_learning_cycle():
    """手动触发学习周期"""
    try:
        if not strategy_learner:
            raise HTTPException(status_code=503, detail="策略学习器未初始化")
        
        report = await strategy_learner.force_learning_cycle()
        
        return {
            "learning_report": report.__dict__ if hasattr(report, '__dict__') else report,
            "message": "学习周期执行完成"
        }
    
    except Exception as e:
        logger.error(f"触发学习周期失败: {e}")
        raise HTTPException(status_code=500, detail=f"触发学习周期失败: {str(e)}")


@app.get("/strategy/database/stats")
async def get_database_stats():
    """获取数据库统计信息"""
    try:
        if not strategy_history_db:
            raise HTTPException(status_code=503, detail="策略历史数据库未初始化")
        
        stats = strategy_history_db.get_database_stats()
        
        return stats
    
    except Exception as e:
        logger.error(f"获取数据库统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取数据库统计失败: {str(e)}")


@app.get("/strategy/compare")
async def compare_strategies(strategies: str, time_window: str = "30d", content_type: Optional[str] = None):
    """比较多个策略的性能"""
    try:
        if not strategy_evaluator:
            raise HTTPException(status_code=503, detail="策略评估器未初始化")
        
        # 解析策略名称列表
        strategy_list = [s.strip() for s in strategies.split(',')]
        
        comparison = strategy_evaluator.compare_strategies(strategy_list, time_window, content_type)
        
        return comparison
    
    except Exception as e:
        logger.error(f"比较策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"比较策略失败: {str(e)}")


# 主函数
def main():
    """启动 API 服务器"""
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()