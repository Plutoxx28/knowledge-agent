#!/usr/bin/env python3
"""
AI工具编排系统 - 智能动态工具组合的知识处理器
"""
import asyncio
import json
import logging
import time
import uuid
import re
from typing import Dict, Any, List, Optional, Callable, Union
from openai import OpenAI
from config import settings

# 导入核心模块
from core.progress_tracker import SimpleProgressTracker
from core.knowledge_processor import SimpleKnowledgeProcessor
from core.ai_orchestrator import AIToolOrchestrator

logger = logging.getLogger(__name__)

# 停止检查函数
def check_should_stop(task_id: str):
    """检查任务是否应该停止"""
    if not task_id:
        return
    
    try:
        # 导入api_server的active_tasks
        from api_server import active_tasks
        if task_id in active_tasks and active_tasks[task_id] == "stopped":
            from api_server import ProcessingStoppedException
            raise ProcessingStoppedException(f"任务 {task_id} 被用户停止")
    except ImportError:
        # 如果无法导入，跳过检查
        pass

# === 便捷入口函数 ===

async def process_with_ai_orchestration(content: str, 
                                      content_type: str = "auto",
                                      metadata: Dict[str, Any] = None,
                                      websocket_broadcast_func: Optional[Callable] = None) -> Dict[str, Any]:
    """便捷的AI编排处理入口函数"""
    orchestrator = AIToolOrchestrator(websocket_broadcast_func)
    return await orchestrator.process_content_with_orchestration(content, content_type, metadata)


async def process_with_traditional_method(content: str,
                                        content_type: str = "text", 
                                        metadata: Dict[str, Any] = None,
                                        websocket_broadcast_func: Optional[Callable] = None) -> Dict[str, Any]:
    """传统处理方法入口函数"""
    processor = SimpleKnowledgeProcessor(websocket_broadcast_func)
    return await processor.process_content(content, content_type, metadata)


async def process_content_smart(content: str,
                              content_type: str = "auto",
                              metadata: Dict[str, Any] = None, 
                              enable_ai_orchestration: bool = True,
                              websocket_broadcast_func: Optional[Callable] = None) -> Dict[str, Any]:
    """智能处理入口 - 自动选择最佳处理方式"""
    
    if enable_ai_orchestration:
        try:
            # 尝试AI编排
            return await process_with_ai_orchestration(content, content_type, metadata, websocket_broadcast_func)
        except Exception as e:
            logger.warning(f"AI编排失败，降级到传统方法: {e}")
            # 降级到传统方法
            return await process_with_traditional_method(content, content_type, metadata, websocket_broadcast_func)
    else:
        # 直接使用传统方法
        return await process_with_traditional_method(content, content_type, metadata, websocket_broadcast_func)