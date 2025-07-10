"""
进度跟踪器 - 提供统一的进度管理和WebSocket广播功能
"""
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)


class SimpleProgressTracker:
    """简化的进度跟踪器"""
    
    def __init__(self, websocket_broadcast_func: Optional[Callable] = None):
        self.websocket_broadcast = websocket_broadcast_func
        self.task_id = str(uuid.uuid4())
        self.stage_weights = {
            "planning": 10,
            "tool_creation": 5,
            "analysis": 20,
            "extraction": 25,
            "enhancement": 15,
            "quality_control": 10,
            "synthesis": 5
        }
        self.completed_progress = 0
        
    def _calculate_stage_progress(self, stage: str, stage_progress: float = 1.0) -> int:
        """计算阶段进度的整体百分比"""
        stage_weight = self.stage_weights.get(stage, 10)
        stage_contribution = stage_weight * stage_progress
        total_progress = self.completed_progress + stage_contribution
        return min(100, int(total_progress))
        
    def _complete_stage(self, stage: str):
        """标记阶段完成，更新已完成进度"""
        stage_weight = self.stage_weights.get(stage, 10)
        self.completed_progress += stage_weight
        self.completed_progress = min(100, self.completed_progress)
        
    async def update_progress(self, stage: str, message: str, progress_percent: int = None, workers: List[str] = None, stage_progress: float = None):
        """更新进度"""
        # 计算实际进度百分比
        if progress_percent is not None:
            # 使用指定的进度百分比（仅用于外部直接设置）
            actual_progress = progress_percent
        elif stage_progress is not None:
            # 使用阶段进度计算（推荐方式）
            actual_progress = self._calculate_stage_progress(stage, stage_progress)
        else:
            # 默认使用阶段完成度计算
            actual_progress = self._calculate_stage_progress(stage, 1.0)
            
        # 如果是completed阶段，确保设置为100%
        if stage == "completed":
            actual_progress = 100
            # 确保不会超过100%
            self.completed_progress = 100
        
        progress_data = {
            "task_id": self.task_id,
            "stage": stage,
            "current_step": message,
            "progress_percent": actual_progress,
            "workers": workers or [],
            "timestamp": time.time()
        }
        
        logger.info(f"进度更新: {stage} - {message} ({actual_progress}%)")
        
        if self.websocket_broadcast:
            try:
                await self.websocket_broadcast(progress_data)
                logger.info(f"WebSocket进度广播成功")
            except Exception as e:
                logger.error(f"WebSocket进度广播失败: {e}")
        else:
            logger.info("没有WebSocket广播函数，跳过进度广播")