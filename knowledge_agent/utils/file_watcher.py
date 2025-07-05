"""
文件监控器 - 监控知识库文档变化并自动更新
"""
import os
import time
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import threading
import hashlib

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class KnowledgeBaseWatcher(FileSystemEventHandler):
    """知识库文件监控器"""
    
    def __init__(self, link_manager, callback: Optional[Callable] = None, 
                 debounce_interval: float = 2.0):
        """
        初始化文件监控器
        
        Args:
            link_manager: LinkManager实例
            callback: 变化通知回调函数
            debounce_interval: 防抖动间隔（秒）
        """
        self.link_manager = link_manager
        self.callback = callback
        self.debounce_interval = debounce_interval
        
        # 防抖动机制 - 存储待处理的文件
        self.pending_files = {}
        self.debounce_timer = None
        self.lock = threading.Lock()
        
        logger.info("文件监控器初始化完成")
    
    def on_modified(self, event):
        """文件修改事件"""
        if not event.is_directory and self._is_markdown_file(event.src_path):
            self._schedule_update(event.src_path, 'modified')
    
    def on_created(self, event):
        """文件创建事件"""
        if not event.is_directory and self._is_markdown_file(event.src_path):
            self._schedule_update(event.src_path, 'created')
    
    def on_deleted(self, event):
        """文件删除事件"""
        if not event.is_directory and self._is_markdown_file(event.src_path):
            self._schedule_update(event.src_path, 'deleted')
    
    def on_moved(self, event):
        """文件移动事件"""
        if not event.is_directory:
            # 处理源文件删除
            if self._is_markdown_file(event.src_path):
                self._schedule_update(event.src_path, 'deleted')
            
            # 处理目标文件创建
            if self._is_markdown_file(event.dest_path):
                self._schedule_update(event.dest_path, 'created')
    
    def _is_markdown_file(self, file_path: str) -> bool:
        """检查是否为Markdown文件"""
        return file_path.lower().endswith('.md')
    
    def _schedule_update(self, file_path: str, event_type: str):
        """调度文件更新（带防抖动）"""
        with self.lock:
            # 记录待处理的文件
            self.pending_files[file_path] = {
                'event_type': event_type,
                'timestamp': time.time()
            }
            
            # 取消之前的定时器
            if self.debounce_timer:
                self.debounce_timer.cancel()
            
            # 启动新的定时器
            self.debounce_timer = threading.Timer(
                self.debounce_interval, 
                self._process_pending_updates
            )
            self.debounce_timer.start()
            
            logger.debug(f"调度文件更新: {file_path} ({event_type})")
    
    def _process_pending_updates(self):
        """处理所有待更新的文件"""
        with self.lock:
            if not self.pending_files:
                return
            
            files_to_process = dict(self.pending_files)
            self.pending_files.clear()
        
        logger.info(f"开始处理 {len(files_to_process)} 个文件的更新")
        
        updated_count = 0
        for file_path, file_info in files_to_process.items():
            try:
                success = self._handle_file_change(file_path, file_info['event_type'])
                if success:
                    updated_count += 1
            except Exception as e:
                logger.error(f"处理文件 {file_path} 失败: {e}")
        
        logger.info(f"完成文件更新，成功处理 {updated_count}/{len(files_to_process)} 个文件")
        
        # 通知前端
        if self.callback:
            try:
                self.callback({
                    'type': 'files_updated',
                    'updated_count': updated_count,
                    'total_count': len(files_to_process),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"通知回调失败: {e}")
    
    def _handle_file_change(self, file_path: str, event_type: str) -> bool:
        """处理单个文件的变化"""
        try:
            file_path_obj = Path(file_path)
            
            if event_type == 'deleted':
                # 删除文档
                success = self.link_manager.remove_document(str(file_path_obj))
                logger.info(f"删除文档: {file_path} - {'成功' if success else '失败'}")
                return success
            else:
                # 检查文件是否存在
                if not file_path_obj.exists():
                    logger.warning(f"文件不存在: {file_path}")
                    return False
                
                # 更新或创建文档
                success = self.link_manager.update_document_incremental(file_path_obj)
                logger.info(f"更新文档: {file_path} - {'成功' if success else '失败'}")
                return success
                
        except Exception as e:
            logger.error(f"处理文件变化失败 {file_path}: {e}")
            return False


class FileWatcherManager:
    """文件监控管理器"""
    
    def __init__(self, knowledge_base_path: str, link_manager, 
                 callback: Optional[Callable] = None):
        """
        初始化监控管理器
        
        Args:
            knowledge_base_path: 知识库路径
            link_manager: LinkManager实例
            callback: 变化通知回调函数
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.link_manager = link_manager
        self.callback = callback
        
        # 创建监控器实例
        self.watcher = KnowledgeBaseWatcher(
            link_manager=link_manager,
            callback=callback
        )
        
        # 创建观察者
        self.observer = Observer()
        self.is_running = False
        
        logger.info(f"文件监控管理器初始化完成，监控路径: {self.knowledge_base_path}")
    
    def start(self):
        """启动文件监控"""
        if self.is_running:
            logger.warning("文件监控已在运行")
            return
        
        try:
            # 确保监控路径存在
            if not self.knowledge_base_path.exists():
                logger.error(f"监控路径不存在: {self.knowledge_base_path}")
                return
            
            # 启动监控
            self.observer.schedule(
                self.watcher,
                str(self.knowledge_base_path),
                recursive=True
            )
            
            self.observer.start()
            self.is_running = True
            
            logger.info(f"文件监控已启动，监控路径: {self.knowledge_base_path}")
            
        except Exception as e:
            logger.error(f"启动文件监控失败: {e}")
            raise
    
    def stop(self):
        """停止文件监控"""
        if not self.is_running:
            return
        
        try:
            self.observer.stop()
            self.observer.join(timeout=10)  # 最多等待10秒
            self.is_running = False
            
            logger.info("文件监控已停止")
            
        except Exception as e:
            logger.error(f"停止文件监控失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            'is_running': self.is_running,
            'knowledge_base_path': str(self.knowledge_base_path),
            'observer_alive': self.observer.is_alive() if self.observer else False,
            'monitored_path_exists': self.knowledge_base_path.exists()
        }


def create_file_watcher(knowledge_base_path: str, link_manager, 
                       callback: Optional[Callable] = None) -> FileWatcherManager:
    """
    创建文件监控器的便捷函数
    
    Args:
        knowledge_base_path: 知识库路径
        link_manager: LinkManager实例
        callback: 变化通知回调函数
        
    Returns:
        FileWatcherManager实例
    """
    return FileWatcherManager(
        knowledge_base_path=knowledge_base_path,
        link_manager=link_manager,
        callback=callback
    ) 