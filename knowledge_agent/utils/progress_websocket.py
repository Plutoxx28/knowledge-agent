"""
WebSocket进度推送服务器 - 实时推送处理进度信息
"""
import asyncio
import websockets
import json
import logging
from typing import Dict, Set, Optional, Any
from dataclasses import asdict
import uuid
from threading import Thread
import time

logger = logging.getLogger(__name__)

class ProgressWebSocketServer:
    """WebSocket进度推送服务器"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.connections: Set[websockets.WebSocketServerProtocol] = set()  # 为了兼容性
        self.task_progress: Dict[str, Dict[str, Any]] = {}
        self.server = None
        self.running = False
        
    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """注册新的客户端连接"""
        self.clients.add(websocket)
        self.connections.add(websocket)
        client_id = id(websocket)
        logger.info(f"客户端 {client_id} 已连接")
        
        # 发送当前所有任务的进度信息
        if self.task_progress:
            for task_id, progress in self.task_progress.items():
                await self.send_to_client(websocket, {
                    "type": "progress_update",
                    "data": progress
                })
    
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """注销客户端连接"""
        self.clients.discard(websocket)
        self.connections.discard(websocket)
        client_id = id(websocket)
        logger.info(f"客户端 {client_id} 已断开")
    
    async def add_connection(self, websocket):
        """添加WebSocket连接（FastAPI兼容方法）"""
        await self.register_client(websocket)
        self.connections.add(websocket)
    
    async def remove_connection(self, websocket):
        """移除WebSocket连接（FastAPI兼容方法）"""
        await self.unregister_client(websocket)
    
    async def broadcast(self, message):
        """广播消息到所有连接（兼容方法）"""
        await self.broadcast_progress(message)
    
    async def send_to_client(self, websocket: websockets.WebSocketServerProtocol, message: Dict[str, Any]):
        """向单个客户端发送消息"""
        try:
            await websocket.send(json.dumps(message, ensure_ascii=False))
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_client(websocket)
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    
    async def broadcast_progress(self, progress_data: Dict[str, Any]):
        """广播进度更新到所有客户端"""
        if not self.clients:
            logger.warning("没有连接的客户端，跳过进度广播")
            return
            
        task_id = progress_data.get("task_id")
        if task_id:
            self.task_progress[task_id] = progress_data
        
        message = {
            "type": "progress_update",
            "data": progress_data,
            "timestamp": time.time()
        }
        
        logger.info(f"广播进度更新给 {len(self.clients)} 个客户端: {progress_data.get('current_step', 'unknown')}")
        
        # 并发发送给所有客户端
        results = await asyncio.gather(
            *[self.send_to_client(client, message) for client in self.clients.copy()],
            return_exceptions=True
        )
        
        # 检查发送结果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"发送给客户端 {i} 失败: {result}")
        
        logger.info(f"进度更新已广播给 {len(self.clients)} 个客户端")
    
    async def handle_client_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe_task":
                task_id = data.get("task_id")
                if task_id and task_id in self.task_progress:
                    await self.send_to_client(websocket, {
                        "type": "progress_update",
                        "data": self.task_progress[task_id]
                    })
            
            elif message_type == "get_all_tasks":
                await self.send_to_client(websocket, {
                    "type": "all_tasks",
                    "data": self.task_progress
                })
            
            elif message_type == "ping":
                await self.send_to_client(websocket, {
                    "type": "pong",
                    "timestamp": time.time()
                })
                
        except json.JSONDecodeError:
            logger.warning(f"收到无效JSON消息: {message}")
        except Exception as e:
            logger.error(f"处理客户端消息失败: {e}")
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """处理客户端连接"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"客户端处理错误: {e}")
        finally:
            await self.unregister_client(websocket)
    
    async def start_server(self):
        """启动WebSocket服务器"""
        self.running = True
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        )
        logger.info(f"WebSocket服务器已启动: ws://{self.host}:{self.port}")
    
    async def stop_server(self):
        """停止WebSocket服务器"""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket服务器已停止")
    
    def start_in_thread(self):
        """在新线程中启动服务器"""
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.start_server())
                loop.run_forever()
            except Exception as e:
                logger.error(f"WebSocket服务器线程错误: {e}")
            finally:
                loop.close()
        
        thread = Thread(target=run_server, daemon=True)
        thread.start()
        logger.info("WebSocket服务器已在后台线程启动")
        return thread

# 全局WebSocket服务器实例
_global_ws_server: Optional[ProgressWebSocketServer] = None

def get_global_websocket_server() -> ProgressWebSocketServer:
    """获取全局WebSocket服务器实例"""
    global _global_ws_server
    if _global_ws_server is None:
        _global_ws_server = ProgressWebSocketServer()
    return _global_ws_server

def start_global_websocket_server(host: str = "localhost", port: int = 8765):
    """启动全局WebSocket服务器"""
    server = get_global_websocket_server()
    server.host = host
    server.port = port
    return server.start_in_thread()

async def broadcast_progress_update(progress_data: Dict[str, Any]):
    """广播进度更新（异步版本）"""
    server = get_global_websocket_server()
    if server.running:
        await server.broadcast_progress(progress_data)

def broadcast_progress_sync(progress_data: Dict[str, Any]):
    """广播进度更新（同步版本）"""
    server = get_global_websocket_server()
    if server.running and server.clients:
        # 在事件循环中运行
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行，创建任务
                asyncio.create_task(server.broadcast_progress(progress_data))
            else:
                # 如果事件循环未运行，直接运行
                loop.run_until_complete(server.broadcast_progress(progress_data))
        except RuntimeError:
            # 没有事件循环，创建新的
            asyncio.run(server.broadcast_progress(progress_data))

class ProgressBroadcaster:
    """进度广播器 - 与KnowledgeOrchestrator集成"""
    
    def __init__(self, websocket_server: Optional[ProgressWebSocketServer] = None):
        self.ws_server = websocket_server or get_global_websocket_server()
    
    def __call__(self, progress):
        """进度回调函数"""
        try:
            progress_data = progress.to_dict()
            
            # 记录进度数据用于调试
            logger.info(f"[{progress.task_id[:8]}] 发送进度更新: {progress.current_step} "
                      f"({progress.completed_steps}/{progress.total_steps}) - 客户端数量: {len(self.ws_server.clients)}")
            
            # 同步广播进度
            broadcast_progress_sync(progress_data)
            
            # 记录关键进度点
            if progress.stage.value in ["analyzing", "generating_workers", "completed"]:
                logger.info(f"[{progress.task_id[:8]}] {progress.current_step} "
                          f"({progress.completed_steps}/{progress.total_steps})")
            
        except Exception as e:
            logger.error(f"广播进度失败: {e}")

def create_progress_callback() -> ProgressBroadcaster:
    """创建进度回调函数"""
    return ProgressBroadcaster()

if __name__ == "__main__":
    # 启动服务器
    start_global_websocket_server()
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        print("服务器已停止")