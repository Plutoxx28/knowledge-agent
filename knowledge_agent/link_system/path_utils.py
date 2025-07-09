"""
路径工具 - 处理文件路径、验证和转换的工具函数
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


class PathUtils:
    """路径处理工具类"""
    
    @staticmethod
    def resolve_document_path(doc_path: str, knowledge_base_path: Path) -> str:
        """
        解析文档路径，处理旧路径到新路径的转换
        
        Args:
            doc_path: 文档路径（可能是旧格式）
            knowledge_base_path: 知识库根路径
            
        Returns:
            解析后的绝对路径
        """
        try:
            # 如果已经是绝对路径且存在，直接返回
            if os.path.isabs(doc_path) and os.path.exists(doc_path):
                return doc_path
            
            # 旧路径映射 - 处理用户路径变化
            old_patterns = [
                '/Users/pluto/Desktop/knowledgeBase/',
                '/Users/pluto/Desktop/knowledge_base/',
                '/Users/pluto/Desktop/KnowledgeBase/',
                'knowledgeBase/',
                'knowledge_base/',
                'KnowledgeBase/'
            ]
            
            # 移除旧路径前缀
            clean_path = doc_path
            for pattern in old_patterns:
                if clean_path.startswith(pattern):
                    clean_path = clean_path[len(pattern):]
                    break
            
            # 构建新的绝对路径
            resolved_path = knowledge_base_path / clean_path
            
            # 如果路径存在，返回
            if resolved_path.exists():
                return str(resolved_path)
            
            # 尝试在知识库中查找文件
            filename = os.path.basename(clean_path)
            found_path = PathUtils.find_file_in_knowledge_base(filename, knowledge_base_path)
            if found_path:
                logger.info(f"通过文件名查找到文档: {filename} -> {found_path}")
                return found_path
            
            logger.warning(f"无法解析文档路径: {doc_path}")
            return str(resolved_path)  # 返回构建的路径，即使不存在
            
        except Exception as e:
            logger.error(f"解析文档路径失败 {doc_path}: {e}")
            return doc_path
    
    @staticmethod
    def should_process_file(file_path: Path) -> bool:
        """
        判断是否应该处理该文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否应该处理
        """
        # 跳过隐藏文件和目录
        if any(part.startswith('.') for part in file_path.parts):
            return False
        
        # 跳过特殊目录
        skip_dirs = {'node_modules', '__pycache__', '.git', '.vscode', 'temp', 'tmp'}
        if any(part in skip_dirs for part in file_path.parts):
            return False
        
        # 只处理markdown文件
        return file_path.suffix.lower() in ['.md', '.markdown']
    
    @staticmethod
    def get_relative_path(file_path: str, base_path: Path) -> str:
        """
        获取相对于知识库的相对路径
        
        Args:
            file_path: 文件路径
            base_path: 基础路径
            
        Returns:
            相对路径
        """
        try:
            path = Path(file_path)
            if path.is_absolute():
                return str(path.relative_to(base_path))
            return file_path
        except Exception:
            return file_path
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """
        规范化路径格式
        
        Args:
            path: 原始路径
            
        Returns:
            规范化后的路径
        """
        return str(Path(path).resolve())
    
    @staticmethod
    def is_valid_markdown_file(file_path: str) -> bool:
        """
        检查是否为有效的Markdown文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为有效的Markdown文件
        """
        try:
            path = Path(file_path)
            return (path.exists() and 
                   path.is_file() and 
                   path.suffix.lower() in ['.md', '.markdown'])
        except Exception:
            return False
    
    @staticmethod
    def find_file_in_knowledge_base(filename: str, knowledge_base_path: Path) -> Optional[str]:
        """
        在知识库中查找指定文件名的文件
        
        Args:
            filename: 文件名
            knowledge_base_path: 知识库路径
            
        Returns:
            找到的文件路径，未找到返回None
        """
        try:
            for root, dirs, files in os.walk(knowledge_base_path):
                # 跳过隐藏目录
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                if filename in files:
                    return os.path.join(root, filename)
            return None
        except Exception as e:
            logger.error(f"搜索文件失败 {filename}: {e}")
            return None
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """
        计算文件的MD5哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件的MD5哈希值
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败 {file_path}: {e}")
            return ""
    
    @staticmethod
    def create_backup_path(file_path: str) -> str:
        """
        创建备份文件路径
        
        Args:
            file_path: 原文件路径
            
        Returns:
            备份文件路径
        """
        path = Path(file_path)
        backup_name = f"{path.stem}.backup{path.suffix}"
        return str(path.parent / backup_name)
    
    @staticmethod
    def ensure_directory_exists(dir_path: str) -> bool:
        """
        确保目录存在，如果不存在则创建
        
        Args:
            dir_path: 目录路径
            
        Returns:
            是否成功（目录存在或创建成功）
        """
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"创建目录失败 {dir_path}: {e}")
            return False
    
    @staticmethod
    def is_under_knowledge_base(file_path: str, knowledge_base_path: Path) -> bool:
        """
        检查文件是否在知识库路径下
        
        Args:
            file_path: 文件路径
            knowledge_base_path: 知识库路径
            
        Returns:
            是否在知识库路径下
        """
        try:
            file_path_obj = Path(file_path).resolve()
            knowledge_base_resolved = knowledge_base_path.resolve()
            return knowledge_base_resolved in file_path_obj.parents or file_path_obj == knowledge_base_resolved
        except Exception:
            return False
    
    @staticmethod
    def get_file_list(directory: str, pattern: str = "*.md") -> List[str]:
        """
        获取目录下匹配模式的文件列表
        
        Args:
            directory: 目录路径
            pattern: 文件模式
            
        Returns:
            文件路径列表
        """
        try:
            path = Path(directory)
            return [str(f) for f in path.rglob(pattern) if PathUtils.should_process_file(f)]
        except Exception as e:
            logger.error(f"获取文件列表失败 {directory}: {e}")
            return []
    
    @staticmethod
    def safe_join_path(*parts) -> str:
        """
        安全地连接路径部分
        
        Args:
            *parts: 路径部分
            
        Returns:
            连接后的路径
        """
        try:
            return str(Path(*parts))
        except Exception:
            return os.path.join(*parts)