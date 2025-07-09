"""
链接管理器 - 整合所有链接系统功能的主要接口
"""

import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .data_models import ConceptLink, DocumentMeta
from .database_manager import DatabaseManager
from .content_parser import ContentParser
from .link_resolver import LinkResolver
from .query_service import QueryService
from .path_utils import PathUtils
from .analysis_service import AnalysisService

logger = logging.getLogger(__name__)


class LinkManager:
    """双向链接管理器 - 重构后的主要接口"""
    
    def __init__(self, knowledge_base_path: str, db_path: str = None):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.db_path = db_path or str(self.knowledge_base_path / ".link_manager.db")
        
        # 初始化各个组件
        self.db = DatabaseManager(self.db_path)
        self.parser = ContentParser()
        self.resolver = LinkResolver(self.db, self.parser)
        self.query = QueryService(self.db, PathUtils)
        self.analysis = AnalysisService(self.db, self.query)
        
        logger.info(f"链接管理器初始化完成，知识库路径: {self.knowledge_base_path}")
    
    def scan_knowledge_base(self) -> Dict[str, Any]:
        """扫描知识库，更新所有链接关系"""
        stats = {
            'processed_files': 0,
            'total_concepts': 0,
            'total_links': 0,
            'errors': []
        }
        
        try:
            # 扫描所有Markdown文件
            for file_path in self.knowledge_base_path.rglob("*.md"):
                if PathUtils.should_process_file(file_path):
                    try:
                        success = self._process_document(file_path)
                        if success:
                            stats['processed_files'] += 1
                    except Exception as e:
                        error_msg = f"处理文件 {file_path} 失败: {str(e)}"
                        logger.error(error_msg)
                        stats['errors'].append(error_msg)
            
            # 解析所有链接
            self.resolver.resolve_all_links()
            
            # 更新统计信息
            final_stats = self.query.get_stats()
            stats.update(final_stats)
            
            logger.info(f"知识库扫描完成: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"知识库扫描失败: {e}")
            stats['errors'].append(str(e))
            return stats
    
    def scan_knowledge_base_simple(self) -> Dict[str, Any]:
        """简化的知识库扫描，只更新文档基本信息"""
        stats = {
            'processed_files': 0,
            'skipped_files': 0,
            'errors': []
        }
        
        try:
            for file_path in self.knowledge_base_path.rglob("*.md"):
                if PathUtils.should_process_file(file_path):
                    try:
                        updated = self.db.process_document_simple(file_path, self.parser.extract_title)
                        if updated:
                            stats['processed_files'] += 1
                        else:
                            stats['skipped_files'] += 1
                    except Exception as e:
                        error_msg = f"处理文件 {file_path} 失败: {str(e)}"
                        logger.error(error_msg)
                        stats['errors'].append(error_msg)
            
            logger.info(f"简化扫描完成: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"简化扫描失败: {e}")
            stats['errors'].append(str(e))
            return stats
    
    def _process_document(self, doc_path: Path) -> bool:
        """处理单个文档"""
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 计算文件哈希
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # 检查是否需要更新
            if self.db.is_file_up_to_date(str(doc_path), file_hash):
                logger.debug(f"文件无变化，跳过: {doc_path}")
                return False
            
            # 提取文档信息
            title = self.parser.extract_title(content)
            concept_links = self.parser.extract_concept_links(content, str(doc_path))
            defined_concepts = self.parser.extract_defined_concepts(content)
            
            # 更新数据库
            self.db.update_document_in_db(
                doc_path=str(doc_path),
                title=title,
                defined_concepts=defined_concepts,
                concept_links=concept_links,
                file_hash=file_hash
            )
            
            logger.info(f"处理文档成功: {doc_path} (标题: {title})")
            return True
            
        except Exception as e:
            logger.error(f"处理文档失败 {doc_path}: {e}")
            return False
    
    def update_document_incremental(self, doc_path: str) -> bool:
        """增量更新单个文档"""
        try:
            path = Path(doc_path)
            if not path.exists():
                logger.error(f"文档不存在: {doc_path}")
                return False
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 计算文件哈希
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # 检查是否需要更新
            if self.db.is_file_up_to_date(str(path), file_hash):
                logger.debug(f"文件无变化，跳过更新: {doc_path}")
                return True
            
            # 提取文档信息
            title = self.parser.extract_title(content)
            concept_links = self.parser.extract_concept_links(content, str(path))
            defined_concepts = self.parser.extract_defined_concepts(content)
            
            # 更新数据库
            self.db.update_document_in_db(
                doc_path=str(path),
                title=title,
                defined_concepts=defined_concepts,
                concept_links=concept_links,
                file_hash=file_hash
            )
            
            # 重新解析链接关系
            self.resolver.resolve_all_links()
            
            logger.info(f"更新文档成功: {doc_path} (标题: {title})")
            return True
            
        except Exception as e:
            logger.error(f"更新文档失败 {doc_path}: {e}")
            return False
    
    # 查询接口 - 委托给QueryService
    def get_concept_links(self, concept_name: str) -> List[ConceptLink]:
        """获取概念的链接信息"""
        return self.query.get_concept_links(concept_name)
    
    def get_document_links(self, doc_path: str) -> Dict[str, List[str]]:
        """获取文档的链接信息"""
        return self.query.get_document_links(doc_path)
    
    def get_all_concepts(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """获取所有概念"""
        return self.query.get_all_concepts(limit, offset)
    
    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """获取所有文档"""
        return self.query.get_all_documents(limit, offset, str(self.knowledge_base_path))
    
    def get_concept_info(self, concept_name: str) -> Optional[Dict]:
        """获取概念详细信息"""
        return self.query.get_concept_info(concept_name)
    
    def get_document_info(self, doc_id: str) -> Optional[Dict]:
        """获取文档详细信息"""
        return self.query.get_document_info(doc_id)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.query.get_stats()
    
    # 分析接口 - 委托给AnalysisService
    def generate_link_report(self) -> Dict:
        """生成链接报告"""
        return self.analysis.generate_link_report()
    
    def discover_links_for_document(self, doc_path: str, threshold: float = 0.7) -> List[Dict]:
        """为文档发现链接"""
        return self.query.discover_links_for_document(doc_path, threshold)
    
    # 管理接口
    def remove_document(self, doc_path: str) -> bool:
        """删除文档"""
        return self.db.remove_document(doc_path)
    
    def find_concept_target(self, concept_name: str) -> Optional[str]:
        """查找概念目标"""
        return self.resolver.find_concept_target(concept_name)
    
    # 路径解析
    def _resolve_document_path(self, doc_path: str) -> str:
        """解析文档路径"""
        return PathUtils.resolve_document_path(doc_path, self.knowledge_base_path)
    
    def _estimate_word_count(self, file_path: str) -> int:
        """估算字数"""
        return self.parser.extract_word_count(file_path) if Path(file_path).exists() else 0
    
    # 向后兼容性方法
    def scan_knowledge_base_with_callback(self, callback=None) -> Dict:
        """带回调的知识库扫描（向后兼容）"""
        return self.scan_knowledge_base()
    
    def get_concept_graph_data(self, max_concepts: int = 100) -> Dict:
        """获取概念图谱数据"""
        return self.analysis.get_concept_graph_data(max_concepts)
    
    def generate_comprehensive_report(self) -> Dict:
        """生成综合报告"""
        return self.analysis.generate_comprehensive_report()
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索概念"""
        return self.query.search_concepts(query, limit)
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict]:
        """搜索文档"""
        return self.query.search_documents(query, limit)
    
    def get_related_concepts(self, concept_name: str) -> List[str]:
        """获取相关概念"""
        return self.resolver.discover_related_concepts(concept_name)
    
    def validate_link_integrity(self) -> Dict:
        """验证链接完整性"""
        return self.resolver.validate_link_integrity()


# 向后兼容的导出
def main():
    """命令行工具入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='知识库链接管理器')
    parser.add_argument('knowledge_base', help='知识库路径')
    parser.add_argument('--scan', action='store_true', help='扫描知识库并更新链接')
    parser.add_argument('--report', action='store_true', help='生成链接报告')
    parser.add_argument('--concept', help='查询特定概念的链接信息')
    
    args = parser.parse_args()
    
    manager = LinkManager(args.knowledge_base)
    
    if args.scan:
        print("扫描知识库中...")
        stats = manager.scan_knowledge_base()
        print(f"扫描完成: {stats}")
    
    if args.report:
        report = manager.generate_link_report()
        print("=== 链接系统报告 ===")
        print(f"文档总数: {report['total_documents']}")
        print(f"概念总数: {report['total_concepts']}")
        print(f"链接总数: {report['total_links']}")
        print(f"已解析链接: {report['resolved_links']}")
        print(f"解析率: {report['resolution_rate']:.1%}")
        print(f"孤立概念数: {report['orphaned_count']}")
        
        if report['orphaned_concepts']:
            print("\n孤立概念（前10个）:")
            for concept in report['orphaned_concepts'][:10]:
                print(f"  - {concept}")
    
    if args.concept:
        links = manager.get_concept_links(args.concept)
        print(f"=== 概念 '{args.concept}' 的链接信息 ===")
        for link in links:
            print(f"来源: {link.source_doc}:{link.line_number}")
            print(f"目标: {link.target_doc or '未找到'}")
            print(f"上下文: {link.context}")
            print()


if __name__ == '__main__':
    main()