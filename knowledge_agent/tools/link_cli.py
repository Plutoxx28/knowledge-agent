#!/usr/bin/env python3
"""
链接管理命令行工具
提供扫描、查询、报告等功能
"""

import argparse
import sys
import os
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.link_manager import LinkManager
from utils.link_renderer import LinkRenderer, ConceptGraphGenerator


def scan_command(args):
    """扫描知识库命令"""
    print(f"扫描知识库: {args.knowledge_base}")
    
    manager = LinkManager(args.knowledge_base)
    stats = manager.scan_knowledge_base()
    
    print("扫描完成！")
    print(f"处理文件数: {stats['scanned_files']}")
    print(f"发现概念数: {stats['total_concepts']}")
    print(f"总链接数: {stats['total_links']}")


def report_command(args):
    """生成报告命令"""
    print(f"生成链接报告: {args.knowledge_base}")
    
    manager = LinkManager(args.knowledge_base)
    report = manager.generate_link_report()
    
    print("=== 链接系统报告 ===")
    print(f"文档总数: {report['total_documents']}")
    print(f"概念总数: {report['total_concepts']}")
    print(f"链接总数: {report['total_links']}")
    print(f"已解析链接: {report['resolved_links']}")
    print(f"解析率: {report['resolution_rate']:.1%}")
    print(f"孤立概念数: {report['orphaned_count']}")
    
    if report['orphaned_concepts'] and args.verbose:
        print(f"\n孤立概念（前{min(20, len(report['orphaned_concepts']))}个）:")
        for concept in report['orphaned_concepts'][:20]:
            print(f"  - {concept}")
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n详细报告已保存到: {args.output}")


def concept_command(args):
    """查询概念命令"""
    print(f"查询概念: {args.concept}")
    
    manager = LinkManager(args.knowledge_base)
    links = manager.get_concept_links(args.concept)
    
    if not links:
        print(f"未找到概念 '{args.concept}' 的链接信息")
        return
    
    print(f"找到 {len(links)} 个链接:")
    
    for i, link in enumerate(links, 1):
        print(f"\n{i}. 来源文档: {Path(link.source_doc).name}")
        print(f"   行号: {link.line_number}")
        print(f"   目标: {Path(link.target_doc).name if link.target_doc else '未找到'}")
        print(f"   上下文: {link.context}")
        
        if args.verbose:
            print(f"   创建时间: {link.created_at}")
            print(f"   完整路径: {link.source_doc}")


def render_command(args):
    """渲染文档命令"""
    if not os.path.exists(args.document):
        print(f"文档不存在: {args.document}")
        return
    
    print(f"渲染文档: {args.document}")
    
    manager = LinkManager(args.knowledge_base)
    renderer = LinkRenderer(manager)
    
    if args.format == 'html':
        content = renderer.render_document_with_navigation(args.document)
        output_ext = '.html'
    else:
        with open(args.document, 'r', encoding='utf-8') as f:
            content = f.read()
        content = renderer.render_terminal(content, args.document)
        output_ext = '.txt'
    
    if args.output:
        output_path = args.output
    else:
        doc_name = Path(args.document).stem
        output_path = f"{doc_name}_rendered{output_ext}"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"渲染完成: {output_path}")
    
    if args.format == 'html':
        print(f"可以在浏览器中打开: file://{os.path.abspath(output_path)}")


def graph_command(args):
    """生成概念图谱命令"""
    print(f"生成概念图谱: {args.knowledge_base}")
    
    manager = LinkManager(args.knowledge_base)
    graph_generator = ConceptGraphGenerator(manager)
    
    if args.format == 'html':
        content = graph_generator.generate_graph_html(args.max_concepts)
        output_ext = '.html'
    else:
        # JSON格式
        data = graph_generator.generate_graph_data(args.max_concepts)
        content = json.dumps(data, ensure_ascii=False, indent=2)
        output_ext = '.json'
    
    output_path = args.output or f"concept_graph{output_ext}"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"图谱生成完成: {output_path}")
    print(f"包含 {len(graph_generator.generate_graph_data(args.max_concepts)['nodes'])} 个概念节点")
    
    if args.format == 'html':
        print(f"可以在浏览器中打开: file://{os.path.abspath(output_path)}")


def list_command(args):
    """列出所有概念命令"""
    print(f"列出所有概念: {args.knowledge_base}")
    
    manager = LinkManager(args.knowledge_base)
    concepts = manager.get_all_concepts()
    
    if not concepts:
        print("未找到任何概念")
        return
    
    print(f"找到 {len(concepts)} 个概念:")
    
    # 按引用次数排序
    concepts.sort(key=lambda x: x['reference_count'], reverse=True)
    
    for i, concept in enumerate(concepts[:args.limit], 1):
        status = "✅" if concept['has_target'] else "❌"
        print(f"{i:3d}. {status} {concept['name']} ({concept['reference_count']} 次引用)")
        
        if args.verbose and concept['primary_doc']:
            print(f"     主文档: {Path(concept['primary_doc']).name}")


def check_command(args):
    """检查链接完整性命令"""
    print(f"检查链接完整性: {args.knowledge_base}")
    
    manager = LinkManager(args.knowledge_base)
    
    # 生成报告
    report = manager.generate_link_report()
    
    print("=== 链接完整性检查 ===")
    
    # 检查解析率
    if report['resolution_rate'] < 0.8:
        print(f"链接解析率较低: {report['resolution_rate']:.1%}")
    else:
        print(f"链接解析率良好: {report['resolution_rate']:.1%}")
    
    # 检查孤立概念
    if report['orphaned_count'] > 0:
        print(f"发现 {report['orphaned_count']} 个孤立概念")
        
        if args.fix:
            print("尝试修复孤立概念...")
            # 这里可以实现自动修复逻辑
            # 例如：为孤立概念创建占位符文档
            print("建议：为重要的孤立概念创建专门的文档")
    else:
        print("没有发现孤立概念")
    
    # 检查文件完整性
    broken_links = []
    concepts = manager.get_all_concepts()
    
    for concept in concepts:
        if concept['primary_doc'] and not os.path.exists(concept['primary_doc']):
            broken_links.append(concept)
    
    if broken_links:
        print(f"发现 {len(broken_links)} 个断开的文件链接:")
        for concept in broken_links[:10]:  # 只显示前10个
            print(f"  - {concept['name']} -> {concept['primary_doc']}")
    else:
        print("所有文件链接完整")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="知识库链接管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 扫描知识库
  python link_cli.py scan /path/to/knowledge_base
  
  # 生成报告
  python link_cli.py report /path/to/knowledge_base --output report.json
  
  # 查询概念
  python link_cli.py concept /path/to/knowledge_base "RAG技术"
  
  # 渲染文档
  python link_cli.py render /path/to/knowledge_base document.md --format html
  
  # 生成概念图谱
  python link_cli.py graph /path/to/knowledge_base --output graph.html
  
  # 检查链接完整性
  python link_cli.py check /path/to/knowledge_base --fix
        """
    )
    
    # 全局参数
    parser.add_argument('knowledge_base', help='知识库路径')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # scan 命令
    scan_parser = subparsers.add_parser('scan', help='扫描知识库并更新链接数据库')
    
    # report 命令
    report_parser = subparsers.add_parser('report', help='生成链接分析报告')
    report_parser.add_argument('-o', '--output', help='输出文件路径')
    
    # concept 命令
    concept_parser = subparsers.add_parser('concept', help='查询特定概念的链接信息')
    concept_parser.add_argument('concept', help='概念名称')
    
    # render 命令
    render_parser = subparsers.add_parser('render', help='渲染文档，将概念链接转换为可点击链接')
    render_parser.add_argument('document', help='要渲染的文档路径')
    render_parser.add_argument('-f', '--format', choices=['html', 'terminal'], default='html', help='输出格式')
    render_parser.add_argument('-o', '--output', help='输出文件路径')
    
    # graph 命令
    graph_parser = subparsers.add_parser('graph', help='生成概念图谱')
    graph_parser.add_argument('-f', '--format', choices=['html', 'json'], default='html', help='输出格式')
    graph_parser.add_argument('-o', '--output', help='输出文件路径')
    graph_parser.add_argument('-m', '--max-concepts', type=int, default=100, help='最大概念数量')
    
    # list 命令
    list_parser = subparsers.add_parser('list', help='列出所有概念')
    list_parser.add_argument('-l', '--limit', type=int, default=50, help='显示数量限制')
    
    # check 命令
    check_parser = subparsers.add_parser('check', help='检查链接完整性')
    check_parser.add_argument('--fix', action='store_true', help='尝试自动修复问题')
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查知识库路径
    if not os.path.exists(args.knowledge_base):
        print(f"知识库路径不存在: {args.knowledge_base}")
        sys.exit(1)
    
    # 执行命令
    if args.command == 'scan':
        scan_command(args)
    elif args.command == 'report':
        report_command(args)
    elif args.command == 'concept':
        concept_command(args)
    elif args.command == 'render':
        render_command(args)
    elif args.command == 'graph':
        graph_command(args)
    elif args.command == 'list':
        list_command(args)
    elif args.command == 'check':
        check_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
