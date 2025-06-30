"""
链接渲染器 - 将Markdown中的[[概念]]转换为可点击的链接
支持HTML和终端两种渲染模式
"""

import re
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from .link_manager import LinkManager


class LinkRenderer:
    """链接渲染器"""
    
    def __init__(self, link_manager: LinkManager):
        self.link_manager = link_manager
        self.concept_pattern = re.compile(r'\[\[([^\]]+)\]\]')
    
    def render_html(self, markdown_content: str, current_doc_path: str = None) -> str:
        """将Markdown中的概念链接渲染为HTML链接
        
        Args:
            markdown_content: 原始Markdown内容
            current_doc_path: 当前文档路径（用于生成相对链接）
            
        Returns:
            str: 渲染后的HTML内容
        """
        def replace_concept_link(match):
            concept_name = match.group(1).strip()
            target_doc = self.link_manager.find_concept_target(concept_name)
            
            if target_doc:
                # 生成相对链接
                if current_doc_path:
                    relative_path = os.path.relpath(target_doc, os.path.dirname(current_doc_path))
                    href = relative_path.replace('\\', '/')  # 确保使用正斜杠
                else:
                    href = target_doc
                
                # 生成HTML链接
                return f'<a href="{href}" class="concept-link" data-concept="{concept_name}" title="跳转到: {concept_name}">{concept_name}</a>'
            else:
                # 未找到目标文档，显示为未链接状态
                return f'<span class="concept-link-missing" data-concept="{concept_name}" title="未找到文档: {concept_name}">{concept_name}</span>'
        
        return self.concept_pattern.sub(replace_concept_link, markdown_content)
    
    def render_terminal(self, markdown_content: str, current_doc_path: str = None) -> str:
        """将Markdown中的概念链接渲染为终端友好格式
        
        Args:
            markdown_content: 原始Markdown内容
            current_doc_path: 当前文档路径
            
        Returns:
            str: 渲染后的内容
        """
        def replace_concept_link(match):
            concept_name = match.group(1).strip()
            target_doc = self.link_manager.find_concept_target(concept_name)
            
            if target_doc:
                # 使用ANSI颜色码高亮可链接的概念
                return f'\033[94m{concept_name}\033[0m'  # 蓝色
            else:
                # 未链接的概念用红色显示
                return f'\033[91m{concept_name}\033[0m'  # 红色
        
        return self.concept_pattern.sub(replace_concept_link, markdown_content)
    
    def get_backlinks_html(self, doc_path: str) -> str:
        """生成文档的反向链接HTML
        
        Args:
            doc_path: 文档路径
            
        Returns:
            str: 反向链接的HTML内容
        """
        links_info = self.link_manager.get_document_links(doc_path)
        inbound_links = links_info.get('inbound', [])
        
        if not inbound_links:
            return ""
        
        html_parts = ['<div class="backlinks">', '<h3>📈 反向链接</h3>', '<ul>']
        
        for source_doc in inbound_links:
            doc_name = Path(source_doc).stem
            relative_path = os.path.relpath(source_doc, os.path.dirname(doc_path))
            html_parts.append(f'<li><a href="{relative_path}">{doc_name}</a></li>')
        
        html_parts.extend(['</ul>', '</div>'])
        return '\n'.join(html_parts)
    
    def get_related_concepts_html(self, doc_path: str) -> str:
        """生成文档的相关概念HTML
        
        Args:
            doc_path: 文档路径
            
        Returns:
            str: 相关概念的HTML内容
        """
        links_info = self.link_manager.get_document_links(doc_path)
        outbound_concepts = links_info.get('outbound', [])
        
        if not outbound_concepts:
            return ""
        
        html_parts = ['<div class="related-concepts">', '<h3>🔗 相关概念</h3>', '<ul>']
        
        for concept in outbound_concepts:
            target_doc = self.link_manager.find_concept_target(concept)
            if target_doc and target_doc != doc_path:
                doc_name = Path(target_doc).stem
                relative_path = os.path.relpath(target_doc, os.path.dirname(doc_path))
                html_parts.append(f'<li><a href="{relative_path}">{concept}</a> → {doc_name}</li>')
            else:
                html_parts.append(f'<li><span class="concept-orphaned">{concept}</span> (未找到文档)</li>')
        
        html_parts.extend(['</ul>', '</div>'])
        return '\n'.join(html_parts)
    
    def render_document_with_navigation(self, doc_path: str) -> str:
        """渲染文档并添加导航信息
        
        Args:
            doc_path: 文档路径
            
        Returns:
            str: 完整的HTML内容
        """
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return f"<p>文档未找到: {doc_path}</p>"
        
        # 渲染概念链接
        rendered_content = self.render_html(content, doc_path)
        
        # 添加导航信息
        backlinks_html = self.get_backlinks_html(doc_path)
        related_concepts_html = self.get_related_concepts_html(doc_path)
        
        # 生成完整HTML
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{Path(doc_path).stem}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .concept-link {{ color: #0066cc; text-decoration: none; border-bottom: 1px dotted #0066cc; }}
        .concept-link:hover {{ background-color: #f0f8ff; }}
        .concept-link-missing {{ color: #cc0000; text-decoration: line-through; }}
        .backlinks, .related-concepts {{ margin-top: 2em; padding: 1em; background-color: #f8f9fa; border-radius: 4px; }}
        .backlinks h3, .related-concepts h3 {{ margin-top: 0; }}
        .concept-orphaned {{ color: #666; font-style: italic; }}
    </style>
</head>
<body>
    <div class="document-content">
        {rendered_content}
    </div>
    {backlinks_html}
    {related_concepts_html}
</body>
</html>
"""
        return full_html


class ConceptGraphGenerator:
    """概念图谱生成器"""
    
    def __init__(self, link_manager: LinkManager):
        self.link_manager = link_manager
    
    def generate_graph_data(self, max_concepts: int = 100) -> Dict:
        """生成概念图谱数据
        
        Args:
            max_concepts: 最大概念数量
            
        Returns:
            Dict: 图谱数据，包含nodes和links
        """
        concepts = self.link_manager.get_all_concepts()
        
        # 限制概念数量，优先选择引用次数多的概念
        concepts = concepts[:max_concepts]
        
        # 生成节点数据
        nodes = []
        concept_to_id = {}
        
        for i, concept in enumerate(concepts):
            concept_to_id[concept['name']] = i
            nodes.append({
                'id': i,
                'label': concept['name'],
                'size': min(10 + concept['reference_count'] * 2, 50),
                'color': '#0066cc' if concept['has_target'] else '#cc0000',
                'title': f"{concept['name']} (引用次数: {concept['reference_count']})"
            })
        
        # 生成边数据
        links = []
        processed_pairs = set()
        
        for concept in concepts:
            concept_links = self.link_manager.get_concept_links(concept['name'])
            
            for link in concept_links:
                if link.target_doc:
                    # 找到目标文档定义的概念
                    target_concepts = self._get_document_concepts(link.target_doc)
                    
                    for target_concept in target_concepts:
                        if target_concept in concept_to_id:
                            source_id = concept_to_id[concept['name']]
                            target_id = concept_to_id[target_concept]
                            
                            # 避免重复边
                            pair = tuple(sorted([source_id, target_id]))
                            if pair not in processed_pairs and source_id != target_id:
                                links.append({
                                    'source': source_id,
                                    'target': target_id,
                                    'weight': 1
                                })
                                processed_pairs.add(pair)
        
        return {
            'nodes': nodes,
            'links': links
        }
    
    def _get_document_concepts(self, doc_path: str) -> List[str]:
        """获取文档定义的概念"""
        import sqlite3
        
        with sqlite3.connect(self.link_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT concept_name FROM concept_documents 
                WHERE doc_path = ? AND is_primary = 1
            ''', (doc_path,))
            
            return [row[0] for row in cursor.fetchall()]
    
    def generate_graph_html(self, max_concepts: int = 100) -> str:
        """生成概念图谱的HTML可视化
        
        Args:
            max_concepts: 最大概念数量
            
        Returns:
            str: HTML内容
        """
        graph_data = self.generate_graph_data(max_concepts)
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>概念图谱</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        #graph { width: 100%; height: 600px; border: 1px solid #ccc; }
        .node { cursor: pointer; }
        .link { stroke: #999; stroke-opacity: 0.6; }
        .node-label { font-size: 12px; text-anchor: middle; }
        #controls { margin-bottom: 10px; }
    </style>
</head>
<body>
    <h1>🕸️ 概念图谱</h1>
    <div id="controls">
        <button onclick="restartSimulation()">重新布局</button>
        <button onclick="centerGraph()">居中显示</button>
    </div>
    <div id="graph"></div>

    <script>
        const data = """ + str(graph_data).replace("'", '"') + """;
        
        const width = 800;
        const height = 600;
        
        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        const link = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.weight));
        
        const node = svg.append("g")
            .selectAll("circle")
            .data(data.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => d.size / 2)
            .attr("fill", d => d.color)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        const label = svg.append("g")
            .selectAll("text")
            .data(data.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .attr("dy", -15)
            .text(d => d.label);
        
        node.append("title")
            .text(d => d.title);
        
        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        });
        
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        function restartSimulation() {
            simulation.alpha(1).restart();
        }
        
        function centerGraph() {
            simulation.force("center", d3.forceCenter(width / 2, height / 2));
            simulation.alpha(1).restart();
        }
    </script>
</body>
</html>
"""
        return html_template


def main():
    """命令行工具入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='链接渲染器')
    parser.add_argument('knowledge_base', help='知识库路径')
    parser.add_argument('--render', help='渲染指定文档')
    parser.add_argument('--graph', action='store_true', help='生成概念图谱')
    parser.add_argument('--output', help='输出文件路径')
    
    args = parser.parse_args()
    
    link_manager = LinkManager(args.knowledge_base)
    renderer = LinkRenderer(link_manager)
    
    if args.render:
        html_content = renderer.render_document_with_navigation(args.render)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"已生成HTML文件: {args.output}")
        else:
            print(html_content)
    
    if args.graph:
        graph_generator = ConceptGraphGenerator(link_manager)
        html_content = graph_generator.generate_graph_html()
        
        output_path = args.output or 'concept_graph.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"已生成概念图谱: {output_path}")


if __name__ == '__main__':
    main()