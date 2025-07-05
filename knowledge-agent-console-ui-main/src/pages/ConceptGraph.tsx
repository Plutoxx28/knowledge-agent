import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ZoomIn, Download, RefreshCw, GitBranch, Target, FileText, ChevronRight, Settings, Search, Maximize2 } from 'lucide-react';
import { Input } from '@/components/ui/input';
interface GraphNode {
  id: string;
  label: string;
  type: 'concept' | 'document';
  size: number;
  color: string;
  x?: number;
  y?: number;
  metadata: {
    referenceCount: number;
    hasDocument: boolean;
    category?: string;
  };
}
interface GraphLink {
  source: string;
  target: string;
  weight: number;
  type: 'concept-link' | 'document-link';
}
const mockNodes: GraphNode[] = [{
  id: 'ai',
  label: '人工智能',
  type: 'concept',
  size: 20,
  color: '#3b82f6',
  metadata: {
    referenceCount: 25,
    hasDocument: true,
    category: 'AI理论'
  }
}, {
  id: 'ml',
  label: '机器学习',
  type: 'concept',
  size: 18,
  color: '#3b82f6',
  metadata: {
    referenceCount: 20,
    hasDocument: true,
    category: 'AI理论'
  }
}, {
  id: 'dl',
  label: '深度学习',
  type: 'concept',
  size: 16,
  color: '#3b82f6',
  metadata: {
    referenceCount: 18,
    hasDocument: true,
    category: 'AI理论'
  }
}, {
  id: 'nlp',
  label: '自然语言处理',
  type: 'concept',
  size: 15,
  color: '#3b82f6',
  metadata: {
    referenceCount: 15,
    hasDocument: true,
    category: 'NLP'
  }
}, {
  id: 'cv',
  label: '计算机视觉',
  type: 'concept',
  size: 14,
  color: '#3b82f6',
  metadata: {
    referenceCount: 12,
    hasDocument: false,
    category: 'CV'
  }
}, {
  id: 'doc1',
  label: 'AI技术发展趋势分析',
  type: 'document',
  size: 12,
  color: '#10b981',
  metadata: {
    referenceCount: 8,
    hasDocument: true
  }
}, {
  id: 'doc2',
  label: '机器学习算法对比',
  type: 'document',
  size: 10,
  color: '#10b981',
  metadata: {
    referenceCount: 6,
    hasDocument: true
  }
}];
const mockLinks: GraphLink[] = [{
  source: 'ai',
  target: 'ml',
  weight: 0.9,
  type: 'concept-link'
}, {
  source: 'ml',
  target: 'dl',
  weight: 0.8,
  type: 'concept-link'
}, {
  source: 'ai',
  target: 'nlp',
  weight: 0.7,
  type: 'concept-link'
}, {
  source: 'ai',
  target: 'cv',
  weight: 0.6,
  type: 'concept-link'
}, {
  source: 'doc1',
  target: 'ai',
  weight: 0.9,
  type: 'document-link'
}, {
  source: 'doc1',
  target: 'ml',
  weight: 0.7,
  type: 'document-link'
}, {
  source: 'doc2',
  target: 'ml',
  weight: 0.8,
  type: 'document-link'
}, {
  source: 'doc2',
  target: 'dl',
  weight: 0.6,
  type: 'document-link'
}];
export default function ConceptGraph() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [links, setLinks] = useState<GraphLink[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [layoutAlgorithm, setLayoutAlgorithm] = useState('force');
  const [nodeFilter, setNodeFilter] = useState('all');
  const [maxNodes, setMaxNodes] = useState([50]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [nodePositions, setNodePositions] = useState<{
    [key: string]: {
      x: number;
      y: number;
    };
  }>({});
  const selectedNodeData = selectedNode ? nodes.find(n => n.id === selectedNode) : null;

  // 从后端获取概念图数据
  const fetchConceptGraphData = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:8000/concept-graph?max_concepts=${maxNodes[0]}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // 转换后端数据格式为前端需要的格式
      const transformedNodes: GraphNode[] = data.nodes.map((node: any) => ({
        id: node.id,
        label: node.label,
        type: node.type as 'concept' | 'document',
        size: Math.max(8, Math.min(node.size || 10, 30)), // 限制节点大小范围
        color: node.color || '#3b82f6',
        metadata: {
          referenceCount: node.metadata?.referenceCount || 0,
          hasDocument: node.metadata?.hasDocument || false,
          category: node.metadata?.category || 'concept'
        }
      }));
      
      const transformedLinks: GraphLink[] = data.links.map((link: any) => ({
        source: link.source,
        target: link.target,
        weight: link.weight || 1,
        type: link.type as 'concept-link' | 'document-link'
      }));
      
      setNodes(transformedNodes);
      setLinks(transformedLinks);
      
    } catch (error) {
      console.error('获取概念图数据失败:', error);
      // 如果获取失败，使用备用的mock数据
      setNodes(mockNodes);
      setLinks(mockLinks);
    } finally {
      setLoading(false);
    }
  };

  // 初始加载数据
  useEffect(() => {
    fetchConceptGraphData();
  }, []);

  // 当maxNodes变化时重新获取数据
  useEffect(() => {
    if (!loading) {
      fetchConceptGraphData();
    }
  }, [maxNodes]);

  // 初始化和更新节点位置（只在必要时重新计算）
  useEffect(() => {
    if (Object.keys(nodePositions).length === 0 && nodes.length > 0) {
      calculateNodePositions();
    }
  }, [nodes, links, layoutAlgorithm, nodeFilter, maxNodes]);

  // 渲染图谱（分离位置计算和渲染逻辑）
  useEffect(() => {
    if (Object.keys(nodePositions).length > 0) {
      renderGraph();
    }
  }, [nodePositions, selectedNode, hoveredNode]);
  const calculateNodePositions = () => {
    // 过滤节点
    let filteredNodes = nodes;
    if (nodeFilter === 'concept') {
      filteredNodes = nodes.filter(n => n.type === 'concept');
    } else if (nodeFilter === 'document') {
      filteredNodes = nodes.filter(n => n.type === 'document');
    }

    // 限制节点数量
    filteredNodes = filteredNodes.slice(0, maxNodes[0]);

    // 计算位置（只计算一次，避免重复计算导致跳动）
    const width = 800;
    const height = 600;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.3;
    const newPositions: {
      [key: string]: {
        x: number;
        y: number;
      };
    } = {};
    filteredNodes.forEach((node, index) => {
      const angle = index / filteredNodes.length * 2 * Math.PI;
      newPositions[node.id] = {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius
      };
    });
    setNodePositions(newPositions);
  };
  const renderGraph = () => {
    if (!svgRef.current) return;
    const svg = svgRef.current;

    // 清空 SVG
    svg.innerHTML = '';

    // 过滤节点
    let filteredNodes = nodes;
    if (nodeFilter === 'concept') {
      filteredNodes = nodes.filter(n => n.type === 'concept');
    } else if (nodeFilter === 'document') {
      filteredNodes = nodes.filter(n => n.type === 'document');
    }
    filteredNodes = filteredNodes.slice(0, maxNodes[0]);

    // 绘制连接线
    links.forEach(link => {
      const sourcePos = nodePositions[link.source];
      const targetPos = nodePositions[link.target];
      const sourceNode = filteredNodes.find(n => n.id === link.source);
      const targetNode = filteredNodes.find(n => n.id === link.target);
      if (sourcePos && targetPos && sourceNode && targetNode) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', sourcePos.x.toString());
        line.setAttribute('y1', sourcePos.y.toString());
        line.setAttribute('x2', targetPos.x.toString());
        line.setAttribute('y2', targetPos.y.toString());
        line.setAttribute('stroke', '#6b7280');
        line.setAttribute('stroke-width', (link.weight * 2).toString());
        line.setAttribute('opacity', '0.6');
        svg.appendChild(line);
      }
    });

    // 绘制节点
    filteredNodes.forEach(node => {
      const pos = nodePositions[node.id];
      if (!pos) return;
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', pos.x.toString());
      circle.setAttribute('cy', pos.y.toString());
      circle.setAttribute('r', (node.size / 2).toString());
      circle.setAttribute('fill', node.color);

      // 设置边框样式（根据状态）
      if (selectedNode === node.id) {
        circle.setAttribute('stroke', '#1d4ed8');
        circle.setAttribute('stroke-width', '3');
      } else if (hoveredNode === node.id) {
        circle.setAttribute('stroke', '#3b82f6');
        circle.setAttribute('stroke-width', '2');
        circle.setAttribute('opacity', '0.8');
      } else {
        circle.setAttribute('stroke', 'white');
        circle.setAttribute('stroke-width', '2');
      }
      circle.setAttribute('cursor', 'pointer');
      circle.style.transition = 'opacity 0.2s ease, stroke 0.2s ease';

      // 事件处理
      circle.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        setSelectedNode(node.id);
      });
      circle.addEventListener('mouseenter', e => {
        e.preventDefault();
        setHoveredNode(node.id);
      });
      circle.addEventListener('mouseleave', e => {
        e.preventDefault();
        setHoveredNode(null);
      });
      svg.appendChild(circle);

      // 添加标签
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', pos.x.toString());
      text.setAttribute('y', (pos.y + node.size / 2 + 16).toString());
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('font-size', '12');
      text.setAttribute('fill', '#374151');
      text.setAttribute('pointer-events', 'none');
      text.textContent = node.label.length > 10 ? node.label.slice(0, 10) + '...' : node.label;
      svg.appendChild(text);
    });
  };
  const resetZoom = () => {
    setNodePositions({});
  };
  const refreshGraph = () => {
    setLoading(true);
    // 重新从后端获取数据
    fetchConceptGraphData();
  };
  const exportGraph = () => {
    console.log('导出图谱');
  };
  const getNodeIcon = (type: string) => {
    return type === 'concept' ? <GitBranch className="h-4 w-4" /> : <FileText className="h-4 w-4" />;
  };
  const getNodeLinks = (nodeId: string) => {
    return links.filter(link => link.source === nodeId || link.target === nodeId).map(link => ({
      id: link.source === nodeId ? link.target : link.source,
      target: link.source === nodeId ? link.target : link.source,
      targetLabel: nodes.find(n => n.id === (link.source === nodeId ? link.target : link.source))?.label || ''
    }));
  };
  return <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2 text-left">概念图谱</h1>
        <p className="text-lg text-gray-600 text-left my-[15px]">可视化探索知识概念之间的关系</p>
      </div>


      {/* Graph Container */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
            <CardContent className="p-0">
              <div className="relative">
                <svg ref={svgRef} className="w-full h-[600px] border-b rounded-t-lg bg-gradient-to-br from-blue-50 to-indigo-50" viewBox="0 0 800 600" style={{
                userSelect: 'none'
              }} />

                {/* Legend */}
                <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm border rounded-lg p-3 shadow-sm">
                  <h4 className="text-sm font-medium mb-2">图例</h4>
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                      <span>概念节点</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-green-500"></div>
                      <span>文档节点</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-0.5 bg-gray-400"></div>
                      <span>关联链接</span>
                    </div>
                  </div>
                </div>

                {/* Loading Overlay */}
                {loading && <div className="absolute inset-0 bg-white/80 flex items-center justify-center rounded-lg">
                    <div className="flex items-center gap-2">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                      <span>正在生成图谱...</span>
                    </div>
                  </div>}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Node Details Panel */}
        <div className="space-y-6">
          {selectedNodeData ? <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {getNodeIcon(selectedNodeData.type)}
                  {selectedNodeData.label}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-sm text-gray-600">类型</Label>
                    <div className="font-medium">
                      {selectedNodeData.type === 'concept' ? '概念' : '文档'}
                    </div>
                  </div>
                  <div>
                    <Label className="text-sm text-gray-600">引用次数</Label>
                    <div className="font-medium">{selectedNodeData.metadata.referenceCount}</div>
                  </div>
                  {selectedNodeData.type === 'concept' && <>
                      <div>
                        <Label className="text-sm text-gray-600">是否有文档</Label>
                        <div>
                          {selectedNodeData.metadata.hasDocument ? <Badge className="bg-green-100 text-green-800">有</Badge> : <Badge className="bg-red-100 text-red-800">无</Badge>}
                        </div>
                      </div>
                      {selectedNodeData.metadata.category && <div>
                          <Label className="text-sm text-gray-600">分类</Label>
                          <div className="font-medium">{selectedNodeData.metadata.category}</div>
                        </div>}
                    </>}
                </div>

                {/* Related Links */}
                <div>
                  <Label className="text-sm text-gray-600 mb-2 block">相关链接</Label>
                  <div className="space-y-1 max-h-32 overflow-y-auto">
                    {getNodeLinks(selectedNode).map(link => <div key={link.id} className="flex items-center gap-2 text-sm">
                        <ChevronRight className="w-3 h-3 text-gray-400" />
                        <span className="cursor-pointer hover:text-blue-600 transition-colors" onClick={() => setSelectedNode(link.target)}>
                          {link.targetLabel}
                        </span>
                      </div>)}
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2 pt-4 border-t">
                  {selectedNodeData.type === 'concept' && selectedNodeData.metadata.hasDocument && <Button size="sm" className="flex-1">
                      <FileText className="w-4 h-4 mr-1" />
                      查看文档
                    </Button>}
                  <Button size="sm" variant="outline" className="flex-1">
                    <Target className="w-4 h-4 mr-1" />
                    聚焦
                  </Button>
                </div>
              </CardContent>
            </Card> : <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardContent className="p-8 text-center">
                <GitBranch className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">选择一个节点</h3>
                <p className="text-gray-600">点击图谱中的节点查看详细信息</p>
              </CardContent>
            </Card>}

          {/* Graph Statistics */}
          <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-lg">图谱统计</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-2xl font-bold text-blue-600">
                    {nodes.filter(n => n.type === 'concept').length}
                  </div>
                  <div className="text-gray-600">概念节点</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-green-600">
                    {nodes.filter(n => n.type === 'document').length}
                  </div>
                  <div className="text-gray-600">文档节点</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-purple-600">{links.length}</div>
                  <div className="text-gray-600">关联链接</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-orange-600">
                    {Math.round(links.length / nodes.length * 100) / 100}
                  </div>
                  <div className="text-gray-600">平均连接度</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>;
}