# Knowledge Agent 前端控制台 PRD

> 为 Knowledge Agent 智能知识整理系统设计的现代化 Web 控制台界面

**版本**: v2.2.0  
**创建日期**: 2024-06-30  
**目标**: 为 Knowledge Agent 系统提供直观、高效的 Web 操作界面

---

## 📋 产品概述

### 产品定位
Knowledge Agent 前端控制台是一个基于 React + TypeScript 的现代化 Web 应用，为用户提供可视化的知识管理操作界面，支持内容处理、链接管理、概念图谱展示等核心功能。

### 核心价值
- **简化操作**：将复杂的命令行操作转换为直观的图形界面
- **实时反馈**：提供处理进度、结果状态的实时可视化
- **知识导航**：通过概念图谱和链接系统实现知识的快速导航
- **协作友好**：支持多用户场景下的知识库管理

## 🎯 用户群体

### 主要用户
- **知识工作者**：研究人员、分析师、内容创作者
- **AI从业者**：机器学习工程师、AI研究人员
- **团队协作者**：需要共享和管理知识的团队成员

### 使用场景
- 处理AI对话记录，生成结构化知识笔记
- 管理技术文档和概念间的关联关系
- 可视化浏览和导航知识图谱
- 批量处理和整理知识内容

## 🏗️ 技术架构要求

### 前端技术栈
```typescript
{
  "framework": "React 18+",
  "language": "TypeScript 5+",
  "styling": "TailwindCSS + shadcn/ui",
  "stateManagement": "Zustand",
  "routing": "React Router v6",
  "dataFetching": "TanStack Query (React Query)",
  "charts": "Recharts + D3.js",
  "icons": "Lucide React",
  "build": "Vite",
  "testing": "Vitest + Testing Library"
}
```

### 后端集成
```typescript
interface APIEndpoints {
  // 内容处理
  processContent: 'POST /api/process',
  getProcessStatus: 'GET /api/process/{taskId}/status',
  
  // 知识库管理
  getDocuments: 'GET /api/documents',
  getDocument: 'GET /api/documents/{id}',
  deleteDocument: 'DELETE /api/documents/{id}',
  
  // 链接系统
  scanKnowledgeBase: 'POST /api/links/scan',
  getLinkReport: 'GET /api/links/report',
  getConceptLinks: 'GET /api/concepts/{name}/links',
  
  // 概念图谱
  getConceptGraph: 'GET /api/graph',
  searchConcepts: 'GET /api/concepts/search'
}
```

### 实时通信
- **WebSocket**: 处理进度推送
- **Server-Sent Events**: 系统状态更新
- **Polling**: 作为降级方案

## 🎨 界面设计规范

### 设计系统
```typescript
interface DesignSystem {
  colors: {
    primary: 'blue-600',      // 主色调
    secondary: 'gray-600',    // 次要色
    success: 'green-600',     // 成功状态
    warning: 'yellow-600',    // 警告状态
    error: 'red-600',         // 错误状态
    background: 'white',      // 背景色
    surface: 'gray-50'        // 卡片背景
  },
  typography: {
    fontFamily: 'Inter, system-ui, sans-serif',
    sizes: {
      xs: '0.75rem',
      sm: '0.875rem',
      base: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      '2xl': '1.5rem',
      '3xl': '1.875rem'
    }
  },
  spacing: 'Tailwind default (4px base)',
  borderRadius: {
    sm: '0.375rem',
    md: '0.5rem',
    lg: '0.75rem'
  }
}
```

### 响应式设计
- **桌面端**: ≥1024px (主要目标)
- **平板端**: 768px-1023px (优化体验)
- **移动端**: <768px (基础功能)

## 📱 功能模块详细设计

### 1. 主布局 (Layout)

```typescript
interface MainLayoutProps {
  children: React.ReactNode;
}

interface LayoutStructure {
  header: {
    logo: 'Knowledge Agent',
    navigation: ['处理中心', '知识库', '概念图谱', '设置'],
    userActions: ['通知', '用户菜单']
  },
  sidebar: {
    width: '280px',
    collapsible: true,
    sections: ['快速操作', '最近文档', '概念导航']
  },
  main: {
    content: 'dynamic based on route',
    maxWidth: '1200px',
    padding: '24px'
  },
  footer: {
    systemStatus: 'processing status indicator',
    version: 'v2.2.0'
  }
}
```

**实现要求**:
- 使用 React Router 进行路由管理
- 响应式侧边栏（移动端可收起）
- 面包屑导航
- 暗黑模式切换

### 2. 内容处理中心 (Processing Hub)

```typescript
interface ProcessingHubState {
  inputMode: 'text' | 'conversation' | 'url' | 'file';
  content: string;
  metadata: {
    source: string;
    topic: string;
    tags: string[];
  };
  options: {
    enableLinking: boolean;
    generateSummary: boolean;
    extractConcepts: boolean;
    strategy: 'standard' | 'hierarchical' | 'streaming';
  };
  processingStatus: ProcessingStatus;
}

interface ProcessingStatus {
  taskId: string;
  stage: 'parsing' | 'structuring' | 'linking' | 'saving';
  progress: number; // 0-100
  message: string;
  estimatedTime: number; // seconds
  error?: string;
}
```

#### 2.1 输入区域
```tsx
<div className="processing-input">
  {/* 输入方式选择 */}
  <Tabs defaultValue="text">
    <TabsList>
      <TabsTrigger value="text">📝 文本输入</TabsTrigger>
      <TabsTrigger value="conversation">💬 对话记录</TabsTrigger>
      <TabsTrigger value="url">🔗 URL链接</TabsTrigger>
      <TabsTrigger value="file">📄 文件上传</TabsTrigger>
    </TabsList>
    
    <TabsContent value="text">
      <Textarea 
        placeholder="请输入要处理的文本内容..."
        className="min-h-[200px]"
        value={content}
        onChange={setContent}
      />
    </TabsContent>
    {/* 其他Tab内容 */}
  </Tabs>
  
  {/* 配置面板 */}
  <Collapsible>
    <CollapsibleTrigger>⚙️ 处理配置</CollapsibleTrigger>
    <CollapsibleContent>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label>处理策略</Label>
          <Select value={strategy} onValueChange={setStrategy}>
            <SelectItem value="standard">标准处理</SelectItem>
            <SelectItem value="hierarchical">层次化处理</SelectItem>
            <SelectItem value="streaming">流式处理</SelectItem>
          </Select>
        </div>
        
        <div className="space-y-2">
          <div className="flex items-center space-x-2">
            <Checkbox id="linking" checked={enableLinking} />
            <Label htmlFor="linking">启用概念链接</Label>
          </div>
          <div className="flex items-center space-x-2">
            <Checkbox id="summary" checked={generateSummary} />
            <Label htmlFor="summary">生成摘要</Label>
          </div>
        </div>
      </div>
    </CollapsibleContent>
  </Collapsible>
  
  {/* 元数据输入 */}
  <div className="metadata-inputs grid grid-cols-3 gap-4">
    <Input placeholder="来源" value={source} onChange={setSource} />
    <Input placeholder="主题" value={topic} onChange={setTopic} />
    <TagInput placeholder="标签" value={tags} onChange={setTags} />
  </div>
  
  {/* 操作按钮 */}
  <div className="actions flex gap-2">
    <Button onClick={startProcessing} disabled={!content || processing}>
      {processing ? <Spinner /> : <Play />}
      {processing ? '处理中...' : '开始处理'}
    </Button>
    <Button variant="outline" onClick={clearInput}>
      <X /> 清空
    </Button>
  </div>
</div>
```

#### 2.2 进度监控
```tsx
<div className="processing-monitor">
  {processingStatus && (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Spinner className="animate-spin" />
          处理进度
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* 总体进度 */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>当前阶段：{stageLabels[processingStatus.stage]}</span>
            <span>{processingStatus.progress}%</span>
          </div>
          <Progress value={processingStatus.progress} />
        </div>
        
        {/* 详细信息 */}
        <div className="mt-4 text-sm text-gray-600">
          <div>任务ID: {processingStatus.taskId}</div>
          <div>当前任务: {processingStatus.message}</div>
          <div>预计剩余: {formatTime(processingStatus.estimatedTime)}</div>
        </div>
        
        {/* 阶段指示器 */}
        <div className="mt-4 flex items-center gap-2">
          {stages.map((stage, index) => (
            <div key={stage} className="flex items-center">
              <div className={`w-3 h-3 rounded-full ${
                getStageStatus(stage, processingStatus.stage)
              }`} />
              <span className="ml-1 text-xs">{stageLabels[stage]}</span>
              {index < stages.length - 1 && (
                <ChevronRight className="w-3 h-3 mx-2 text-gray-400" />
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )}
</div>
```

#### 2.3 结果展示
```tsx
<div className="processing-result">
  {result && (
    <Tabs defaultValue="preview">
      <TabsList>
        <TabsTrigger value="preview">📖 预览</TabsTrigger>
        <TabsTrigger value="raw">📄 原始内容</TabsTrigger>
        <TabsTrigger value="stats">📊 统计信息</TabsTrigger>
      </TabsList>
      
      <TabsContent value="preview">
        <Card>
          <CardContent className="p-6">
            {/* Markdown渲染 */}
            <div className="prose max-w-none">
              <ReactMarkdown 
                components={{
                  // 自定义概念链接渲染
                  text: ({ children }) => {
                    return renderConceptLinks(children);
                  }
                }}
              >
                {result.content}
              </ReactMarkdown>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
      
      <TabsContent value="stats">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard 
            title="概念数量" 
            value={result.statistics.conceptCount}
            icon={<Hash />}
          />
          <StatCard 
            title="内部链接" 
            value={result.statistics.internalLinks}
            icon={<Link />}
          />
          <StatCard 
            title="处理时长" 
            value={formatDuration(result.statistics.processingTime)}
            icon={<Clock />}
          />
          <StatCard 
            title="质量评分" 
            value={`${result.statistics.qualityScore}/100`}
            icon={<Star />}
          />
        </div>
      </TabsContent>
    </Tabs>
    
    {/* 操作按钮 */}
    <div className="mt-4 flex gap-2">
      <Button onClick={saveToKnowledgeBase}>
        <Save /> 保存到知识库
      </Button>
      <Button variant="outline" onClick={copyContent}>
        <Copy /> 复制内容
      </Button>
      <Button variant="outline" onClick={exportFile}>
        <Download /> 导出文件
      </Button>
      <Button variant="outline" onClick={viewLinks}>
        <ExternalLink /> 查看链接
      </Button>
    </div>
  )}
</div>
```

### 3. 知识库管理 (Knowledge Base)

```typescript
interface KnowledgeBaseState {
  documents: Document[];
  categories: Category[];
  currentView: 'list' | 'grid' | 'tree';
  searchQuery: string;
  filters: {
    category: string[];
    tags: string[];
    dateRange: [Date, Date];
  };
  selectedDocuments: string[];
}

interface Document {
  id: string;
  title: string;
  path: string;
  category: string;
  tags: string[];
  concepts: string[];
  createdAt: Date;
  updatedAt: Date;
  size: number;
  wordCount: number;
  linkCount: number;
}
```

#### 3.1 文档列表视图
```tsx
<div className="knowledge-base">
  {/* 搜索和过滤 */}
  <div className="search-filters mb-6">
    <div className="flex gap-4 items-center">
      <div className="relative flex-1">
        <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
        <Input 
          placeholder="搜索文档、概念或内容..."
          className="pl-10"
          value={searchQuery}
          onChange={setSearchQuery}
        />
      </div>
      
      <Select value={viewMode} onValueChange={setViewMode}>
        <SelectTrigger className="w-32">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="list">
            <List className="w-4 h-4 mr-2" /> 列表
          </SelectItem>
          <SelectItem value="grid">
            <Grid className="w-4 h-4 mr-2" /> 网格
          </SelectItem>
          <SelectItem value="tree">
            <Tree className="w-4 h-4 mr-2" /> 树形
          </SelectItem>
        </SelectContent>
      </Select>
      
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline">
            <Filter /> 筛选
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-56">
          <DropdownMenuLabel>分类</DropdownMenuLabel>
          {categories.map(category => (
            <DropdownMenuCheckboxItem
              key={category.id}
              checked={filters.category.includes(category.id)}
              onCheckedChange={(checked) => 
                toggleFilter('category', category.id, checked)
              }
            >
              {category.name}
            </DropdownMenuCheckboxItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
      
      <Button onClick={refreshDocuments}>
        <RefreshCw /> 刷新
      </Button>
    </div>
  </div>
  
  {/* 文档列表 */}
  <div className="documents-container">
    {viewMode === 'list' && (
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-12">
              <Checkbox 
                checked={allSelected}
                onCheckedChange={toggleSelectAll}
              />
            </TableHead>
            <TableHead>文档</TableHead>
            <TableHead>分类</TableHead>
            <TableHead>概念数</TableHead>
            <TableHead>更新时间</TableHead>
            <TableHead className="w-24">操作</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {documents.map(doc => (
            <TableRow key={doc.id}>
              <TableCell>
                <Checkbox 
                  checked={selectedDocuments.includes(doc.id)}
                  onCheckedChange={(checked) => 
                    toggleSelectDocument(doc.id, checked)
                  }
                />
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-gray-400" />
                  <div>
                    <div className="font-medium">{doc.title}</div>
                    <div className="text-sm text-gray-500">
                      {doc.wordCount} 字 · {doc.linkCount} 个链接
                    </div>
                  </div>
                </div>
              </TableCell>
              <TableCell>
                <Badge variant="secondary">{doc.category}</Badge>
              </TableCell>
              <TableCell>{doc.concepts.length}</TableCell>
              <TableCell>{formatDate(doc.updatedAt)}</TableCell>
              <TableCell>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="sm">
                      <MoreHorizontal className="w-4 h-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    <DropdownMenuItem onClick={() => viewDocument(doc.id)}>
                      <Eye className="w-4 h-4 mr-2" /> 查看
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => editDocument(doc.id)}>
                      <Edit className="w-4 h-4 mr-2" /> 编辑
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => viewLinks(doc.id)}>
                      <Link className="w-4 h-4 mr-2" /> 查看链接
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem 
                      onClick={() => deleteDocument(doc.id)}
                      className="text-red-600"
                    >
                      <Trash className="w-4 h-4 mr-2" /> 删除
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    )}
    
    {/* 批量操作栏 */}
    {selectedDocuments.length > 0 && (
      <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 
                      bg-white border rounded-lg shadow-lg p-4 flex items-center gap-2">
        <span className="text-sm">
          已选择 {selectedDocuments.length} 个文档
        </span>
        <Button size="sm" variant="outline" onClick={exportSelected}>
          <Download className="w-4 h-4 mr-1" /> 导出
        </Button>
        <Button size="sm" variant="outline" onClick={tagSelected}>
          <Tag className="w-4 h-4 mr-1" /> 标签
        </Button>
        <Button size="sm" variant="outline" onClick={moveSelected}>
          <FolderOpen className="w-4 h-4 mr-1" /> 移动
        </Button>
        <Button size="sm" variant="destructive" onClick={deleteSelected}>
          <Trash className="w-4 h-4 mr-1" /> 删除
        </Button>
      </div>
    )}
  </div>
</div>
```

### 4. 概念图谱 (Concept Graph)

```typescript
interface ConceptGraphState {
  nodes: GraphNode[];
  links: GraphLink[];
  selectedNode: string | null;
  hoveredNode: string | null;
  filterOptions: {
    nodeType: 'all' | 'concept' | 'document';
    linkStrength: [number, number];
    maxNodes: number;
  };
  layoutOptions: {
    algorithm: 'force' | 'circle' | 'tree';
    strength: number;
    distance: number;
  };
}

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
```

#### 4.1 图谱可视化
```tsx
<div className="concept-graph">
  {/* 控制面板 */}
  <div className="graph-controls mb-4 p-4 bg-gray-50 rounded-lg">
    <div className="flex items-center gap-4">
      {/* 布局控制 */}
      <div className="flex items-center gap-2">
        <Label>布局算法:</Label>
        <Select value={layoutAlgorithm} onValueChange={setLayoutAlgorithm}>
          <SelectItem value="force">力导向</SelectItem>
          <SelectItem value="circle">环形</SelectItem>
          <SelectItem value="tree">树形</SelectItem>
        </Select>
      </div>
      
      {/* 节点过滤 */}
      <div className="flex items-center gap-2">
        <Label>显示节点:</Label>
        <Select value={nodeFilter} onValueChange={setNodeFilter}>
          <SelectItem value="all">全部</SelectItem>
          <SelectItem value="concept">仅概念</SelectItem>
          <SelectItem value="document">仅文档</SelectItem>
        </Select>
      </div>
      
      {/* 节点数量限制 */}
      <div className="flex items-center gap-2">
        <Label>最大节点数:</Label>
        <Slider
          value={[maxNodes]}
          onValueChange={(value) => setMaxNodes(value[0])}
          max={200}
          min={10}
          step={10}
          className="w-24"
        />
        <span className="text-sm text-gray-600">{maxNodes}</span>
      </div>
      
      {/* 操作按钮 */}
      <div className="flex gap-2 ml-auto">
        <Button size="sm" variant="outline" onClick={resetZoom}>
          <ZoomIn /> 重置视图
        </Button>
        <Button size="sm" variant="outline" onClick={exportGraph}>
          <Download /> 导出图片
        </Button>
        <Button size="sm" variant="outline" onClick={refreshGraph}>
          <RefreshCw /> 刷新数据
        </Button>
      </div>
    </div>
  </div>
  
  {/* 图谱容器 */}
  <div className="graph-container relative">
    <div id="concept-graph-svg" className="w-full h-[600px] border rounded-lg">
      {/* D3.js 图谱将在这里渲染 */}
    </div>
    
    {/* 图例 */}
    <div className="absolute top-4 right-4 bg-white border rounded-lg p-3 shadow-sm">
      <h4 className="text-sm font-medium mb-2">图例</h4>
      <div className="space-y-1 text-xs">
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
          <span>概念链接</span>
        </div>
      </div>
    </div>
    
    {/* 加载状态 */}
    {loading && (
      <div className="absolute inset-0 bg-white/80 flex items-center justify-center">
        <div className="flex items-center gap-2">
          <Spinner className="animate-spin" />
          <span>正在生成图谱...</span>
        </div>
      </div>
    )}
  </div>
  
  {/* 节点详情面板 */}
  {selectedNode && (
    <div className="mt-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {getNodeIcon(selectedNodeData.type)}
            {selectedNodeData.label}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label className="text-sm text-gray-600">类型</Label>
              <div>{selectedNodeData.type === 'concept' ? '概念' : '文档'}</div>
            </div>
            <div>
              <Label className="text-sm text-gray-600">引用次数</Label>
              <div>{selectedNodeData.metadata.referenceCount}</div>
            </div>
            {selectedNodeData.type === 'concept' && (
              <div>
                <Label className="text-sm text-gray-600">是否有文档</Label>
                <div>
                  {selectedNodeData.metadata.hasDocument ? (
                    <Badge className="bg-green-100 text-green-800">有</Badge>
                  ) : (
                    <Badge className="bg-red-100 text-red-800">无</Badge>
                  )}
                </div>
              </div>
            )}
          </div>
          
          {/* 相关链接 */}
          <div className="mt-4">
            <Label className="text-sm text-gray-600 mb-2 block">相关链接</Label>
            <div className="space-y-1">
              {getNodeLinks(selectedNode).map(link => (
                <div key={link.id} className="flex items-center gap-2 text-sm">
                  <ChevronRight className="w-3 h-3" />
                  <span 
                    className="cursor-pointer hover:text-blue-600"
                    onClick={() => selectNode(link.target)}
                  >
                    {link.targetLabel}
                  </span>
                </div>
              ))}
            </div>
          </div>
          
          {/* 操作按钮 */}
          <div className="mt-4 flex gap-2">
            {selectedNodeData.type === 'concept' && 
             selectedNodeData.metadata.hasDocument && (
              <Button size="sm" onClick={() => openDocument(selectedNode)}>
                <FileText className="w-4 h-4 mr-1" /> 查看文档
              </Button>
            )}
            <Button size="sm" variant="outline" onClick={() => focusNode(selectedNode)}>
              <Target className="w-4 h-4 mr-1" /> 聚焦
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )}
</div>
```

### 5. 设置页面 (Settings)

```typescript
interface SettingsState {
  api: {
    openrouterApiKey: string;
    modelName: string;
    baseUrl: string;
    timeout: number;
  };
  processing: {
    defaultStrategy: string;
    enableLinking: boolean;
    maxChunkSize: number;
    chunkOverlap: number;
  };
  ui: {
    theme: 'light' | 'dark' | 'system';
    language: 'zh' | 'en';
    sidebarCollapsed: boolean;
  };
  knowledgeBase: {
    path: string;
    autoSave: boolean;
    backupEnabled: boolean;
    syncEnabled: boolean;
  };
}
```

#### 5.1 设置表单
```tsx
<div className="settings">
  <div className="max-w-2xl mx-auto">
    <h1 className="text-3xl font-bold mb-8">系统设置</h1>
    
    <Tabs defaultValue="api" className="space-y-6">
      <TabsList className="grid grid-cols-4 w-full">
        <TabsTrigger value="api">API配置</TabsTrigger>
        <TabsTrigger value="processing">处理配置</TabsTrigger>
        <TabsTrigger value="ui">界面设置</TabsTrigger>
        <TabsTrigger value="storage">存储设置</TabsTrigger>
      </TabsList>
      
      <TabsContent value="api" className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>OpenRouter API 配置</CardTitle>
            <CardDescription>
              配置AI模型的API连接信息
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="api-key">API 密钥</Label>
              <div className="relative">
                <Input
                  id="api-key"
                  type={showApiKey ? "text" : "password"}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="输入你的 OpenRouter API 密钥"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-0 top-0 h-full px-3"
                  onClick={() => setShowApiKey(!showApiKey)}
                >
                  {showApiKey ? <EyeOff /> : <Eye />}
                </Button>
              </div>
            </div>
            
            <div>
              <Label htmlFor="model">模型选择</Label>
              <Select value={modelName} onValueChange={setModelName}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="google/gemini-2.5-pro">
                    Gemini 2.5 Pro
                  </SelectItem>
                  <SelectItem value="anthropic/claude-3-opus">
                    Claude 3 Opus
                  </SelectItem>
                  <SelectItem value="openai/gpt-4">
                    GPT-4
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Label htmlFor="timeout">请求超时 (秒)</Label>
              <Input
                id="timeout"
                type="number"
                value={timeout}
                onChange={(e) => setTimeout(Number(e.target.value))}
                min={10}
                max={300}
              />
            </div>
            
            {/* API 连接测试 */}
            <div className="pt-4 border-t">
              <Button onClick={testApiConnection} disabled={testing}>
                {testing ? <Spinner className="mr-2" /> : <Zap className="mr-2" />}
                测试连接
              </Button>
              {testResult && (
                <div className={`mt-2 text-sm ${
                  testResult.success ? 'text-green-600' : 'text-red-600'
                }`}>
                  {testResult.message}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </TabsContent>
      
      <TabsContent value="processing" className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>处理配置</CardTitle>
            <CardDescription>
              配置内容处理的默认参数
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label>默认处理策略</Label>
              <RadioGroup value={defaultStrategy} onValueChange={setDefaultStrategy}>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="standard" id="standard" />
                  <Label htmlFor="standard">标准处理</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="hierarchical" id="hierarchical" />
                  <Label htmlFor="hierarchical">层次化处理</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="streaming" id="streaming" />
                  <Label htmlFor="streaming">流式处理</Label>
                </div>
              </RadioGroup>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <Checkbox 
                  id="enable-linking" 
                  checked={enableLinking}
                  onCheckedChange={setEnableLinking}
                />
                <Label htmlFor="enable-linking">默认启用概念链接</Label>
              </div>
            </div>
            
            <div>
              <Label>最大分块大小</Label>
              <div className="flex items-center space-x-2">
                <Slider
                  value={[maxChunkSize]}
                  onValueChange={(value) => setMaxChunkSize(value[0])}
                  max={5000}
                  min={1000}
                  step={500}
                  className="flex-1"
                />
                <span className="text-sm text-gray-600 w-16">
                  {maxChunkSize}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
      
      {/* 保存按钮 */}
      <div className="flex justify-end gap-2">
        <Button variant="outline" onClick={resetSettings}>
          重置
        </Button>
        <Button onClick={saveSettings} disabled={saving}>
          {saving ? <Spinner className="mr-2" /> : <Save className="mr-2" />}
          保存设置
        </Button>
      </div>
    </Tabs>
  </div>
</div>
```

## 🔄 实时功能实现

### WebSocket 连接管理
```typescript
class WebSocketManager {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  
  connect(url: string) {
    this.ws = new WebSocket(url);
    
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleMessage(data);
    };
    
    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.attemptReconnect();
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }
  
  private handleMessage(data: any) {
    switch (data.type) {
      case 'processing-progress':
        updateProcessingProgress(data.payload);
        break;
      case 'processing-complete':
        handleProcessingComplete(data.payload);
        break;
      case 'processing-error':
        handleProcessingError(data.payload);
        break;
    }
  }
  
  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        this.connect(getWebSocketUrl());
      }, 1000 * Math.pow(2, this.reconnectAttempts));
    }
  }
}
```

## 📱 组件库规范

### 共享组件
```typescript
// components/ui/
export { Button } from './button';
export { Input } from './input';
export { Card, CardHeader, CardTitle, CardContent } from './card';
export { Tabs, TabsList, TabsTrigger, TabsContent } from './tabs';
export { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from './select';
export { Checkbox } from './checkbox';
export { RadioGroup, RadioGroupItem } from './radio-group';
export { Progress } from './progress';
export { Slider } from './slider';
export { Badge } from './badge';
export { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from './table';
export { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent } from './dropdown-menu';
export { Dialog, DialogTrigger, DialogContent, DialogHeader, DialogTitle } from './dialog';
export { Toast, useToast } from './toast';
export { Spinner } from './spinner';

// 业务组件
export { ConceptLink } from './concept-link';
export { DocumentCard } from './document-card';
export { ProcessingStatus } from './processing-status';
export { StatCard } from './stat-card';
export { TagInput } from './tag-input';
```

### 自定义 Hooks
```typescript
// hooks/
export const useProcessing = () => {
  const [status, setStatus] = useState<ProcessingStatus | null>(null);
  
  const startProcessing = async (input: ProcessingInput) => {
    // 实现处理逻辑
  };
  
  return { status, startProcessing };
};

export const useWebSocket = (url: string) => {
  const [isConnected, setIsConnected] = useState(false);
  
  useEffect(() => {
    const ws = new WebSocketManager();
    ws.connect(url);
    
    return () => ws.disconnect();
  }, [url]);
  
  return { isConnected };
};

export const useKnowledgeBase = () => {
  const queryClient = useQueryClient();
  
  const { data: documents, isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: fetchDocuments
  });
  
  const mutation = useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => {
      queryClient.invalidateQueries(['documents']);
    }
  });
  
  return { documents, isLoading, deleteDocument: mutation.mutate };
};
```

## 🎨 样式规范

### TailwindCSS 配置
```javascript
// tailwind.config.js
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        gray: {
          50: '#f9fafb',
          100: '#f3f4f6',
          600: '#4b5563',
          700: '#374151',
          900: '#111827',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
      }
    }
  },
  plugins: [require('@tailwindcss/typography')]
};
```

## 🧪 测试要求

### 测试覆盖率目标
- **单元测试**: >80%
- **集成测试**: >60%
- **E2E测试**: 核心流程100%

### 测试用例示例
```typescript
// tests/components/ProcessingHub.test.tsx
describe('ProcessingHub', () => {
  it('should render input form correctly', () => {
    render(<ProcessingHub />);
    expect(screen.getByPlaceholderText('请输入要处理的文本内容...')).toBeInTheDocument();
  });
  
  it('should start processing when form is submitted', async () => {
    const mockProcess = jest.fn();
    render(<ProcessingHub onProcess={mockProcess} />);
    
    fireEvent.change(screen.getByRole('textbox'), {
      target: { value: 'test content' }
    });
    fireEvent.click(screen.getByText('开始处理'));
    
    expect(mockProcess).toHaveBeenCalledWith({
      content: 'test content',
      type: 'text',
      options: expect.any(Object)
    });
  });
});
```

## 🚀 部署要求

### 构建配置
```javascript
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          ui: ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu'],
          charts: ['d3', 'recharts']
        }
      }
    }
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000'
    }
  }
});
```

### 环境变量
```bash
# .env.production
VITE_API_BASE_URL=https://api.knowledge-agent.com
VITE_WS_URL=wss://ws.knowledge-agent.com
VITE_VERSION=2.2.0
```

## 📋 开发优先级

### Phase 1 (MVP) - 2周
1. ✅ 项目基础架构搭建
2. ✅ 主布局和路由系统
3. ✅ 内容处理中心基础功能
4. ✅ 简单的文档列表展示
5. ✅ 基础设置页面

### Phase 2 (完整功能) - 4周
1. ✅ 实时进度监控和WebSocket集成
2. ✅ 完整的知识库管理功能
3. ✅ 概念图谱可视化
4. ✅ 高级搜索和过滤
5. ✅ 批量操作功能

### Phase 3 (优化增强) - 2周
1. ✅ 性能优化和代码分割
2. ✅ 完整的测试覆盖
3. ✅ 错误处理和用户体验优化
4. ✅ 无障碍访问支持
5. ✅ 部署和CI/CD配置

---

**AI Coding工具指导说明**:

1. **严格遵循TypeScript类型定义**，确保类型安全
2. **使用shadcn/ui组件库**，保持设计一致性
3. **实现响应式设计**，支持多设备访问
4. **注重性能优化**，使用React.memo和useMemo
5. **添加适当的错误边界**和加载状态
6. **确保无障碍访问**，添加ARIA标签
7. **编写单元测试**，保证代码质量
8. **遵循代码规范**，使用ESLint和Prettier

这个PRD提供了完整的技术实现指导，AI coding工具可以根据这些详细规范生成高质量的前端代码。