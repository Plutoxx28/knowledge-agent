import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger, DropdownMenuCheckboxItem, DropdownMenuLabel } from '@/components/ui/dropdown-menu';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog';
import { useToast } from '@/hooks/use-toast';
import { Search, RefreshCw, Link, Trash, FileText, Download, Tag, FolderOpen, TrendingUp, Activity, Loader2 } from 'lucide-react';
import { StatCard } from '@/components/ui/stat-card';
import { apiClient, formatError } from '@/lib/api';

interface Document {
  id: string;
  title: string;
  category?: string;
  concepts?: string[];
  created_at: string;
  word_count?: number;
  concept_count?: number;
}
const KnowledgeBase = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [filters, setFilters] = useState({
    tags: [] as string[]
  });
  const [documents, setDocuments] = useState<Document[]>([]);
  const [concepts, setConcepts] = useState<any[]>([]);
  const [stats, setStats] = useState({ documents: 0, concepts: 0, links: 0 });
  const [loading, setLoading] = useState(true);
  const [filteredDocuments, setFilteredDocuments] = useState<Document[]>([]);

  // 加载数据
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async (rescan: boolean = false) => {
    setLoading(true);
    try {
      // 如果需要重新扫描，先执行重新扫描
      if (rescan) {
        const rescanResult = await apiClient.rescanKnowledgeBase();
        if (rescanResult.success) {
          toast({
            title: "重新扫描成功",
            description: `扫描了 ${rescanResult.stats?.scanned_files || 0} 个文件，更新了 ${rescanResult.stats?.updated_files || 0} 个文件`,
          });
        }
      }

      // 并行加载文档、概念和统计数据
      const [documentsRes, conceptsRes, statsRes] = await Promise.all([
        apiClient.getDocuments(100, 0),
        apiClient.getConcepts(100, 0),
        apiClient.getStats().catch(() => ({ documents: 0, concepts: 0, links: 0, last_updated: '' }))
      ]);

      setDocuments(documentsRes.documents || []);
      setConcepts(conceptsRes.concepts || []);
      setStats({
        documents: statsRes.documents || documentsRes.total || 0,
        concepts: statsRes.concepts || conceptsRes.total || 0,
        links: statsRes.links || 0
      });

      if (!rescan) {
        toast({
          title: "数据加载成功",
          description: `加载了 ${documentsRes.documents?.length || 0} 个文档和 ${conceptsRes.concepts?.length || 0} 个概念`,
        });
      }
    } catch (error) {
      console.error('加载数据失败:', error);
      toast({
        title: rescan ? "重新扫描失败" : "数据加载失败",
        description: formatError(error),
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  // 从概念中提取可用的标签
  const availableTags = [...new Set(concepts.map(c => c.type).filter(Boolean))].slice(0, 10);
  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('zh-CN');
    } catch {
      return '未知时间';
    }
  };
  const toggleSelectDocument = (id: string, checked: boolean) => {
    if (checked) {
      setSelectedDocuments(prev => [...prev, id]);
    } else {
      setSelectedDocuments(prev => prev.filter(docId => docId !== id));
    }
  };
  const toggleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedDocuments(documents.map(doc => doc.id));
    } else {
      setSelectedDocuments([]);
    }
  };
  const toggleTagFilter = (tag: string, checked: boolean) => {
    setFilters(prev => ({
      ...prev,
      tags: checked ? [...prev.tags, tag] : prev.tags.filter(item => item !== tag)
    }));
  };

  // 过滤文档的函数
  const filterDocuments = () => {
    let filtered = documents;
    
    // 根据搜索关键词过滤
    if (searchQuery.trim()) {
      filtered = filtered.filter(doc => 
        doc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        doc.category.toLowerCase().includes(searchQuery.toLowerCase()) ||
        doc.concepts.some(concept => concept.toLowerCase().includes(searchQuery.toLowerCase()))
      );
    }
    
    // 根据标签过滤
    if (filters.tags.length > 0) {
      // 这里可以根据实际需要实现标签过滤逻辑
    }
    
    setFilteredDocuments(filtered);
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setFilteredDocuments([]);
      toast({
        title: "搜索完成",
        description: "显示所有文档",
      });
      return;
    }

    try {
      const results = await apiClient.searchDocuments({
        query: searchQuery,
        limit: 50,
        threshold: 0.3
      });

      setFilteredDocuments(results.documents.map(doc => ({
        id: doc.id,
        title: doc.title || '无标题',
        created_at: doc.created_at || new Date().toISOString(),
        concept_count: doc.concept_count || 0
      })));

      toast({
        title: "搜索完成",
        description: `找到 ${results.documents.length} 个相关文档`,
      });
    } catch (error) {
      toast({
        title: "搜索失败",
        description: formatError(error),
        variant: "destructive",
      });
    }
  };

  const handleDeleteDocument = async (id: string, title: string) => {
    try {
      // 调用删除API
      const result = await apiClient.deleteDocument(id);
      
      if (result.success) {
        // 从本地状态中移除已删除的文档
        setDocuments(prev => prev.filter(doc => doc.id !== id));
        setFilteredDocuments(prev => prev.filter(doc => doc.id !== id));
        
        // 更新统计数据
        setStats(prev => ({
          ...prev,
          documents: prev.documents - 1
        }));
        
        toast({
          title: "文档已删除",
          description: result.message,
        });
      }
    } catch (error) {
      console.error('删除文档失败:', error);
      toast({
        title: "删除失败",
        description: formatError(error),
        variant: "destructive",
      });
    }
  };
  const allSelected = selectedDocuments.length === documents.length;
  return <div className="space-y-8">
      {/* Header Section */}
      <div className="flex items-start justify-between">
        <div className="space-y-4">
          <h1 className="font-bold text-gray-900 text-3xl">
            知识库
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl">
            智能管理和组织你的知识文档，构建属于你的知识图谱
          </p>
        </div>

        {/* Stats Cards - using unified StatCard component */}
        <div className="flex gap-4">
          <StatCard 
            title="文档" 
            value={loading ? "--" : stats.documents} 
            icon={<FileText className="w-5 h-5" />} 
            variant="primary" 
          />
          <StatCard 
            title="概念" 
            value={loading ? "--" : stats.concepts} 
            icon={<TrendingUp className="w-5 h-5" />} 
            variant="success" 
          />
          <StatCard 
            title="链接" 
            value={loading ? "--" : stats.links} 
            icon={<Activity className="w-5 h-5" />} 
            variant="info" 
          />
        </div>
      </div>

      {/* Search and Filter Section */}
      <Card className="bg-card/70 backdrop-blur-sm border-border/50 shadow-xl rounded-3xl overflow-hidden">
        <CardContent className="p-8">
          <div className="flex flex-row gap-4 items-center">
            <div className="relative flex-1 group">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400 group-focus-within:text-blue-500 transition-colors" />
              <Input 
                placeholder="搜索文档、概念或内容..." 
                className="pl-12 pr-24 h-12 text-base bg-white/80 border-gray-200/50 rounded-2xl focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 transition-all" 
                value={searchQuery} 
                onChange={e => setSearchQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleSearch()}
              />
              <Button 
                onClick={handleSearch}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 h-8 px-4 bg-blue-500 hover:bg-blue-600 text-white rounded-xl"
              >
                搜索
              </Button>
            </div>

            <div className="flex gap-3">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="h-12 rounded-2xl bg-white/80 border-gray-200/50 hover:bg-white/90">
                    <Tag className="mr-2 h-4 w-4" />
                    按标签筛选
                    {filters.tags.length > 0 && <Badge variant="secondary" className="ml-2">
                        {filters.tags.length}
                      </Badge>}
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56 rounded-2xl">
                  <DropdownMenuLabel>选择标签</DropdownMenuLabel>
                  {availableTags.map(tag => <DropdownMenuCheckboxItem key={tag} checked={filters.tags.includes(tag)} onCheckedChange={checked => toggleTagFilter(tag, !!checked)} className="rounded-xl">
                      {tag}
                    </DropdownMenuCheckboxItem>)}
                </DropdownMenuContent>
              </DropdownMenu>

              <Button 
                variant="outline" 
                className="h-12 rounded-2xl bg-white/80 border-gray-200/50 hover:bg-white/90"
                onClick={() => loadData(true)}
                disabled={loading}
              >
                {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
                重新扫描
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Documents Section */}
      <Card className="bg-card/70 backdrop-blur-sm border-border/50 shadow-xl rounded-3xl overflow-hidden">
        <CardContent className="p-0">
          <div className="overflow-hidden">
            <Table>
              <TableHeader className="bg-gradient-to-r from-gray-50/80 to-blue-50/50">
                <TableRow className="border-gray-200/50 hover:bg-transparent">
                  <TableHead className="w-12 py-6">
                    <div className="flex items-center justify-center">
                      <Checkbox checked={allSelected} onCheckedChange={toggleSelectAll} className="rounded-lg" />
                    </div>
                  </TableHead>
                  <TableHead className="font-semibold text-gray-700 py-6">文档信息</TableHead>
                  <TableHead className="font-semibold text-gray-700 py-6">更新时间</TableHead>
                  <TableHead className="w-24 font-semibold text-gray-700 py-6 text-center">操作</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={4} className="py-12 text-center">
                      <div className="flex items-center justify-center gap-2">
                        <Loader2 className="h-5 w-5 animate-spin" />
                        <span className="text-gray-500">加载中...</span>
                      </div>
                    </TableCell>
                  </TableRow>
                ) : (filteredDocuments.length > 0 ? filteredDocuments : documents).length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={4} className="py-12 text-center">
                      <div className="text-gray-500">
                        {filteredDocuments.length === 0 && searchQuery ? '没有找到匹配的文档' : '暂无文档数据'}
                      </div>
                    </TableCell>
                  </TableRow>
                ) : (filteredDocuments.length > 0 ? filteredDocuments : documents).map((doc) => <TableRow key={doc.id} className="border-gray-200/30 hover:bg-blue-50/30 transition-all duration-200 group">
                    <TableCell className="py-6">
                      <div className="flex items-center justify-center">
                        <Checkbox checked={selectedDocuments.includes(doc.id)} onCheckedChange={checked => toggleSelectDocument(doc.id, !!checked)} className="rounded-lg" />
                      </div>
                    </TableCell>
                    <TableCell className="py-6">
                      <div className="flex items-start gap-4">
                        <div className="w-12 h-12 bg-gradient-to-br from-blue-500/10 to-indigo-500/10 rounded-2xl flex items-center justify-center border border-blue-200/30">
                          <FileText className="w-5 h-5 text-blue-600" />
                        </div>
                        <div className="space-y-1">
                          <div 
                            className="font-semibold text-gray-900 group-hover:text-blue-700 transition-colors cursor-pointer"
                            onClick={() => navigate(`/document/${doc.id}`)}
                          >
                            {doc.title || '无标题'}
                          </div>
                          <div className="flex items-center gap-4 text-sm text-gray-500">
                            <span className="flex items-center gap-1">
                              <FileText className="w-3 h-3" />
                              {doc.word_count ? `${doc.word_count.toLocaleString()} 字` : '未知大小'}
                            </span>
                            <span className="flex items-center gap-1">
                              <Link className="w-3 h-3" />
                              {doc.concept_count || 0} 个概念
                            </span>
                            {doc.category && (
                              <span className="flex items-center gap-1">
                                <Tag className="w-3 h-3" />
                                {doc.category}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    </TableCell>
                    <TableCell className="py-6">
                      <span className="text-sm text-gray-600">{formatDate(doc.created_at)}</span>
                    </TableCell>
                    <TableCell className="py-6 text-center">
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button variant="ghost" size="sm" className="h-8 px-3 text-red-600 hover:text-red-700 hover:bg-red-50 rounded-xl">
                            <Trash className="w-4 h-4 mr-2" />
                            删除文档
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent className="rounded-2xl">
                          <AlertDialogHeader>
                            <AlertDialogTitle>确认删除</AlertDialogTitle>
                            <AlertDialogDescription>
                              你确定要删除文档 "{doc.title}" 吗？此操作无法撤销。
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel className="rounded-xl">取消</AlertDialogCancel>
                            <AlertDialogAction 
                              className="rounded-xl bg-red-500 hover:bg-red-600"
                              onClick={() => handleDeleteDocument(doc.id, doc.title)}
                            >
                              删除
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </TableCell>
                  </TableRow>)}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Batch Operations Bar */}
      {selectedDocuments.length > 0 && 
        <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-50">
          <div className="bg-white/95 backdrop-blur-xl border border-gray-200/50 rounded-2xl shadow-2xl shadow-gray-900/10 px-6 py-4 max-w-4xl">
            <div className="flex items-center justify-between gap-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
                  <span className="text-lg font-bold text-white">{selectedDocuments.length}</span>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900 whitespace-nowrap">已选择 {selectedDocuments.length} 个文档</p>
                </div>
              </div>
              
              <div className="h-10 w-px bg-gray-200"></div>
              
              <div className="flex items-center gap-3">
                <Button size="sm" variant="outline" className="rounded-xl bg-white hover:bg-gray-50 border-gray-200 text-gray-700 hover:text-gray-900 transition-all">
                  <Download className="w-4 h-4 mr-2" />
                  导出
                </Button>
                <Button size="sm" variant="outline" className="rounded-xl bg-white hover:bg-gray-50 border-gray-200 text-gray-700 hover:text-gray-900 transition-all">
                  <Tag className="w-4 h-4 mr-2" />
                  标签
                </Button>
                <Button size="sm" variant="outline" className="rounded-xl bg-white hover:bg-gray-50 border-gray-200 text-gray-700 hover:text-gray-900 transition-all">
                  <FolderOpen className="w-4 h-4 mr-2" />
                  移动
                </Button>
                <Button size="sm" variant="destructive" className="rounded-xl bg-red-500 hover:bg-red-600 text-white transition-all">
                  <Trash className="w-4 h-4 mr-2" />
                  删除
                </Button>
              </div>
            </div>
          </div>
        </div>}
    </div>;
};
export default KnowledgeBase;
