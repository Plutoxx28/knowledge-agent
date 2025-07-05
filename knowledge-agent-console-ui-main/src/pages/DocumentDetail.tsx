import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TagInput } from '@/components/ui/tag-input';
import { ArrowLeft, Save, Edit, Eye, FileText, Clock, Link, Tag, Calendar, Users, Loader2 } from 'lucide-react';
import { apiClient, formatError } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface DocumentData {
  id: string;
  title: string;
  doc_path: string;
  concepts: string[];
  concept_count: number;
  outbound_links: string[];
  inbound_links: string[];
  created_at: string;
  word_count: number;
  content?: string;
}

const DocumentDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [document, setDocument] = useState<DocumentData | null>(null);
  const [content, setContent] = useState('');
  const [title, setTitle] = useState('');
  const [tags, setTags] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDocument();
  }, [id]);

  const loadDocument = async () => {
    if (!id) {
      setError('文档ID无效');
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const docData = await apiClient.getDocument(id);
      
      if (docData) {
        setDocument(docData);
        setTitle(docData.title || '无标题');
        setTags(docData.concepts || []);
        
        // 如果有文档路径，尝试读取文件内容
        if (docData.doc_path) {
          await loadFileContent(docData.doc_path);
        } else {
          setContent('暂无内容');
        }
      } else {
        setError('文档不存在');
      }
    } catch (error) {
      console.error('加载文档失败:', error);
      setError(formatError(error));
    } finally {
      setLoading(false);
    }
  };

  const loadFileContent = async (docPath: string) => {
    try {
      if (!id) return;
      
      const contentData = await apiClient.getDocumentContent(id);
      setContent(contentData.content || '暂无内容');
    } catch (error) {
      console.error('加载文档内容失败:', error);
      setContent(`文件内容加载失败: ${formatError(error)}\n\n文档路径: ${docPath}`);
    }
  };

  const handleSave = () => {
    setIsEditing(false);
    toast({
      title: '保存成功',
      description: '文档已保存（模拟操作）',
    });
    console.log('保存文档:', { title, tags, content });
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('zh-CN', {
        month: '2-digit',
        day: '2-digit'
      });
    } catch {
      return '未知';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex items-center gap-2">
          <Loader2 className="h-6 w-6 animate-spin" />
          <span className="text-gray-500">加载文档中...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center space-y-4">
          <div className="text-red-600 text-lg font-medium">{error}</div>
          <Button onClick={() => navigate('/knowledge-base')} variant="outline">
            <ArrowLeft className="w-4 h-4 mr-2" />
            返回知识库
          </Button>
        </div>
      </div>
    );
  }

  if (!document) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center space-y-4">
          <div className="text-gray-600">文档不存在</div>
          <Button onClick={() => navigate('/knowledge-base')} variant="outline">
            <ArrowLeft className="w-4 h-4 mr-2" />
            返回知识库
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between gap-6">
        <div className="flex-1 space-y-6">
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate('/knowledge-base')}
            className="w-fit rounded-xl hover-scale"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            返回知识库
          </Button>
          
          <div className="space-y-3">
            <h1 className="text-3xl font-bold text-gray-900">
              {isEditing ? "编辑文档" : document.title}
            </h1>
            {!isEditing && document.concepts && document.concepts.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {document.concepts.slice(0, 5).map((concept, index) => (
                  <Badge key={index} variant="outline" className="rounded-lg">
                    {concept}
                  </Badge>
                ))}
                {document.concepts.length > 5 && (
                  <Badge variant="secondary" className="rounded-lg">
                    +{document.concepts.length - 5} 更多
                  </Badge>
                )}
              </div>
            )}
            {isEditing && (
              <p className="text-gray-600">修改文档内容和元信息</p>
            )}
          </div>
        </div>
        
        {/* 操作按钮在右上角 */}
        <div className="flex items-start gap-3 mt-2">
          {isEditing ? (
            <>
              <Button
                variant="outline"
                onClick={() => setIsEditing(false)}
                className="rounded-xl"
              >
                取消
              </Button>
              <Button
                onClick={handleSave}
                className="rounded-xl"
              >
                <Save className="w-4 h-4 mr-2" />
                保存
              </Button>
            </>
          ) : (
            <>
              <Button
                onClick={() => setIsEditing(true)}
                className="rounded-xl"
              >
                <Edit className="w-4 h-4 mr-2" />
                编辑文档
              </Button>
              <Button variant="outline" className="rounded-xl">
                <FileText className="w-4 h-4 mr-2" />
                导出文档
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Main Content */}
        <div className="lg:col-span-3">
          <Card className="bg-card/70 backdrop-blur-sm border-border/50 shadow-xl rounded-3xl overflow-hidden">
            
            <CardContent className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">文档详情</h3>
              {isEditing ? (
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium text-gray-700 block mb-2">
                      文档标题
                    </label>
                    <Input
                      value={title}
                      onChange={(e) => setTitle(e.target.value)}
                      className="rounded-xl"
                      placeholder="文档标题"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-700 block mb-2">
                      标签
                    </label>
                    <TagInput
                      value={tags}
                      onChange={setTags}
                      placeholder="添加标签..."
                      className="rounded-xl"
                    />
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-700 block mb-2">
                      文档内容
                    </label>
                    <Textarea
                      value={content}
                      onChange={(e) => setContent(e.target.value)}
                      className="min-h-[500px] rounded-xl resize-none"
                      placeholder="在这里编写文档内容..."
                    />
                  </div>
                </div>
              ) : (
                <div className="prose prose-lg max-w-none">
                  <div className="bg-gray-50 rounded-xl p-6">
                    <pre className="whitespace-pre-wrap text-gray-800 font-sans leading-relaxed">
                      {content}
                    </pre>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Document Info */}
          <Card className="bg-card/70 backdrop-blur-sm border-border/50 shadow-xl rounded-3xl overflow-hidden">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg text-gray-900">文档信息</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-gray-500">字数</span>
                  <div className="font-medium">{document.word_count?.toLocaleString() || '未知'} 字</div>
                </div>
                <div>
                  <span className="text-gray-500">概念</span>
                  <div className="font-medium">{document.concept_count || 0} 个</div>
                </div>
                <div>
                  <span className="text-gray-500">链接</span>
                  <div className="font-medium">{(document.outbound_links?.length || 0) + (document.inbound_links?.length || 0)} 个</div>
                </div>
                <div>
                  <span className="text-gray-500">创建</span>
                  <div className="font-medium">{formatDate(document.created_at)}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Related Concepts */}
          <Card className="bg-card/70 backdrop-blur-sm border-border/50 shadow-xl rounded-3xl overflow-hidden">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg text-gray-900">相关概念</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {document.concepts && document.concepts.length > 0 ? (
                  document.concepts.map((concept) => (
                    <Badge 
                      key={concept} 
                      variant="outline" 
                      className="rounded-lg hover:bg-gray-50"
                    >
                      {concept}
                    </Badge>
                  ))
                ) : (
                  <div className="text-gray-500 text-sm">暂无相关概念</div>
                )}              </div>
            </CardContent>
          </Card>

        </div>
      </div>
    </div>
  );
};

export default DocumentDetail;