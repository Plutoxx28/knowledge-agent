import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Spinner } from '@/components/ui/spinner';
import { StatCard } from '@/components/ui/stat-card';
import { TagInput } from '@/components/ui/tag-input';
import { Play, X, Settings, Hash, Link, Clock, Star, Save, Copy, Download, ExternalLink, ChevronDown, ChevronRight, Upload, FileText, MessageSquare, Globe } from 'lucide-react';
import { apiClient, progressWebSocket, formatError, type ProcessingOptions, type ProcessingResponse } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
interface ProcessingResult {
  content: string;
  statistics: {
    conceptCount: number;
    internalLinks: number;
    processingTime: number;
    qualityScore: number;
  };
}
const ProcessingHub = () => {
  const [inputMode, setInputMode] = useState<'text' | 'conversation' | 'url' | 'file'>('text');
  const [content, setContent] = useState('');
  const [metadata, setMetadata] = useState({
    source: '',
    topic: '',
    tags: [] as string[]
  });
  const [options, setOptions] = useState<ProcessingOptions>({
    strategy: 'standard',
    enableLinking: true,
    generateSummary: true,
    extractConcepts: true
  });
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<ProcessingResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentStatus, setCurrentStatus] = useState<string>('');
  const [processingSteps, setProcessingSteps] = useState<Array<{step: string, status: 'pending' | 'processing' | 'completed' | 'error', message?: string}>>([]);
  const { toast } = useToast();
  const handleStartProcessing = async () => {
    if (!content.trim()) return;
    
    setProcessing(true);
    setProgress(0);
    setError(null);
    setResult(null);
    setCurrentStatus('初始化处理...');
    
    // 初始化处理步骤
    const initialSteps = [
      { step: 'analyzing', status: 'pending' as const, message: 'Agent识别中' },
      { step: 'generating_workers', status: 'pending' as const, message: '生成工作者' },
      { step: 'worker_processing', status: 'pending' as const, message: '工作者处理中' },
      { step: 'finalizing', status: 'pending' as const, message: '完成处理' }
    ];
    setProcessingSteps(initialSteps);

    try {
      // 准备请求数据
      const requestData = {
        content: content.trim(),
        type: inputMode,
        metadata: {
          source: metadata.source || 'user_input',
          topic: metadata.topic,
          tags: metadata.tags,
          timestamp: new Date().toISOString()
        },
        options: options
      };

      // 连接WebSocket接收进度更新
      const ws = new WebSocket('ws://localhost:8000/ws/progress');
      let currentProgressStep = 0;
      
      ws.onopen = () => {
        console.log('WebSocket连接已建立');
      };
      
      ws.onmessage = (event) => {
        try {
          const progressData = JSON.parse(event.data);
          console.log('收到进度更新:', progressData);
          
          // 更新当前状态
          if (progressData.message) {
            setCurrentStatus(progressData.message);
          }
          
          // 更新进度百分比
          if (progressData.step !== undefined) {
            const progressPercentage = (progressData.step / 5) * 100;
            setProgress(progressPercentage);
          }
          
          // 更新处理步骤状态
          setProcessingSteps(prevSteps => {
            const newSteps = [...prevSteps];
            
            // 根据进度数据更新步骤状态
            if (progressData.stage) {
              const stepIndex = newSteps.findIndex(step => step.step === progressData.stage);
              if (stepIndex !== -1) {
                newSteps[stepIndex] = {
                  ...newSteps[stepIndex],
                  status: 'processing',
                  message: progressData.message || newSteps[stepIndex].message
                };
                
                // 将之前的步骤标记为完成
                for (let i = 0; i < stepIndex; i++) {
                  if (newSteps[i].status !== 'completed') {
                    newSteps[i].status = 'completed';
                  }
                }
              }
            }
            
            return newSteps;
          });
          
        } catch (err) {
          console.error('解析进度数据失败:', err);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket错误:', error);
      };
      
      ws.onclose = () => {
        console.log('WebSocket连接已关闭');
      };

      // 发送处理请求
      const response: ProcessingResponse = await apiClient.processContent(requestData);
      
      // 关闭WebSocket连接
      ws.close();
      
      if (response.success) {
        // 处理成功，标记所有步骤为完成
        setProcessingSteps(prevSteps => 
          prevSteps.map(step => ({ ...step, status: 'completed' as const }))
        );
        setCurrentStatus('处理完成！');
        setProgress(100);
        
        setResult({
          content: response.result?.structured_content || response.result?.content || '处理完成',
          statistics: response.statistics || {
            conceptCount: 0,
            internalLinks: 0,
            processingTime: 0,
            qualityScore: 0
          }
        });
        
        toast({
          title: "处理成功",
          description: response.message || "内容已成功处理",
        });
      } else {
        // 处理失败
        throw new Error(response.errors?.join(', ') || '处理失败');
      }
    } catch (err) {
      const errorMessage = formatError(err);
      setError(errorMessage);
      setCurrentStatus('处理失败');
      
      // 标记当前处理步骤为错误状态
      setProcessingSteps(prevSteps => {
        const newSteps = [...prevSteps];
        const processingIndex = newSteps.findIndex(step => step.status === 'processing');
        if (processingIndex !== -1) {
          newSteps[processingIndex].status = 'error';
        }
        return newSteps;
      });
      
      toast({
        title: "处理失败",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setProcessing(false);
    }
  };
  const handleClearInput = () => {
    setContent('');
    setMetadata({
      source: '',
      topic: '',
      tags: []
    });
    setResult(null);
    setError(null);
    setProgress(0);
    setCurrentStatus('');
    setProcessingSteps([]);
  };
  const handleCheckboxChange = (field: keyof ProcessingOptions) => (checked: boolean) => {
    setOptions(prev => ({
      ...prev,
      [field]: checked
    }));
  };

  const handleFileUpload = async (file: File) => {
    if (!file) return;
    
    const allowedTypes = ['.md', '.txt', '.doc', '.docx'];
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
      toast({
        title: "文件格式不支持",
        description: `仅支持 MD, TXT, DOC, DOCX 格式的文件，当前文件类型：${fileExtension}`,
        variant: "destructive",
      });
      return;
    }
    
    setProcessing(true);
    setError(null);
    setResult(null);
    setCurrentStatus('正在上传文件...');
    setProgress(20);
    
    // 初始化文件上传步骤
    const uploadSteps = [
      { step: 'uploading', status: 'processing' as const, message: '正在上传文件' },
      { step: 'analyzing', status: 'pending' as const, message: '分析文件内容' },
      { step: 'processing', status: 'pending' as const, message: '处理文件' },
      { step: 'finalizing', status: 'pending' as const, message: '完成处理' }
    ];
    setProcessingSteps(uploadSteps);
    
    try {
      // 直接使用 API 上传文件
      const response: ProcessingResponse = await apiClient.uploadFile(file);
      
      if (response.success) {
        // 上传成功，标记所有步骤为完成
        setProcessingSteps(prevSteps => 
          prevSteps.map(step => ({ ...step, status: 'completed' as const }))
        );
        setCurrentStatus('文件处理完成！');
        setProgress(100);
        
        setResult({
          content: response.result?.structured_content || response.result?.content || '文件处理完成',
          statistics: response.statistics || {
            conceptCount: 0,
            internalLinks: 0,
            processingTime: 0,
            qualityScore: 0
          }
        });
        
        // 更新元数据
        setMetadata(prev => ({
          ...prev,
          source: file.name
        }));
        
        toast({
          title: "文件上传成功",
          description: response.message || `文件 ${file.name} 已成功处理`,
        });
      } else {
        throw new Error(response.errors?.join(', ') || '文件处理失败');
      }
    } catch (err) {
      const errorMessage = formatError(err);
      setError(errorMessage);
      setCurrentStatus('文件处理失败');
      
      // 标记当前处理步骤为错误状态
      setProcessingSteps(prevSteps => {
        const newSteps = [...prevSteps];
        const processingIndex = newSteps.findIndex(step => step.status === 'processing');
        if (processingIndex !== -1) {
          newSteps[processingIndex].status = 'error';
        }
        return newSteps;
      });
      
      toast({
        title: "文件上传失败",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setProcessing(false);
    }
  };
  return <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">处理中心</h1>
          <p className="text-gray-600 mt-1 my-[15px]">智能处理各种内容，生成结构化知识</p>
        </div>
        
      </div>

      <div className="space-y-6">
        {/* 输入区域 */}
        <Card className="bg-card/70 backdrop-blur-sm border-border/50 shadow-xl rounded-3xl overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              内容输入
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Tabs value={inputMode} onValueChange={value => setInputMode(value as any)}>
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="text" className="flex items-center gap-1">
                  <FileText className="h-4 w-4" />
                  文本
                </TabsTrigger>
                <TabsTrigger value="conversation" className="flex items-center gap-1">
                  <MessageSquare className="h-4 w-4" />
                  对话
                </TabsTrigger>
                <TabsTrigger value="url" className="flex items-center gap-1">
                  <Globe className="h-4 w-4" />
                  URL
                </TabsTrigger>
                <TabsTrigger value="file" className="flex items-center gap-1">
                  <Upload className="h-4 w-4" />
                  文件
                </TabsTrigger>
              </TabsList>

              <TabsContent value="text" className="mt-4">
                <Textarea placeholder="请输入要处理的文本内容..." className="min-h-[200px] resize-none" value={content} onChange={e => setContent(e.target.value)} />
              </TabsContent>

              <TabsContent value="conversation" className="mt-4">
                <Textarea placeholder="请粘贴对话记录..." className="min-h-[200px] resize-none" value={content} onChange={e => setContent(e.target.value)} />
              </TabsContent>

              <TabsContent value="url" className="mt-4">
                <div className="space-y-2">
                  <Input placeholder="输入要处理的URL..." value={content} onChange={e => setContent(e.target.value)} />
                  
                </div>
              </TabsContent>

              <TabsContent value="file" className="mt-4">
                <div 
                  className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-gray-400 transition-colors"
                  onClick={() => document.getElementById('file-input')?.click()}
                  onDragOver={(e) => {
                    e.preventDefault();
                    e.currentTarget.classList.add('border-blue-400', 'bg-blue-50');
                  }}
                  onDragLeave={(e) => {
                    e.preventDefault();
                    e.currentTarget.classList.remove('border-blue-400', 'bg-blue-50');
                  }}
                  onDrop={(e) => {
                    e.preventDefault();
                    e.currentTarget.classList.remove('border-blue-400', 'bg-blue-50');
                    const files = Array.from(e.dataTransfer.files);
                    handleFileUpload(files[0]);
                  }}
                >
                  <Upload className="mx-auto h-12 w-12 text-gray-400" />
                  <p className="mt-2 text-sm text-gray-600">
                    点击上传或拖拽文件到这里
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    支持 MD, TXT, DOC, DOCX 格式
                  </p>
                  <input
                    id="file-input"
                    type="file"
                    accept=".md,.txt,.doc,.docx"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleFileUpload(file);
                    }}
                  />
                </div>
              </TabsContent>
            </Tabs>

            {/* 处理配置 */}
            <Collapsible>
              <CollapsibleTrigger className="flex items-center gap-2 text-sm font-medium hover:text-blue-600">
                <Settings className="h-4 w-4" />
                处理配置
                <ChevronDown className="h-4 w-4" />
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-3 space-y-4 border-t pt-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-sm font-medium">处理策略</Label>
                    <Select value={options.strategy} onValueChange={value => setOptions(prev => ({
                    ...prev,
                    strategy: value as any
                  }))}>
                      <SelectTrigger className="mt-1">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="standard">标准处理</SelectItem>
                        <SelectItem value="hierarchical">层次化处理</SelectItem>
                        <SelectItem value="streaming">流式处理</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <Checkbox id="linking" checked={options.enableLinking} onCheckedChange={handleCheckboxChange('enableLinking')} />
                      <Label htmlFor="linking" className="text-sm">启用概念链接</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox id="summary" checked={options.generateSummary} onCheckedChange={handleCheckboxChange('generateSummary')} />
                      <Label htmlFor="summary" className="text-sm">生成摘要</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox id="concepts" checked={options.extractConcepts} onCheckedChange={handleCheckboxChange('extractConcepts')} />
                      <Label htmlFor="concepts" className="text-sm">提取概念</Label>
                    </div>
                  </div>
                </div>
              </CollapsibleContent>
            </Collapsible>

            {/* 元数据输入 */}
            <div className="grid grid-cols-3 gap-4">
              <div>
                <Label className="text-sm font-medium">来源</Label>
                <Input placeholder="内容来源" className="mt-1" value={metadata.source} onChange={e => setMetadata(prev => ({
                ...prev,
                source: e.target.value
              }))} />
              </div>
              <div>
                <Label className="text-sm font-medium">主题</Label>
                <Input placeholder="主题分类" className="mt-1" value={metadata.topic} onChange={e => setMetadata(prev => ({
                ...prev,
                topic: e.target.value
              }))} />
              </div>
              <div>
                <Label className="text-sm font-medium">标签</Label>
                <TagInput placeholder="添加标签" value={metadata.tags} onChange={tags => setMetadata(prev => ({
                ...prev,
                tags
              }))} />
              </div>
            </div>

            {/* 操作按钮 */}
            <div className="flex gap-2 pt-2">
              <Button onClick={handleStartProcessing} disabled={!content.trim() || processing} className="flex-1">
                {processing ? <Spinner className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
                {processing ? '处理中...' : '开始处理'}
              </Button>
              <Button variant="outline" onClick={handleClearInput}>
                <X className="mr-2 h-4 w-4" />
                清空
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* 处理进度 */}
        {processing && <Card className="bg-card/70 backdrop-blur-sm border-border/50 shadow-xl rounded-3xl overflow-hidden">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Spinner className="h-5 w-5 animate-spin text-blue-600" />
                处理进度
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* 总体进度 */}
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="font-medium">{currentStatus || '正在初始化...'}</span>
                  <span className="text-blue-600 font-semibold">{Math.round(progress)}%</span>
                </div>
                <Progress value={progress} className="h-3" />
              </div>
              
              {/* 详细步骤 */}
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-gray-700">处理步骤</h4>
                <div className="space-y-2">
                  {processingSteps.map((step, index) => (
                    <div key={step.step} className="flex items-center gap-3 p-2 rounded-lg bg-gray-50/50">
                      <div className="flex-shrink-0">
                        {step.status === 'completed' && (
                          <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center">
                            <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          </div>
                        )}
                        {step.status === 'processing' && (
                          <div className="w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center">
                            <Spinner className="w-3 h-3 text-white" />
                          </div>
                        )}
                        {step.status === 'pending' && (
                          <div className="w-5 h-5 rounded-full bg-gray-300"></div>
                        )}
                        {step.status === 'error' && (
                          <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center">
                            <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </div>
                        )}
                      </div>
                      <div className="flex-1">
                        <div className={`text-sm font-medium ${
                          step.status === 'completed' ? 'text-green-700' :
                          step.status === 'processing' ? 'text-blue-700' :
                          step.status === 'error' ? 'text-red-700' :
                          'text-gray-500'
                        }`}>
                          {step.message}
                        </div>
                        {step.status === 'processing' && (
                          <div className="text-xs text-gray-600 mt-1">正在执行中...</div>
                        )}
                      </div>
                      {step.status === 'processing' && (
                        <div className="flex-shrink-0">
                          <div className="animate-pulse w-2 h-2 bg-blue-500 rounded-full"></div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>}

        {/* 错误信息 */}
        {error && <Card className="bg-red-50 border-red-200 shadow-xl rounded-3xl overflow-hidden">
            <CardHeader>
              <CardTitle className="text-red-800">处理错误</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-red-700">{error}</p>
            </CardContent>
          </Card>}

        {/* 处理结果 */}
        {result && <Card className="bg-card/70 backdrop-blur-sm border-border/50 shadow-xl rounded-3xl overflow-hidden">
            <CardHeader>
              <CardTitle>处理结果</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="preview">
                <TabsList>
                  <TabsTrigger value="preview">预览</TabsTrigger>
                  <TabsTrigger value="raw">原始内容</TabsTrigger>
                  <TabsTrigger value="stats">统计信息</TabsTrigger>
                </TabsList>

                <TabsContent value="preview" className="mt-4">
                  <div className="prose max-w-none">
                    <div className="whitespace-pre-wrap">{result.content}</div>
                  </div>
                </TabsContent>

                <TabsContent value="raw" className="mt-4">
                  <pre className="bg-gray-50 p-4 rounded-lg text-sm overflow-x-auto">
                    {result.content}
                  </pre>
                </TabsContent>

                <TabsContent value="stats" className="mt-4">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <StatCard title="概念数量" value={result.statistics.conceptCount} icon={<Hash className="h-4 w-4" />} />
                    <StatCard title="内部链接" value={result.statistics.internalLinks} icon={<Link className="h-4 w-4" />} />
                    <StatCard title="处理时长" value={`${result.statistics.processingTime}s`} icon={<Clock className="h-4 w-4" />} />
                    <StatCard title="质量评分" value={`${result.statistics.qualityScore}/100`} icon={<Star className="h-4 w-4" />} />
                  </div>
                </TabsContent>
              </Tabs>

              <div className="mt-4 flex gap-2">
                <Button>
                  <Save className="mr-2 h-4 w-4" />
                  保存到知识库
                </Button>
                <Button variant="outline">
                  <Copy className="mr-2 h-4 w-4" />
                  复制内容
                </Button>
                <Button variant="outline">
                  <Download className="mr-2 h-4 w-4" />
                  导出文件
                </Button>
                <Button variant="outline">
                  <ExternalLink className="mr-2 h-4 w-4" />
                  查看链接
                </Button>
              </div>
            </CardContent>
          </Card>}
      </div>
    </div>;
};
export default ProcessingHub;