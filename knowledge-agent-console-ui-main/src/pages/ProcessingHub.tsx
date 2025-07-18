import { Button } from '@/components/ui/button';
import { useState } from 'react';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Spinner } from '@/components/ui/spinner';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';

import { useToast } from '@/hooks/use-toast';
import { apiClient, formatError, progressWebSocket, type ProcessingOptions, type ProcessingResponse } from '@/lib/api';
import { Copy, FileText, MessageSquare, Play, Square, X } from 'lucide-react';
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
  const [inputMode, setInputMode] = useState<'text' | 'conversation'>('text');
  const [content, setContent] = useState('');
  const [metadata, setMetadata] = useState({
    source: '',
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
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);

  const [forceUpdate, setForceUpdate] = useState(0);
  const { toast } = useToast();

  // 强制组件重新渲染的函数
  const triggerRerender = () => {
    setForceUpdate(prev => prev + 1);
    console.log('触发组件重新渲染:', forceUpdate + 1);
  };

  // 双链渲染函数
  const renderDoubleLinks = (text: string) => {
    const linkPattern = /\[\[([^\]]+)\]\]/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = linkPattern.exec(text)) !== null) {
      // 添加链接前的文本
      if (match.index > lastIndex) {
        parts.push(text.slice(lastIndex, match.index));
      }
      
      // 添加链接
      const conceptName = match[1];
      parts.push(
        <span 
          key={`link-${match.index}`}
          className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-700 rounded-md border border-blue-200 hover:bg-blue-200 cursor-pointer transition-colors"
          title={`概念: ${conceptName}`}
          onClick={() => {
            // 这里可以添加跳转到概念详情的逻辑
            toast({
              title: "概念链接",
              description: `点击了概念: ${conceptName}`,
            });
          }}
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
          </svg>
          {conceptName}
        </span>
      );
      
      lastIndex = match.index + match[0].length;
    }
    
    // 添加剩余的文本
    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }
    
    return parts.length > 0 ? parts : text;
  };

  const handleStartProcessing = async () => {
    if (!content.trim()) return;
    
    // 生成任务ID
    const taskId = crypto.randomUUID();
    setCurrentTaskId(taskId);
    
    console.log('开始处理内容');
    console.log('任务ID:', taskId);
    console.log('输入内容长度:', content.length);
    console.log('输入模式:', inputMode);
    console.log('处理选项:', options);
    
    setProcessing(true);
    setProgress(0);
    setError(null);
    setResult(null);
    setCurrentStatus('初始化处理...');
    setStopping(false);
    
    // 清空步骤，等待后端动态发送
    setProcessingSteps([]);
    console.log('清空步骤，等待后端返回步骤');

    try {
      // 准备请求数据
      const requestData = {
        content: content.trim(),
        type: inputMode,
        metadata: {
          source: metadata.source || 'user_input',
          tags: [],
          timestamp: new Date().toISOString()
        },
        options: options
      };
      
      console.log('📤 准备发送到后端的请求数据:', JSON.stringify(requestData, null, 2));

      // 设置进度监听器
      const progressListener = (message: any) => {
        // 如果消息是 progress_update 但 task_id 不匹配当前任务，直接忽略
        if (message.type === 'progress_update' && message.data && currentTaskId) {
          if (message.data.task_id && message.data.task_id !== currentTaskId) {
            return; // 不属于当前任务
          }
        }
        console.log('=== 接收到WebSocket消息 ===');
        console.log('消息内容:', JSON.stringify(message, null, 2));
        
        try {
          // 处理pong消息
          if (message.type === 'pong') {
            console.log('收到pong回复，连接正常');
            return;
          }
          
          // 处理停止确认消息
          if (message.type === 'processing_stopped') {
            console.log('收到停止确认消息:', message);
            setCurrentStatus('处理结束');
            setProcessing(false);
            setStopping(false);
            setCurrentTaskId(null);
            setResult(null);  // 清空结果
            setProgress(0);   // 重置进度
            setError('用户已停止');
            return;
          }
          
          // 检查消息类型和数据结构
          if (message.type === 'progress_update' && message.data) {
            const progressData = message.data;
            console.log('=== 处理进度更新 ===');
            console.log('当前阶段:', progressData.stage);
            console.log('当前步骤:', progressData.current_step);
            console.log('进度百分比:', progressData.progress_percent);
            console.log('完成步骤数:', progressData.completed_steps);
            console.log('总步骤数:', progressData.total_steps);
            console.log('工作者列表:', progressData.workers);
            console.log('任务复杂度:', progressData.complexity);
            
            // 更新进度百分比
            if (progressData.progress_percent !== undefined) {
              const newProgress = Math.max(0, Math.min(100, progressData.progress_percent));
              console.log('更新进度条:', newProgress + '%');
              setProgress(newProgress);
            } else if (progressData.completed_steps !== undefined && progressData.total_steps > 0) {
              const newProgress = Math.round((progressData.completed_steps / progressData.total_steps) * 100);
              console.log('计算进度条:', newProgress + '%');
              setProgress(newProgress);
            }
            
            // 更新当前状态
            if (progressData.current_step) {
              console.log('更新当前状态:', progressData.current_step);
              setCurrentStatus(progressData.current_step);
            }
            
            // 根据阶段和复杂度动态生成步骤
            if (progressData.stage) {
              console.log('=== 根据阶段信息更新步骤 ===');
              setProcessingSteps(prevSteps => {
                let newSteps = [...prevSteps];
                
                // 如果步骤列表为空，根据复杂度创建步骤
                if (newSteps.length === 0) {
                  const complexity = progressData.complexity;
                  console.log('根据复杂度创建步骤:', complexity);
                  
                  if (complexity === 'simple_task') {
                    newSteps = [
                      { step: 'analyzing', status: 'pending' as const, message: 'Agent识别中' },
                      { step: 'worker_processing', status: 'pending' as const, message: 'Agent处理中' },
                      { step: 'completed', status: 'pending' as const, message: '处理完成' }
                    ];
                  } else if (complexity === 'medium_task') {
                    newSteps = [
                      { step: 'analyzing', status: 'pending' as const, message: 'Agent识别中' },
                      { step: 'generating_workers', status: 'pending' as const, message: '生成工作者' },
                      { step: 'worker_processing', status: 'pending' as const, message: '工作者处理中' },
                      { step: 'finalizing', status: 'pending' as const, message: '完成处理' },
                      { step: 'completed', status: 'pending' as const, message: '处理完成' }
                    ];
                  } else { // complex_task
                    newSteps = [
                      { step: 'analyzing', status: 'pending' as const, message: 'Agent识别中' },
                      { step: 'generating_workers', status: 'pending' as const, message: '生成工作者' },
                      { step: 'worker_processing', status: 'pending' as const, message: '并行处理中' },
                      { step: 'finalizing', status: 'pending' as const, message: '完成处理' },
                      { step: 'completed', status: 'pending' as const, message: '处理完成' }
                    ];
                  }
                }
                
                // 更新步骤状态
                const stage = progressData.stage;
                const isCompleted = stage === 'completed';
                
                // 找到当前阶段对应的步骤
                const stageMapping = {
                  'planning': 0,
                  'tool_creation': 1,
                  'analyzing': 0,
                  'analysis': 0,
                  'generating_workers': 1,
                  'worker_processing': newSteps.length > 3 ? 2 : 1,
                  'extraction': newSteps.length > 3 ? 2 : 1,
                  'enhancement': newSteps.length > 3 ? 2 : 1,
                  'quality_control': newSteps.length - 2,
                  'synthesis': newSteps.length - 2,
                  'finalizing': newSteps.length - 2,
                  'completed': newSteps.length - 1
                };
                
                const currentStepIndex = stageMapping[stage] || 0;
                
                // 更新步骤状态
                newSteps.forEach((step, index) => {
                  if (index < currentStepIndex) {
                    // 之前的步骤标记为完成
                    newSteps[index] = { ...step, status: 'completed' };
                  } else if (index === currentStepIndex) {
                    // 当前步骤
                    newSteps[index] = {
                      ...step,
                      status: isCompleted ? 'completed' : 'processing',
                      message: progressData.current_step || step.message
                    };
                  } else {
                    // 后续步骤保持待处理
                    newSteps[index] = { ...step, status: 'pending' };
                  }
                });
                
                // 如果处理完成，标记所有步骤为完成并停止处理状态
                if (isCompleted) {
                  newSteps.forEach((step, index) => {
                    newSteps[index] = { ...step, status: 'completed' };
                  });
                  // 在下一个tick中停止处理状态，避免100%时还显示处理中
                  setTimeout(() => {
                    setProcessing(false);
                  }, 100);
                }
                
                // 如果有工作者信息，更新相关步骤的消息
                if (progressData.workers && Array.isArray(progressData.workers) && progressData.workers.length > 0) {
                  const workerStepIndex = stageMapping['worker_processing'] || 1;
                  if (newSteps[workerStepIndex]) {
                    newSteps[workerStepIndex] = {
                      ...newSteps[workerStepIndex],
                      message: `${progressData.workers.join(', ')} 处理中`
                    };
                  }
                }
                
                console.log('更新后的步骤:', newSteps.map(s => `${s.step}:${s.status}`));
                return newSteps;
              });
            }
            
            // 强制触发重新渲染
            triggerRerender();
            
          } else {
            console.log('收到其他类型消息:', message.type);
          }
          
        } catch (err) {
          console.error('处理WebSocket消息失败:', err);
          console.error('失败的消息:', message);
        }
      };

      try {
        // 使用已有的progressWebSocket连接
        console.log('🔌 正在连接进度WebSocket...');
        console.log('🔌 WebSocket当前状态:', progressWebSocket.isConnected() ? '已连接' : '未连接');
        
        // 添加进度监听器
        progressWebSocket.addListener(progressListener);
        console.log('👂 已添加WebSocket进度监听器');
        
        // 如果未连接，先连接
        if (!progressWebSocket.isConnected()) {
          console.log('🔌 开始连接WebSocket...');
          await progressWebSocket.connect();
          console.log('✅ 进度WebSocket连接成功');
        } else {
          console.log('✅ 进度WebSocket已连接');
        }

        // 订阅任务
        console.log('📡 订阅任务:', taskId);
        await progressWebSocket.subscribeTask(taskId);
        console.log('✅ 任务订阅成功');

        console.log('📡 开始发送处理请求到 POST /process...');
        console.log('📡 API基础URL:', 'http://localhost:8000');
        
        // 发送处理请求
        const response: ProcessingResponse = await apiClient.processContent(requestData);
        console.log('📥 收到后端响应:', JSON.stringify(response, null, 2));
        
        if (response.success) {
          console.log('✅ 后端处理成功');
          // 处理成功，标记所有步骤为完成
          setProcessingSteps(prevSteps => {
            const completedSteps = prevSteps.map(step => ({ ...step, status: 'completed' as const }));
            console.log('✅ 标记所有步骤为完成:', completedSteps);
            return completedSteps;
          });
          setCurrentStatus('处理完成！');
          setProgress(100);
          console.log('✅ 设置进度为100%');
          
          // 立即停止处理状态，避免100%时还显示处理中
          setProcessing(false);
          
          setResult({
            content: response.result?.structured_content || response.result?.content || '处理完成',
            statistics: {
              conceptCount: response.statistics?.conceptCount || response.statistics?.concept_count || 0,
              internalLinks: response.statistics?.internalLinks || response.statistics?.internal_links || 0,
              processingTime: response.statistics?.processingTime || response.statistics?.processing_time || 0,
              qualityScore: response.statistics?.qualityScore || response.statistics?.quality_score || 0
            }
          });
          
          toast({
            title: "处理成功",
            description: response.message || "内容已成功处理",
          });
        } else {
          console.error('❌ 后端处理失败:', response.errors);
          // 处理失败
          throw new Error(response.errors?.join(', ') || '处理失败');
        }
        
      } catch (err) {
        // 处理请求或WebSocket错误
        console.error('处理过程发生错误:', err);
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
        // 清理WebSocket监听器
        if (progressWebSocket && progressWebSocket.isConnected()) {
          // 取消订阅任务
          const prevTaskId = currentTaskId;
          if (prevTaskId) {
            console.log('📡 取消订阅任务:', prevTaskId);
            await progressWebSocket.unsubscribeTask(prevTaskId);
            console.log('✅ 任务取消订阅成功');
          }
          progressWebSocket.removeListener(progressListener);
        }
      }
    } catch (outerErr) {
      console.error('处理过程发生错误:', outerErr);
      const errorMessage = formatError(outerErr);
      setError(errorMessage);
      setCurrentStatus('处理失败');
      
      toast({
        title: "处理失败",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setProcessing(false);
      setStopping(false);
      setCurrentTaskId(null);
    }
  };

  const handleStopProcessing = async () => {
    if (!currentTaskId) return;
    
    console.log('用户请求停止处理，任务ID:', currentTaskId);
    setStopping(true);
    setCurrentStatus('正在停止处理...');
    
    try {
      // 发送停止信号
      if (progressWebSocket.isConnected()) {
        progressWebSocket.sendStopSignal(currentTaskId);
      }
      
      // 立即更新UI状态
      setProcessingSteps(prevSteps => {
        const newSteps = [...prevSteps];
        const processingIndex = newSteps.findIndex(step => step.status === 'processing');
        if (processingIndex !== -1) {
          newSteps[processingIndex] = {
            ...newSteps[processingIndex],
            status: 'error',
            message: '处理已停止'
          };
        }
        return newSteps;
      });
      
      // 重置状态
      setTimeout(() => {
        setProcessing(false);
        setStopping(false);
        setCurrentStatus('处理已停止');
        setError('用户主动停止了处理');
        setCurrentTaskId(null);
        setResult(null);  // 清空结果
        setProgress(0);   // 重置进度
        
        // 断开WebSocket连接
        if (progressWebSocket.isConnected()) {
          progressWebSocket.disconnect();
        }
      }, 1000);
      
      toast({
        title: "处理结束",
        description: "用户已停止处理",
        variant: "destructive",
      });
    } catch (error) {
      console.error('停止处理失败:', error);
      setStopping(false);
    }
  };

  const handleClearInput = () => {
    setContent('');
    setResult(null);
    setError(null);
    setProgress(0);
    setCurrentStatus('');
    setProcessingSteps([]);
    setCurrentTaskId(null);
    setStopping(false);
  };

  const handleCopyContent = async () => {
    if (!result?.content) return;
    
    try {
      await navigator.clipboard.writeText(result.content);
      toast({
        title: "复制成功",
        description: "内容已复制到剪贴板",
      });
    } catch (err) {
      // 如果现代API失败，尝试使用传统方法
      try {
        const textArea = document.createElement('textarea');
        textArea.value = result.content;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        
        toast({
          title: "复制成功",
          description: "内容已复制到剪贴板",
        });
      } catch (fallbackErr) {
        toast({
          title: "复制失败",
          description: "无法复制到剪贴板，请手动选择复制",
          variant: "destructive",
        });
      }
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">处理中心</h1>
          <p className="text-gray-600 mt-1 my-[15px]">智能处理文本内容，生成结构化知识</p>
        </div>
        <div className="flex items-center gap-4">
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
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="text" className="flex items-center gap-1">
                  <FileText className="h-4 w-4" />
                  文本
                </TabsTrigger>
                <TabsTrigger value="conversation" disabled className="flex items-center gap-1">
                  <MessageSquare className="h-4 w-4" />
                  对话
                </TabsTrigger>
              </TabsList>

              <TabsContent value="text" className="mt-4">
                <Textarea placeholder="请输入要处理的文本内容..." className="min-h-[200px] resize-none" value={content} onChange={e => setContent(e.target.value)} />
              </TabsContent>

              {/* 对话模式暂未开放 */}
              {false && (
                <TabsContent value="conversation" className="mt-4">
                  <Textarea placeholder="请粘贴对话记录..." className="min-h-[200px] resize-none" value={content} onChange={e => setContent(e.target.value)} />
                </TabsContent>
              )}
            </Tabs>

            {/* 操作按钮 */}
            <div className="flex gap-2 pt-2">
              <Button onClick={handleStartProcessing} disabled={!content.trim() || processing} className="flex-1">
                {processing ? <Spinner className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
                {processing ? '处理中...' : '开始处理'}
              </Button>
              
              <Button variant="outline" onClick={handleClearInput} disabled={processing}>
                <X className="mr-2 h-4 w-4" />
                清空
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* 处理进度 */}
        {processing && (
          <Card className="bg-card/70 backdrop-blur-sm border-border/50 shadow-xl rounded-3xl overflow-hidden">
            <CardContent className="p-6">
              {/* 标题行：图标、标题、百分比、停止按钮 */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Spinner className="h-5 w-5 animate-spin text-blue-600" />
                  <span className="font-semibold text-gray-900">处理进度</span>
                  <span className="text-blue-600 font-bold text-lg">{Math.round(progress)}%</span>
                </div>
                
                <Button 
                  variant="destructive" 
                  size="sm"
                  onClick={handleStopProcessing}
                  disabled={stopping}
                  className="h-8 px-3"
                >
                  <Square className="mr-1 h-3 w-3" />
                  {stopping ? '停止中' : '停止'}
                </Button>
              </div>
              
              {/* 发光流水进度条 */}
              <div className="relative w-full h-4 mb-4">
                {/* 进度条轨道 */}
                <div className="w-full h-4 bg-gray-200 rounded-full shadow-inner"></div>
                
                {/* 外围激光脉冲 - 横向扫描 */}
                <div 
                  className="absolute top-0 left-0 h-4 rounded-full overflow-hidden"
                  style={{ 
                    width: `${Math.max(0, Math.min(100, progress))}%`,
                    background: 'transparent',
                    boxShadow: '0 0 8px rgba(59, 130, 246, 0.3)'
                  }}
                >
                  {/* 横向脉冲光带 */}
                  <div 
                    className="absolute top-0 h-4 rounded-full"
                    style={{ 
                      width: '30%',
                      background: 'linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.8), rgba(59, 130, 246, 1), rgba(59, 130, 246, 0.8), transparent)',
                      boxShadow: '0 0 20px rgba(59, 130, 246, 0.8), 0 0 40px rgba(59, 130, 246, 0.4)',
                      animation: 'horizontal-pulse 2s ease-in-out infinite'
                    }}
                  />
                </div>
                
                                 {/* 进度条主体 - 蓝色基础 */}
                <div 
                  className="absolute top-0 left-0 h-4 rounded-full transition-all duration-500 ease-out overflow-hidden"
                  style={{ 
                    width: `${Math.max(0, Math.min(100, progress))}%`,
                    background: 'linear-gradient(90deg, #1e40af, #3b82f6, #60a5fa)'
                  }}
                >
                  {/* 纯蓝色透明度水波纹 */}
                  <div 
                    className="absolute top-0 left-0 w-full h-full"
                    style={{ 
                      background: 'linear-gradient(90deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.6), rgba(59, 130, 246, 0.3), rgba(59, 130, 246, 0.7), rgba(59, 130, 246, 0.2))',
                      backgroundSize: '150% 100%',
                      animation: 'water-ripple-internal 2.5s ease-in-out infinite'
                    }}
                  />
                </div>
              </div>
              
              {/* 当前状态 */}
              <div className="text-sm text-gray-600">
                {currentStatus || '正在处理...'}
              </div>
            </CardContent>
          </Card>
        )}

        {/* 错误信息
        {error && <Card className="bg-red-50 border-red-200 shadow-xl rounded-3xl overflow-hidden">
            <CardHeader>
              <CardTitle className="text-red-800">处理结束</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-red-700">{error}</p>
            </CardContent>
          </Card>} */}

        {/* 处理结果 */}
        {result && <Card className="bg-card/70 backdrop-blur-sm border-border/50 shadow-xl rounded-3xl overflow-hidden">
            <CardHeader>
              <CardTitle>处理结果</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="bg-gray-50 p-4 rounded-lg text-sm overflow-x-auto">
                {result.content}
              </pre>

              <div className="mt-4 flex gap-2">
                <Button variant="outline" onClick={handleCopyContent}>
                  <Copy className="mr-2 h-4 w-4" />
                  复制内容
                </Button>
              </div>
            </CardContent>
          </Card>}
      </div>
    </div>
  );
};
export default ProcessingHub;