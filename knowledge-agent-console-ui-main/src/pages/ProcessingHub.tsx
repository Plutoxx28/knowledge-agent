import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Spinner } from '@/components/ui/spinner';
import { StatCard } from '@/components/ui/stat-card';

import { Play, X, Hash, Link, Clock, Star, Save, Copy, Download, ExternalLink, ChevronRight, Upload, FileText, MessageSquare, Globe } from 'lucide-react';
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
    
    console.log('🚀 ===== 开始处理内容 =====');
    console.log('📝 输入内容长度:', content.length);
    console.log('📝 输入模式:', inputMode);
    console.log('⚙️ 处理选项:', options);
    
    setProcessing(true);
    setProgress(0);
    setError(null);
    setResult(null);
    setCurrentStatus('初始化处理...');
    
    // 清空步骤，等待后端动态发送
    setProcessingSteps([]);
    console.log('🔄 清空步骤，等待后端返回步骤');

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
        console.log('=== 接收到WebSocket消息 ===');
        console.log('消息内容:', JSON.stringify(message, null, 2));
        
        try {
          // 处理pong消息
          if (message.type === 'pong') {
            console.log('收到pong回复，连接正常');
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
                const isCompleted = stage === 'completed' || progressData.progress_percent === 100;
                
                // 找到当前阶段对应的步骤
                const stageMapping = {
                  'analyzing': 0,
                  'generating_workers': 1,
                  'worker_processing': newSteps.length > 3 ? 2 : 1,
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
                
                // 如果处理完成，标记所有步骤为完成
                if (isCompleted) {
                  newSteps.forEach((step, index) => {
                    newSteps[index] = { ...step, status: 'completed' };
                  });
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
    }
  };
  const handleClearInput = () => {
    setContent('');
    setResult(null);
    setError(null);
    setProgress(0);
    setCurrentStatus('');
    setProcessingSteps([]);
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
    setProgress(0);
    
    // 清空步骤，等待后端动态发送
    setProcessingSteps([]);
    console.log('开始文件上传，等待后端返回步骤');
    
    try {
      // 设置进度监听器 - 复用之前的progressListener逻辑
      const progressListener = (message: any) => {
        console.log('=== 文件上传接收到WebSocket消息 ===');
        console.log('消息内容:', JSON.stringify(message, null, 2));
        
        try {
          if (message.type === 'pong') {
            console.log('收到pong回复，连接正常');
            return;
          }
          
          if (message.type === 'progress_update' && message.data) {
            const progressData = message.data;
            console.log('=== 处理文件上传进度更新 ===');
            console.log('当前阶段:', progressData.stage);
            console.log('当前步骤:', progressData.current_step);
            console.log('进度百分比:', progressData.progress_percent);
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
            
            // 根据阶段和复杂度动态生成步骤（与文本处理相同的逻辑）
            if (progressData.stage) {
              console.log('=== 根据阶段信息更新文件上传步骤 ===');
              setProcessingSteps(prevSteps => {
                let newSteps = [...prevSteps];
                
                // 如果步骤列表为空，根据复杂度创建步骤
                if (newSteps.length === 0) {
                  const complexity = progressData.complexity;
                  console.log('根据复杂度创建文件处理步骤:', complexity);
                  
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
                const isCompleted = stage === 'completed' || progressData.progress_percent === 100;
                
                // 找到当前阶段对应的步骤
                const stageMapping = {
                  'analyzing': 0,
                  'generating_workers': 1,
                  'worker_processing': newSteps.length > 3 ? 2 : 1,
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
                
                // 如果处理完成，标记所有步骤为完成
                if (isCompleted) {
                  newSteps.forEach((step, index) => {
                    newSteps[index] = { ...step, status: 'completed' };
                  });
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
                
                console.log('更新后的文件处理步骤:', newSteps.map(s => `${s.step}:${s.status}`));
                return newSteps;
              });
            }
            
            // 强制触发重新渲染
            triggerRerender();
          }
        } catch (err) {
          console.error('处理WebSocket消息失败:', err);
          console.error('失败的消息:', message);
        }
      };

      try {
        // 连接WebSocket监听进度
        console.log('正在连接进度WebSocket...');
        progressWebSocket.addListener(progressListener);
        
        if (!progressWebSocket.isConnected()) {
          await progressWebSocket.connect();
          console.log('进度WebSocket连接成功');
        }

        // 上传文件
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
        // 清理WebSocket监听器
        if (progressWebSocket && progressWebSocket.isConnected()) {
          progressWebSocket.removeListener(progressListener);
        }
      }
    } catch (outerErr) {
      console.error('文件上传过程发生错误:', outerErr);
      const errorMessage = formatError(outerErr);
      setError(errorMessage);
      setCurrentStatus('文件处理失败');
      
      toast({
        title: "文件上传失败",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setProcessing(false);
    }
  };



  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">处理中心</h1>
          <p className="text-gray-600 mt-1 my-[15px]">智能处理各种内容，生成结构化知识</p>
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
        {processing && <Card className="bg-card/70 backdrop-blur-sm border-border/50 shadow-xl rounded-3xl overflow-hidden processing-card">
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
                  {/* <span className="font-medium">处理进度</span> */}
                  <span className="text-blue-600 font-semibold">{Math.round(progress)}%</span>
                </div>
                <div className="progress-enhanced h-3 w-full overflow-hidden rounded-full">
                  <div 
                    className="progress-bar h-full"
                    style={{ width: `${Math.max(0, Math.min(100, progress))}%` }}
                  />
                </div>
              </div>
              
              {/* 简化的当前状态 */}
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-gray-700">当前状态</h4>
                <div className="flex items-center gap-3 p-4 rounded-lg bg-blue-50 border border-blue-200">
                  <Spinner className="w-5 h-5 text-blue-600" />
                  <div className="flex-1">
                    <div className="text-sm font-medium text-blue-700">
                      {currentStatus || '正在处理...'}
                    </div>
                  </div>
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
              <pre className="bg-gray-50 p-4 rounded-lg text-sm overflow-x-auto">
                {result.content}
              </pre>

              <div className="mt-4 flex gap-2">
                <Button variant="outline">
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