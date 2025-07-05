import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Checkbox } from '@/components/ui/checkbox';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Slider } from '@/components/ui/slider';
import { Save, Eye, EyeOff, Zap } from 'lucide-react';
import { Spinner } from '@/components/ui/spinner';

export default function Settings() {
  // API设置
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [modelName, setModelName] = useState('google/gemini-2.5-pro');
  const [requestTimeout, setRequestTimeout] = useState(60);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{success: boolean; message: string} | null>(null);
  
  // 处理设置
  const [defaultStrategy, setDefaultStrategy] = useState('standard');
  const [enableLinking, setEnableLinking] = useState(true);
  const [maxChunkSize, setMaxChunkSize] = useState([2000]);
  
  // 存储设置
  const [knowledgeBasePath, setKnowledgeBasePath] = useState('./knowledge');
  const [autoSave, setAutoSave] = useState(true);
  const [backupEnabled, setBackupEnabled] = useState(true);
  
  const [saving, setSaving] = useState(false);

  const testApiConnection = async () => {
    setTesting(true);
    try {
      // 模拟API测试
      await new Promise(resolve => setTimeout(resolve, 2000));
      setTestResult({ success: true, message: 'API连接测试成功！' });
    } catch (error) {
      setTestResult({ success: false, message: 'API连接失败，请检查配置。' });
    } finally {
      setTesting(false);
    }
  };

  const saveSettings = async () => {
    setSaving(true);
    try {
      // 模拟保存设置
      await new Promise(resolve => setTimeout(resolve, 1000));
      // 这里应该调用实际的API保存设置
      console.log('Settings saved');
    } finally {
      setSaving(false);
    }
  };

  const resetSettings = () => {
    setApiKey('');
    setModelName('google/gemini-2.5-pro');
    setRequestTimeout(60);
    setDefaultStrategy('standard');
    setEnableLinking(true);
    setMaxChunkSize([2000]);
    setKnowledgeBasePath('./knowledge');
    setAutoSave(true);
    setBackupEnabled(true);
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2 text-left">系统设置</h1>
        <p className="text-lg text-gray-600 text-left">配置Knowledge Agent的运行参数</p>
      </div>

      <div className="max-w-4xl">
        <Tabs defaultValue="api" className="space-y-6">
          <TabsList className="grid grid-cols-3 w-full">
            <TabsTrigger value="api">API配置</TabsTrigger>
            <TabsTrigger value="processing">处理配置</TabsTrigger>
            <TabsTrigger value="storage">存储设置</TabsTrigger>
          </TabsList>
          
          <TabsContent value="api" className="space-y-6">
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader className="pb-6">
                <CardTitle className="text-xl">OpenRouter API 配置</CardTitle>
                <CardDescription className="text-base">
                  配置AI模型的API连接信息
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-3">
                  <Label htmlFor="api-key" className="text-sm font-semibold text-gray-800">API 密钥</Label>
                  <div className="relative">
                    <Input
                      id="api-key"
                      type={showApiKey ? "text" : "password"}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="输入你的 OpenRouter API 密钥"
                      className="pr-12"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                      onClick={() => setShowApiKey(!showApiKey)}
                    >
                      {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <Label htmlFor="model" className="text-sm font-semibold text-gray-800">模型选择</Label>
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
                
                <div className="space-y-3">
                  <Label htmlFor="timeout" className="text-sm font-semibold text-gray-800">请求超时 (秒)</Label>
                  <Input
                    id="timeout"
                    type="number"
                    value={requestTimeout}
                    onChange={(e) => setRequestTimeout(Number(e.target.value))}
                    min={10}
                    max={300}
                  />
                </div>
                
                <div className="pt-6 border-t border-gray-100">
                  <Button onClick={testApiConnection} disabled={testing} className="min-w-[120px]">
                    {testing ? <Spinner className="mr-2 h-4 w-4" /> : <Zap className="mr-2 h-4 w-4" />}
                    测试连接
                  </Button>
                  {testResult && (
                    <div className={`mt-3 p-3 rounded-md text-sm ${
                      testResult.success 
                        ? 'bg-green-50 text-green-700 border border-green-200' 
                        : 'bg-red-50 text-red-700 border border-red-200'
                    }`}>
                      {testResult.message}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="processing" className="space-y-6">
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader className="pb-6">
                <CardTitle className="text-xl">处理配置</CardTitle>
                <CardDescription className="text-base">
                  配置内容处理的默认参数
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-8">
                <div className="space-y-6">
                  <Label className="text-base font-semibold text-gray-800">默认处理策略</Label>
                  <RadioGroup value={defaultStrategy} onValueChange={setDefaultStrategy} className="space-y-3">
                    <div className="flex items-center space-x-4 p-4 rounded-lg hover:bg-gray-50 transition-colors border border-transparent hover:border-gray-200">
                      <RadioGroupItem value="standard" id="standard" />
                      <div className="space-y-1 flex-1">
                        <Label htmlFor="standard" className="text-sm font-medium cursor-pointer text-gray-900">标准处理</Label>
                        <p className="text-sm text-gray-600 leading-relaxed">使用标准算法进行文档处理</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4 p-4 rounded-lg hover:bg-gray-50 transition-colors border border-transparent hover:border-gray-200">
                      <RadioGroupItem value="hierarchical" id="hierarchical" />
                      <div className="space-y-1 flex-1">
                        <Label htmlFor="hierarchical" className="text-sm font-medium cursor-pointer text-gray-900">层次化处理</Label>
                        <p className="text-sm text-gray-600 leading-relaxed">按层次结构组织和处理内容</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4 p-4 rounded-lg hover:bg-gray-50 transition-colors border border-transparent hover:border-gray-200">
                      <RadioGroupItem value="streaming" id="streaming" />
                      <div className="space-y-1 flex-1">
                        <Label htmlFor="streaming" className="text-sm font-medium cursor-pointer text-gray-900">流式处理</Label>
                        <p className="text-sm text-gray-600 leading-relaxed">实时流式处理大型文档</p>
                      </div>
                    </div>
                  </RadioGroup>
                </div>
                
                <div className="pt-6 border-t border-gray-100">
                  <div className="flex items-center space-x-4 p-4 rounded-lg hover:bg-gray-50 transition-colors border border-transparent hover:border-gray-200">
                    <Checkbox 
                      id="enable-linking" 
                      checked={enableLinking}
                      onCheckedChange={(checked) => setEnableLinking(checked === true)}
                    />
                    <div className="space-y-1 flex-1">
                      <Label htmlFor="enable-linking" className="text-sm font-medium cursor-pointer text-gray-900">默认启用概念链接</Label>
                      <p className="text-sm text-gray-600 leading-relaxed">自动识别和链接相关概念</p>
                    </div>
                  </div>
                </div>
                
                <div className="pt-6 border-t border-gray-100">
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <Label className="text-base font-semibold text-gray-800">最大分块大小</Label>
                      <span className="text-lg font-mono text-blue-600 bg-blue-50 px-4 py-2 rounded-lg font-semibold">
                        {maxChunkSize[0]}
                      </span>
                    </div>
                    <div className="px-3">
                      <Slider
                        value={maxChunkSize}
                        onValueChange={setMaxChunkSize}
                        max={5000}
                        min={1000}
                        step={500}
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-3 px-1">
                        <span>1000</span>
                        <span>3000</span>
                        <span>5000</span>
                      </div>
                    </div>
                    <div className="text-sm text-gray-600 bg-blue-50 p-4 rounded-lg border border-blue-100">
                      <p className="leading-relaxed">较大的分块可以保持更多上下文信息，但会增加处理时间</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="storage" className="space-y-6">
            <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader className="pb-6">
                <CardTitle className="text-xl">存储设置</CardTitle>
                <CardDescription className="text-base">
                  配置本地知识库存储选项
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-3">
                  <Label htmlFor="kb-path" className="text-sm font-semibold text-gray-800">知识库路径</Label>
                  <Input
                    id="kb-path"
                    value={knowledgeBasePath}
                    onChange={(e) => setKnowledgeBasePath(e.target.value)}
                    placeholder="./knowledge"
                  />
                </div>
                
                <div className="space-y-3 pt-3 border-t border-gray-100">
                  <div className="flex items-center space-x-4 p-4 rounded-lg hover:bg-gray-50 transition-colors border border-transparent hover:border-gray-200">
                    <Checkbox 
                      id="auto-save" 
                      checked={autoSave}
                      onCheckedChange={(checked) => setAutoSave(checked === true)}
                    />
                    <div className="space-y-1 flex-1">
                      <Label htmlFor="auto-save" className="text-sm font-medium cursor-pointer text-gray-900">自动保存</Label>
                      <p className="text-sm text-gray-600 leading-relaxed">自动保存修改的内容到本地存储</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-4 p-4 rounded-lg hover:bg-gray-50 transition-colors border border-transparent hover:border-gray-200">
                    <Checkbox 
                      id="backup-enabled" 
                      checked={backupEnabled}
                      onCheckedChange={(checked) => setBackupEnabled(checked === true)}
                    />
                    <div className="space-y-1 flex-1">
                      <Label htmlFor="backup-enabled" className="text-sm font-medium cursor-pointer text-gray-900">启用备份</Label>
                      <p className="text-sm text-gray-600 leading-relaxed">定期创建知识库备份文件</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <div className="flex justify-end gap-3 pt-6">
            <Button variant="outline" onClick={resetSettings} className="min-w-[100px]">
              重置
            </Button>
            <Button onClick={saveSettings} disabled={saving} className="min-w-[120px]">
              {saving ? <Spinner className="mr-2 h-4 w-4" /> : <Save className="mr-2 h-4 w-4" />}
              保存设置
            </Button>
          </div>
        </Tabs>
      </div>
    </div>
  );
}
