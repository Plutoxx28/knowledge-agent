"""
AI工具编排器 - 智能动态工具组合系统
"""
import asyncio
import json
import logging
import time
import uuid
import re
import hashlib
from typing import Dict, Any, List, Optional, Callable, Union
from openai import OpenAI
from config import settings
from .progress_tracker import SimpleProgressTracker
from .strategy_history import StrategyHistoryDB, ExecutionRecord
from .strategy_optimizer import StrategyOptimizer, OptimizationContext
from .strategy_definitions import strategy_registry

logger = logging.getLogger(__name__)

def check_should_stop(task_id: str):
    """检查任务是否应该停止"""
    if not task_id:
        return
    
    try:
        # 导入api_server的active_tasks
        from api_server import active_tasks
        if task_id in active_tasks and active_tasks[task_id] == "stopped":
            from api_server import ProcessingStoppedException
            raise ProcessingStoppedException(f"任务 {task_id} 被用户停止")
    except ImportError:
        # 如果无法导入，跳过检查
        pass


class AIToolOrchestrator:
    """AI工具编排器 - 智能动态工具组合系统"""
    
    def __init__(self, websocket_broadcast_func: Optional[Callable] = None):
        self.websocket_broadcast = websocket_broadcast_func
        
        # OpenRouter客户端配置
        try:
            self.ai_client = OpenAI(
                base_url=settings.openrouter_base_url,
                api_key=settings.openrouter_api_key,
                default_headers={
                    "HTTP-Referer": "https://knowledge-agent.local",
                    "X-Title": "Knowledge Agent - AI Tool Orchestrator",
                }
            )
            logger.info("AI工具编排器初始化成功")
        except Exception as e:
            logger.error(f"AI客户端初始化失败: {e}")
            self.ai_client = None
        
        # 模型配置
        self.flash_model = "google/gemini-2.5-flash-preview-05-20"
        self.pro_model = "google/gemini-2.5-pro"
        
        # 初始化工具池
        self._initialize_tool_registry()
        
        # 复合工具存储
        self.composite_tools = {}
        
        # 执行历史 (传统内存存储，保持兼容性)
        self.execution_history = []
        
        # 策略历史数据库 (新增)
        try:
            self.strategy_history = StrategyHistoryDB()
            logger.info("策略历史数据库初始化成功")
        except Exception as e:
            logger.error(f"策略历史数据库初始化失败: {e}")
            self.strategy_history = None
        
        # 策略优化器 (新增)
        try:
            self.strategy_optimizer = StrategyOptimizer(
                history_db=self.strategy_history,
                strategy_registry=strategy_registry
            )
            logger.info("策略优化器初始化成功")
        except Exception as e:
            logger.error(f"策略优化器初始化失败: {e}")
            self.strategy_optimizer = None
        
        # 策略优化开关
        self.enable_strategy_optimization = True
    
    def _initialize_tool_registry(self):
        """初始化完整的工具注册表"""
        
        self.tool_registry = {
            # === Flash模型工具 - 分析阶段 ===
            "basic_content_analyzer": {
                "function": self._analyze_content_basic,
                "description": "基础内容分析：主题、类型、复杂度",
                "model": self.flash_model,
                "phase": "analysis",
                "cost": "low"
            },
            "content_type_detector": {
                "function": self._detect_content_type_advanced,
                "description": "精确识别内容类型：代码、学术、新闻、对话等",
                "model": self.flash_model,
                "phase": "analysis", 
                "cost": "low"
            },
            "complexity_assessor": {
                "function": self._assess_processing_complexity,
                "description": "评估内容处理复杂度和所需资源",
                "model": self.flash_model,
                "phase": "analysis",
                "cost": "low"
            },
            
            # === Flash模型工具 - 提取阶段 ===
            "general_concept_extractor": {
                "function": self._extract_concepts_general,
                "description": "提取通用概念和术语",
                "model": self.flash_model,
                "phase": "extraction",
                "cost": "medium"
            },
            "technical_concept_extractor": {
                "function": self._extract_technical_concepts,
                "description": "提取技术概念：API、框架、算法等",
                "model": self.flash_model,
                "phase": "extraction",
                "cost": "medium"
            },
            "academic_concept_extractor": {
                "function": self._extract_academic_concepts,
                "description": "提取学术概念：理论、方法、发现等", 
                "model": self.flash_model,
                "phase": "extraction",
                "cost": "medium"
            },
            "code_block_extractor": {
                "function": self._extract_code_blocks,
                "description": "提取和分析代码片段",
                "model": self.flash_model,
                "phase": "extraction",
                "cost": "low"
            },
            "api_endpoint_identifier": {
                "function": self._identify_api_endpoints,
                "description": "识别API接口和调用方式",
                "model": self.flash_model,
                "phase": "extraction",
                "cost": "medium"
            },
            "citation_extractor": {
                "function": self._extract_citations,
                "description": "提取引用文献和来源链接",
                "model": self.flash_model,
                "phase": "extraction",
                "cost": "low"
            },
            "dialogue_parser": {
                "function": self._parse_dialogue_structure,
                "description": "解析对话结构和角色",
                "model": self.flash_model,
                "phase": "extraction",
                "cost": "low"
            },
            
            # === Flash模型工具 - 增强阶段 ===
            "summary_generator": {
                "function": self._generate_content_summary,
                "description": "生成内容摘要和要点",
                "model": self.flash_model,
                "phase": "enhancement",
                "cost": "medium"
            },
            "outline_creator": {
                "function": self._create_content_outline,
                "description": "创建内容大纲和层级结构",
                "model": self.flash_model,
                "phase": "enhancement",
                "cost": "medium"
            },
            "tag_generator": {
                "function": self._generate_content_tags,
                "description": "生成分类标签和关键词",
                "model": self.flash_model,
                "phase": "enhancement", 
                "cost": "low"
            },
            "relationship_analyzer": {
                "function": self._analyze_concept_relationships,
                "description": "分析概念间的关系：因果、层级、并列",
                "model": self.flash_model,
                "phase": "enhancement",
                "cost": "high"
            },
            
            # === Pro模型工具 - 质量控制阶段 ===
            "content_quality_validator": {
                "function": self._validate_content_quality,
                "description": "深度验证内容完整性和准确性",
                "model": self.pro_model,
                "phase": "quality_control",
                "cost": "high"
            },
            "concept_completeness_checker": {
                "function": self._check_concept_completeness,
                "description": "检查概念提取的完整性和深度",
                "model": self.pro_model,
                "phase": "quality_control",
                "cost": "medium"
            },
            "logical_consistency_checker": {
                "function": self._check_logical_consistency,
                "description": "验证内容逻辑一致性",
                "model": self.pro_model,
                "phase": "quality_control",
                "cost": "high"
            },
            "result_optimizer": {
                "function": self._optimize_processing_results,
                "description": "优化和改进处理结果",
                "model": self.pro_model,
                "phase": "quality_control",
                "cost": "high"
            },
            
            # === Pro模型工具 - 合成阶段 ===
            "advanced_markdown_structurer": {
                "function": self._structure_as_advanced_markdown,
                "description": "高级Markdown结构化，整合所有信息",
                "model": self.pro_model,
                "phase": "synthesis",
                "cost": "high"
            },
            "specialized_formatter": {
                "function": self._format_specialized_content,
                "description": "根据内容类型进行专门化格式化",
                "model": self.pro_model,
                "phase": "synthesis",
                "cost": "high"
            },
            
            # === 无需AI的工具 - Fallback ===
            "regex_concept_extractor": {
                "function": self._fallback_concept_extraction,
                "description": "基于正则表达式的概念提取",
                "model": None,
                "phase": "fallback",
                "cost": "zero"
            },
            "template_structurer": {
                "function": self._create_fallback_structure,
                "description": "基于模板的结构化",
                "model": None,
                "phase": "fallback",
                "cost": "zero"
            }
        }
    
    async def process_content_with_orchestration(self, content: str, 
                                               content_type: str = "auto",
                                               metadata: Dict[str, Any] = None,
                                               options: Dict[str, Any] = None) -> Dict[str, Any]:
        """AI编排的智能内容处理 - 集成策略优化"""
        
        tracker = SimpleProgressTracker(self.websocket_broadcast)
        task_id = options.get("task_id") if options else None
        processing_start_time = time.time()
        
        try:
            # 检查是否应该停止
            check_should_stop(task_id)
            
            # 第一步：分析内容特征
            await tracker.update_progress("analysis", "分析内容特征...", stage_progress=0.2)
            content_features = self._analyze_content_features(content)
            
            # 第二步：策略优化选择 (新增)
            selected_strategy = None
            strategy_recommendation = None
            if self.enable_strategy_optimization and self.strategy_optimizer:
                await tracker.update_progress("strategy_selection", "AI智能策略选择...", stage_progress=0.5)
                
                try:
                    # 创建优化上下文
                    optimization_context = OptimizationContext(
                        content_features=content_features,
                        user_preferences=options.get("user_preferences", {}) if options else {},
                        system_constraints=options.get("system_constraints", {}) if options else {},
                        historical_context={},
                        time_constraints=options.get("time_limit") if options else None,
                        quality_requirements=options.get("min_quality") if options else None
                    )
                    
                    # 智能策略选择
                    strategy_recommendation = self.strategy_optimizer.select_optimal_strategy(
                        content_features, optimization_context
                    )
                    
                    selected_strategy = strategy_recommendation.strategy_name
                    logger.info(f"策略优化器推荐策略: {selected_strategy} (置信度: {strategy_recommendation.confidence_score:.3f})")
                    
                    await tracker.update_progress("strategy_selection", 
                        f"选择策略: {selected_strategy} (置信度: {strategy_recommendation.confidence_score:.2f})", 
                        stage_progress=1.0)
                    
                except Exception as e:
                    logger.error(f"策略优化选择失败，使用传统方法: {e}")
                    selected_strategy = None
            
            tracker._complete_stage("strategy_selection")
            
            # 第三步：制定处理计划
            await tracker.update_progress("planning", "制定处理计划...", stage_progress=0.3)
            
            if selected_strategy:
                # 使用策略优化器推荐的策略
                processing_plan = await self._create_strategy_based_plan(content, content_type, selected_strategy, content_features)
            else:
                # 传统AI规划方法
                processing_plan = await self._ai_create_processing_plan(content, content_type)
            
            await tracker.update_progress("planning", f"处理方案制定完成 - {processing_plan.get('strategy_name', '智能处理')}", stage_progress=1.0)
            tracker._complete_stage("planning")
            
            # 检查是否应该停止
            check_should_stop(task_id)
            
            # 第二步：创建AI设计的复合工具（如果需要）
            if processing_plan.get("create_composite_tools"):
                await tracker.update_progress("tool_creation", "AI创建专用处理工具...", stage_progress=0.5)
                self._create_composite_tools(processing_plan["create_composite_tools"])
                await tracker.update_progress("tool_creation", f"创建了 {len(processing_plan['create_composite_tools'])} 个专用工具", stage_progress=1.0)
                tracker._complete_stage("tool_creation")
            
            # 检查是否应该停止
            check_should_stop(task_id)
            
            # 第三步：执行处理计划
            results = await self._execute_processing_plan(processing_plan, content, tracker, task_id)
            
            # 检查是否应该停止
            check_should_stop(task_id)
            
            # 第四步：Pro模型质量控制
            await tracker.update_progress("quality_control", "Pro模型质量检查中...", stage_progress=0.5)
            
            quality_results = await self._run_quality_control(results, processing_plan)
            
            await tracker.update_progress("quality_control", "质量检查完成", stage_progress=1.0)
            tracker._complete_stage("quality_control")
            
            # 检查是否应该停止
            check_should_stop(task_id)
            
            # 第五步：Pro模型最终合成
            await tracker.update_progress("synthesis", "Pro模型最终合成中...", stage_progress=0.5)
            
            final_content = await self._final_synthesis(results, quality_results, processing_plan)
            
            await tracker.update_progress("synthesis", "最终合成完成", stage_progress=1.0)
            tracker._complete_stage("synthesis")
            
            await tracker.update_progress("completed", "AI智能处理完成！", progress_percent=100)
            
            # 记录执行历史 (包含策略优化信息)
            processing_time = time.time() - processing_start_time
            enhanced_plan = processing_plan.copy()
            enhanced_plan.update({
                "processing_time": processing_time,
                "strategy_optimization_used": selected_strategy is not None,
                "selected_strategy": selected_strategy,
                "strategy_confidence": strategy_recommendation.confidence_score if strategy_recommendation else None,
                "content_features": content_features
            })
            
            self._record_execution_history(content, enhanced_plan, results, final_content)
            
            return {
                "success": True,
                "result": {
                    "content": final_content,
                    "concepts": results.get("concepts", []),
                    "analysis": results.get("analysis", {}),
                    "processing_plan": processing_plan,
                    "quality_score": quality_results.get("overall_score", 0.8),
                    "tools_used": self._get_tools_used(processing_plan),
                    "ai_calls": {
                        "flash_calls": self._count_flash_calls(processing_plan),
                        "pro_calls": self._count_pro_calls(processing_plan)
                    },
                    "metadata": metadata or {},
                    "statistics": self._generate_enhanced_statistics(results)
                },
                "doc_id": tracker.task_id,
                "task_id": tracker.task_id
            }
            
        except Exception as e:
            logger.error(f"AI智能处理失败: {e}")
            # 降级到简化处理
            return await self._fallback_to_simple_processing(content, content_type, metadata, options)
    
    # === 核心处理方法 ===
    
    async def _ai_create_processing_plan(self, content: str, content_type: str) -> Dict[str, Any]:
        """AI创建智能处理计划"""
        
        if not self.ai_client:
            return self._get_default_processing_plan()
        
        available_tools = self._format_available_tools()
        content_preview = content[:2000]
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ai_client.chat.completions.create(
                    model=self.flash_model,  # 用Flash做计划制定
                    messages=[
                        {
                            "role": "system", 
                            "content": """你是一个专业的内容处理策略专家。分析内容并设计最优的处理方案。

可用工具分为几个阶段：
- analysis阶段：内容分析工具（Flash模型）
- extraction阶段：信息提取工具（Flash模型）
- enhancement阶段：内容增强工具（Flash模型）
- quality_control阶段：质量控制工具（Pro模型）
- synthesis阶段：最终合成工具（Pro模型）

你可以：
1. 选择最适合的工具组合
2. 创建专门的复合工具
3. 设计并行和串行执行策略
4. 优化Flash和Pro模型的使用

请返回有效的JSON格式处理方案。"""
                        },
                        {
                            "role": "user",
                            "content": f"""内容类型：{content_type}
内容预览：
{content_preview}

可用工具：
{available_tools}

请设计处理方案：
{{
    "strategy_name": "处理策略名称",
    "content_analysis": "内容分析结果",
    "recommended_approach": "推荐的处理方法",
    
    "create_composite_tools": [
        {{
            "name": "复合工具名",
            "description": "工具描述",
            "atomic_tools": ["tool1", "tool2"],
            "combination_logic": "sequential|parallel|conditional"
        }}
    ],
    
    "execution_phases": [
        {{
            "phase": "analysis",
            "tools": ["tool1", "tool2"],
            "execution_type": "parallel",
            "progress_weight": 15
        }},
        {{
            "phase": "extraction", 
            "tools": ["tool3", "tool4"],
            "execution_type": "sequential",
            "progress_weight": 25
        }},
        {{
            "phase": "enhancement",
            "tools": ["tool5"],
            "execution_type": "sequential", 
            "progress_weight": 20
        }},
        {{
            "phase": "quality_control",
            "tools": ["tool6", "tool7"],
            "execution_type": "parallel",
            "progress_weight": 15
        }},
        {{
            "phase": "synthesis",
            "tools": ["tool8"],
            "execution_type": "sequential",
            "progress_weight": 25
        }}
    ],
    
    "quality_requirements": ["requirement1", "requirement2"],
    "expected_output_format": "advanced_markdown|specialized_format"
}}"""
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.1
                )
            )
            
            plan_text = response.choices[0].message.content
            logger.info(f"AI生成处理方案: {plan_text[:200]}...")
            
            # 解析AI返回的方案
            try:
                # 清理可能的markdown标记
                clean_plan = plan_text.strip()
                if clean_plan.startswith('```json'):
                    clean_plan = clean_plan[7:]
                if clean_plan.endswith('```'):
                    clean_plan = clean_plan[:-3]
                clean_plan = clean_plan.strip()
                
                plan = json.loads(clean_plan)
                return plan
                
            except json.JSONDecodeError:
                logger.warning("AI方案JSON解析失败，使用默认方案")
                return self._get_default_processing_plan()
                
        except Exception as e:
            logger.error(f"AI方案生成失败: {e}")
            return self._get_default_processing_plan()
    
    def _get_default_processing_plan(self) -> Dict[str, Any]:
        """默认处理方案（原有逻辑）"""
        return {
            "strategy_name": "标准处理流程",
            "content_analysis": "使用默认处理流程",
            "recommended_approach": "standard_pipeline", 
            "create_composite_tools": [],
            "execution_phases": [
                {
                    "phase": "analysis",
                    "tools": ["basic_content_analyzer"],
                    "execution_type": "sequential",
                    "progress_weight": 20
                },
                {
                    "phase": "extraction", 
                    "tools": ["general_concept_extractor"],
                    "execution_type": "sequential",
                    "progress_weight": 30
                },
                {
                    "phase": "quality_control",
                    "tools": ["content_quality_validator"],
                    "execution_type": "sequential",
                    "progress_weight": 20
                },
                {
                    "phase": "synthesis",
                    "tools": ["advanced_markdown_structurer"], 
                    "execution_type": "sequential",
                    "progress_weight": 30
                }
            ],
            "quality_requirements": ["basic_validation"],
            "expected_output_format": "advanced_markdown"
        }
    
    # === 核心工具方法实现 ===
    
    async def _call_ai_with_model(self, model: str, messages: List[Dict], max_tokens: int = 1000) -> str:
        """统一的AI调用方法"""
        if not self.ai_client:
            raise Exception("AI客户端未初始化")
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.1
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI调用失败 ({model}): {e}")
            raise e
    
    # === Flash模型工具 - 分析阶段 ===
    
    async def _analyze_content_basic(self, context: Dict) -> Dict[str, Any]:
        """基础内容分析"""
        content = context.get("content", "")
        
        try:
            response = await self._call_ai_with_model(
                model=self.flash_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的内容分析专家。请分析给定内容的主题、复杂度和结构。"
                    },
                    {
                        "role": "user",
                        "content": f"""请分析以下内容：

{content[:1500]}

请以JSON格式返回分析结果：
{{
  "main_topic": "主要话题",
  "complexity": "simple|medium|complex",
  "content_type": "technical|educational|general",
  "key_themes": ["主题1", "主题2"]
}}"""
                    }
                ],
                max_tokens=500
            )
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "main_topic": "未知主题",
                    "complexity": "medium",
                    "content_type": "general",
                    "key_themes": [],
                    "ai_raw_response": response
                }
                
        except Exception as e:
            logger.error(f"基础内容分析失败: {e}")
            return {
                "main_topic": "分析失败",
                "complexity": "medium",
                "content_type": "general",
                "key_themes": [],
                "error": str(e)
            }
    
    # === 其他工具方法实现 ===
    
    async def _detect_content_type_advanced(self, context: Dict) -> Dict[str, Any]:
        """高级内容类型检测"""
        content = context.get("content", "")
        
        try:
            response = await self._call_ai_with_model(
                model=self.flash_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是内容类型识别专家。请精确识别内容的具体类型和特征。"
                    },
                    {
                        "role": "user",
                        "content": f"""请识别以下内容的类型：

{content[:1000]}

可能的类型包括：
- code_tutorial: 代码教程
- academic_paper: 学术论文
- api_documentation: API文档
- conversation: 对话记录
- news_article: 新闻文章
- technical_blog: 技术博客
- meeting_notes: 会议记录
- general_text: 普通文本

请返回JSON：
{{
  "primary_type": "主要类型",
  "confidence": 0.95,
  "secondary_types": ["次要类型"],
  "detected_features": ["特征1", "特征2"],
  "language": "内容语言",
  "programming_languages": ["如果包含代码"]
}}"""
                    }
                ],
                max_tokens=300
            )
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "primary_type": "general_text",
                    "confidence": 0.5,
                    "secondary_types": [],
                    "detected_features": [],
                    "language": "unknown"
                }
                
        except Exception as e:
            logger.error(f"内容类型检测失败: {e}")
            return {
                "primary_type": "general_text",
                "confidence": 0.3,
                "error": str(e)
            }
    
    async def _assess_processing_complexity(self, context: Dict) -> Dict[str, Any]:
        """评估处理复杂度"""
        content = context.get("content", "")
        
        # 基于规则的初步评估
        content_length = len(content)
        line_count = content.count('\n')
        
        if content_length < 500:
            base_complexity = "simple"
        elif content_length < 2000:
            base_complexity = "medium"
        else:
            base_complexity = "complex"
        
        try:
            response = await self._call_ai_with_model(
                model=self.flash_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是处理复杂度评估专家。评估内容的处理难度和所需资源。"
                    },
                    {
                        "role": "user",
                        "content": f"""评估以下内容的处理复杂度：

内容长度：{content_length} 字符
行数：{line_count}
内容预览：
{content[:800]}

请返回JSON：
{{
  "complexity_level": "simple|medium|complex|very_complex",
  "estimated_processing_time": "预估时间（秒）",
  "required_tools": ["建议工具1", "建议工具2"],
  "complexity_factors": ["复杂因子1", "复杂因子2"],
  "recommended_approach": "建议的处理方法"
}}"""
                    }
                ],
                max_tokens=400
            )
            
            try:
                result = json.loads(response)
                result["base_complexity"] = base_complexity
                return result
            except json.JSONDecodeError:
                return {
                    "complexity_level": base_complexity,
                    "estimated_processing_time": "30-60",
                    "required_tools": ["general_concept_extractor"],
                    "base_complexity": base_complexity
                }
                
        except Exception as e:
            logger.error(f"复杂度评估失败: {e}")
            return {
                "complexity_level": base_complexity,
                "estimated_processing_time": "unknown",
                "error": str(e)
            }
    
    # === Flash模型工具 - 提取阶段 ===
    
    async def _extract_concepts_general(self, context: Dict) -> List[Dict[str, Any]]:
        """提取通用概念"""
        content = context.get("content", "")
        
        try:
            response = await self._call_ai_with_model(
                model=self.flash_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的概念提取专家。请从内容中提取核心概念和重要术语。"
                    },
                    {
                        "role": "user",
                        "content": f"""请从以下内容中提取核心概念：

{content[:1500]}

请以JSON数组格式返回概念列表：
[
  {{"term": "概念名称", "definition": "概念定义", "type": "概念类型", "confidence": 0.9}},
  {{"term": "术语名称", "definition": "术语解释", "type": "technical_term", "confidence": 0.8}}
]"""
                    }
                ],
                max_tokens=800
            )
            
            try:
                # 清理markdown标记
                clean_response = response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                concepts = json.loads(clean_response)
                # 添加工具标记
                for concept in concepts:
                    concept['source'] = 'ai_general_extractor'
                    concept['final_score'] = concept.get('confidence', 0.7)
                
                return concepts
                
            except json.JSONDecodeError:
                return self._fallback_concept_extraction(content)
                
        except Exception as e:
            logger.error(f"通用概念提取失败: {e}")
            return self._fallback_concept_extraction(content)
    
    # Fallback方法（复用现有代码）
    def _fallback_concept_extraction(self, content: str) -> List[Dict[str, Any]]:
        """备用概念提取方法"""
        concepts = []
        
        # 简单的关键词提取
        patterns = [
            (r'\b([A-Z]{2,})\b', 'acronym'),
            (r'([一-龟]{2,8})', 'chinese_term'),
            (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', 'proper_noun'),
        ]
        
        for pattern, term_type in patterns:
            matches = re.findall(pattern, content)
            for match in matches[:5]:  # 限制数量
                concepts.append({
                    'term': match,
                    'definition': '',
                    'type': term_type,
                    'confidence': 0.6,
                    'source': 'fallback_extraction',
                    'final_score': 0.6
                })
        
        return concepts[:10]  # 最多10个概念
    
    # === 其他工具方法（简化版） ===
    
    async def _extract_technical_concepts(self, context: Dict) -> List[Dict[str, Any]]:
        """提取技术概念"""
        # 简化实现
        return await self._extract_concepts_general(context)
    
    async def _extract_academic_concepts(self, context: Dict) -> List[Dict[str, Any]]:
        """提取学术概念"""
        # 简化实现
        return await self._extract_concepts_general(context)
    
    async def _extract_code_blocks(self, context: Dict) -> Dict[str, Any]:
        """提取代码块"""
        content = context.get("content", "")
        
        # 简单的代码块提取
        code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', content, re.DOTALL)
        inline_code = re.findall(r'`([^`]+)`', content)
        
        return {
            "code_blocks": code_blocks,
            "inline_code": inline_code,
            "has_code": len(code_blocks) > 0 or len(inline_code) > 0
        }
    
    async def _identify_api_endpoints(self, context: Dict) -> Dict[str, Any]:
        """识别API端点"""
        content = context.get("content", "")
        
        # 简单的API端点识别
        api_patterns = [
            r'(GET|POST|PUT|DELETE)\s+(/[^\s]*)',
            r'(https?://[^\s/]+/[^\s]*)',
            r'(\w+\.\w+\([^)]*\))'  # 函数调用
        ]
        
        apis = []
        for pattern in api_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            apis.extend(matches)
        
        return {
            "api_endpoints": apis[:10],
            "endpoint_count": len(apis)
        }
    
    async def _extract_citations(self, context: Dict) -> List[Dict[str, Any]]:
        """提取引用"""
        content = context.get("content", "")
        
        # 简单的引用提取
        citation_patterns = [
            r'\[(\d+)\]',  # [1]
            r'\(([^)]+\d{4}[^)]*)\)',  # (Author 2023)
            r'https?://[^\s]+'  # URLs
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, content)
            citations.extend([{"citation": match, "type": "reference"} for match in matches])
        
        return citations[:10]
    
    async def _parse_dialogue_structure(self, context: Dict) -> Dict[str, Any]:
        """解析对话结构"""
        content = context.get("content", "")
        
        # 简单的对话解析
        dialogue_patterns = [
            r'(用户|User)[:：]\s*(.+)',
            r'(助手|Assistant|AI)[:：]\s*(.+)',
        ]
        
        dialogues = []
        for pattern in dialogue_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            dialogues.extend([{"role": match[0], "content": match[1]} for match in matches])
        
        return {
            "dialogues": dialogues,
            "is_conversation": len(dialogues) > 0,
            "turn_count": len(dialogues)
        }
    
    # === 增强阶段工具 ===
    
    async def _generate_content_summary(self, context: Dict) -> Dict[str, Any]:
        """生成内容摘要"""
        content = context.get("content", "")
        
        # 简单摘要生成
        lines = content.split('\n')
        important_lines = [line.strip() for line in lines if len(line.strip()) > 20][:5]
        
        return {
            "summary": '\n'.join(important_lines),
            "key_points": important_lines[:3]
        }
    
    async def _create_content_outline(self, context: Dict) -> Dict[str, Any]:
        """创建内容大纲"""
        content = context.get("content", "")
        
        # 简单大纲生成
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        
        return {
            "outline": headers,
            "structure_depth": len(set(len(h.split()) for h in headers)) if headers else 1
        }
    
    async def _generate_content_tags(self, context: Dict) -> List[str]:
        """生成内容标签"""
        content = context.get("content", "")
        analysis = context.get("analysis", {})
        
        # 基于分析结果生成标签
        tags = []
        
        if analysis.get("content_type"):
            tags.append(analysis["content_type"])
        
        if analysis.get("complexity"):
            tags.append(f"complexity_{analysis['complexity']}")
        
        # 添加一些通用标签
        if "代码" in content or "code" in content.lower():
            tags.append("编程")
        if "学习" in content or "研究" in content:
            tags.append("学术")
        
        return tags[:10]
    
    async def _analyze_concept_relationships(self, context: Dict) -> Dict[str, Any]:
        """分析概念关系"""
        concepts = context.get("concepts", [])
        
        # 简单的关系分析
        relationships = []
        
        if len(concepts) >= 2:
            for i, concept1 in enumerate(concepts[:5]):
                for concept2 in concepts[i+1:6]:
                    # 简单的相似性检查
                    term1 = concept1.get('term', '').lower()
                    term2 = concept2.get('term', '').lower()
                    
                    if any(word in term2 for word in term1.split()) or any(word in term1 for word in term2.split()):
                        relationships.append({
                            "from": concept1.get('term'),
                            "to": concept2.get('term'),
                            "relationship": "related",
                            "confidence": 0.7
                        })
        
        return {
            "relationships": relationships[:10],
            "relationship_count": len(relationships)
        }
    
    # === Pro模型工具（简化版） ===
    
    async def _validate_content_quality(self, context: Dict) -> Dict[str, Any]:
        """验证内容质量"""
        return {"validation_score": 0.8, "issues": []}
    
    async def _check_concept_completeness(self, context: Dict) -> Dict[str, Any]:
        """检查概念完整性"""
        return {"completeness_score": 0.85, "missing_concepts": []}
    
    async def _check_logical_consistency(self, context: Dict) -> Dict[str, Any]:
        """检查逻辑一致性"""
        return {"consistency_score": 0.9, "inconsistencies": []}
    
    async def _optimize_processing_results(self, context: Dict) -> Dict[str, Any]:
        """优化处理结果"""
        return {"optimization_suggestions": [], "improved_quality": True}
    
    async def _structure_as_advanced_markdown(self, context: Dict) -> str:
        """高级Markdown结构化"""
        # 使用Pro模型最终合成方法
        content = context.get("content", "")
        concepts = context.get("concepts", [])
        analysis = context.get("analysis", {})
        quality_results = {"overall_score": 0.8, "suggestions": []}
        plan = {"strategy_name": "高级结构化"}
        
        return await self._final_synthesis(
            {"content": content, "concepts": concepts, "analysis": analysis},
            quality_results,
            plan
        )
    
    async def _format_specialized_content(self, context: Dict) -> str:
        """专门化格式化"""
        # 根据内容类型使用不同格式
        content_type = context.get("analysis", {}).get("content_type", "general")
        
        if content_type == "technical":
            return await self._structure_as_advanced_markdown(context)
        else:
            return self._create_fallback_structure(
                context.get("content", ""),
                context.get("concepts", []),
                context.get("analysis", {})
            )
    
    def _create_fallback_structure(self, content: str, concepts: List[Dict] = None, analysis: Dict[str, Any] = None) -> str:
        """创建fallback结构化内容"""
        if concepts is None:
            concepts = []
        if analysis is None:
            analysis = {}
            
        main_topic = analysis.get('main_topic', '知识整理')
        if main_topic in ["未知主题", "分析失败"]:
            # 尝试从内容中提取标题
            lines = content.strip().split('\n')
            for line in lines[:5]:
                line = line.strip()
                if line.startswith('#'):
                    main_topic = line.lstrip('#').strip()
                    break
                elif len(line) > 5 and len(line) < 100:
                    main_topic = line
                    break
            else:
                main_topic = "知识整理"
        
        # 构建四部分结构
        structured_parts = [f"# {main_topic}"]
        
        # 1. 相关反向链接
        structured_parts.append("\n## 相关反向链接\n")
        if concepts and len(concepts) > 0:
            # 基于概念生成简单的反向链接
            valid_concepts = [c for c in concepts[:5] if c.get('term') and len(c['term'].strip()) > 1]
            if valid_concepts:
                for concept in valid_concepts:
                    term = concept['term'].strip()
                    concept_type = concept.get('type', 'general')
                    structured_parts.append(f"- [[{term}]] - {concept_type}相关主题")
            else:
                structured_parts.append("- 暂无相关链接")
        else:
            structured_parts.append("- 暂无相关链接")
        
        # 2. 相关概念
        structured_parts.append("\n## 相关概念\n")
        if concepts and len(concepts) > 0:
            valid_concepts = [c for c in concepts[:8] if c.get('term') and len(c['term'].strip()) > 1]
            if valid_concepts:
                for concept in valid_concepts:
                    term = concept['term'].strip()
                    definition = concept.get('definition', '').strip()
                    if definition:
                        structured_parts.append(f"- [[{term}]]：{definition}")
                    else:
                        structured_parts.append(f"- [[{term}]]")
            else:
                structured_parts.append("- 暂无提取到有效概念")
        else:
            structured_parts.append("- 暂无提取到有效概念")
        
        # 3. 详细内容（完全保留原始输入）
        structured_parts.append(f"\n## 详细内容\n\n{content}")
        
        # 4. 扩展知识
        structured_parts.append("\n## 扩展知识\n")
        # 基于内容和概念生成简单的扩展知识
        if concepts and len(concepts) > 0:
            concept_types = list(set([c.get('type', 'general') for c in concepts if c.get('type')]))
            if concept_types:
                for concept_type in concept_types[:3]:
                    structured_parts.append(f"- {concept_type}相关的深入学习")
            else:
                structured_parts.append("- 相关领域的深入学习")
        else:
            structured_parts.append("- 相关领域的深入学习")
        
        return '\n'.join(structured_parts)
    
    # === 执行引擎 ===
    
    async def _execute_processing_plan(self, plan: Dict[str, Any], content: str, tracker, task_id: str = None) -> Dict[str, Any]:
        """执行AI制定的处理计划"""
        
        results = {"content": content, "original_content": content}
        
        try:
            for phase_config in plan.get("execution_phases", []):
                # 检查是否应该停止
                check_should_stop(task_id)
                
                phase_name = phase_config["phase"]
                tools = phase_config["tools"]
                execution_type = phase_config.get("execution_type", "sequential")
                
                # 开始处理阶段
                await tracker.update_progress(
                    phase_name, 
                    f"执行{phase_name}阶段处理...", 
                    stage_progress=0.3,
                    workers=tools
                )
                
                # 根据执行类型处理
                if execution_type == "parallel":
                    phase_results = await self._execute_tools_parallel(tools, results)
                else:
                    phase_results = await self._execute_tools_sequential(tools, results)
                
                # 合并结果
                results.update(phase_results)
                
                # 完成阶段
                await tracker.update_progress(
                    phase_name,
                    f"{phase_name}阶段完成",
                    stage_progress=1.0
                )
                
                # 标记阶段完成，更新累积进度
                tracker._complete_stage(phase_name)
            
            return results
            
        except Exception as e:
            logger.error(f"执行处理计划失败: {e}")
            # 返回基础结果
            return {
                "content": content,
                "error": str(e),
                "analysis": {"main_topic": "处理失败", "complexity": "unknown"},
                "concepts": []
            }
    
    # === 工具执行引擎 ===
    
    async def _execute_tools_parallel(self, tools: List[str], context: Dict) -> Dict[str, Any]:
        """并行执行工具"""
        
        tasks = []
        for tool_name in tools:
            if tool_name in self.tool_registry:
                tool_config = self.tool_registry[tool_name]
                task = self._execute_single_tool(tool_name, tool_config, context)
                tasks.append((tool_name, task))
            elif tool_name in self.composite_tools:
                # 执行复合工具
                task = self._execute_composite_tool(tool_name, context)
                tasks.append((tool_name, task))
        
        results = {}
        try:
            # 执行所有任务
            task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # 整合并行结果
            for i, (tool_name, _) in enumerate(tasks):
                result = task_results[i]
                if not isinstance(result, Exception):
                    results[f"{tool_name}_result"] = result
                    # 特殊处理某些工具结果
                    if tool_name.endswith("_concept_extractor") and isinstance(result, list):
                        if "concepts" not in results:
                            results["concepts"] = []
                        results["concepts"].extend(result)
                    elif tool_name.endswith("_analyzer") and isinstance(result, dict):
                        if "analysis" not in results:
                            results["analysis"] = {}
                        results["analysis"].update(result)
                else:
                    logger.error(f"工具{tool_name}执行失败: {result}")
        
        except Exception as e:
            logger.error(f"并行执行失败: {e}")
        
        return results
    
    async def _execute_tools_sequential(self, tools: List[str], context: Dict) -> Dict[str, Any]:
        """串行执行工具"""
        
        results = {}
        current_context = context.copy()
        
        for tool_name in tools:
            try:
                if tool_name in self.tool_registry:
                    tool_config = self.tool_registry[tool_name]
                    result = await self._execute_single_tool(tool_name, tool_config, current_context)
                elif tool_name in self.composite_tools:
                    result = await self._execute_composite_tool(tool_name, current_context)
                else:
                    logger.warning(f"工具不存在: {tool_name}")
                    continue
                
                results[f"{tool_name}_result"] = result
                
                # 特殊处理某些工具结果
                if tool_name.endswith("_concept_extractor") and isinstance(result, list):
                    if "concepts" not in results:
                        results["concepts"] = []
                    results["concepts"].extend(result)
                elif tool_name.endswith("_analyzer") and isinstance(result, dict):
                    if "analysis" not in results:
                        results["analysis"] = {}
                    results["analysis"].update(result)
                
                # 更新上下文，让后续工具可以使用前面的结果
                current_context[f"previous_{tool_name}"] = result
                if isinstance(result, dict):
                    current_context.update(result)
                
            except Exception as e:
                logger.error(f"工具{tool_name}执行失败: {e}")
                results[f"{tool_name}_error"] = str(e)
        
        return results
    
    async def _execute_single_tool(self, tool_name: str, tool_config: Dict, context: Dict) -> Any:
        """执行单个工具"""
        
        function = tool_config["function"]
        model = tool_config.get("model")
        
        try:
            # 如果工具不需要AI调用
            if model is None:
                if asyncio.iscoroutinefunction(function):
                    return await function(context)
                else:
                    return function(context)
            else:
                # 需要AI调用的工具
                return await function(context)
                
        except Exception as e:
            logger.error(f"执行工具{tool_name}失败: {e}")
            raise e
    
    # === 复合工具系统 ===
    
    def _create_composite_tools(self, composite_tool_specs: List[Dict]):
        """创建AI设计的复合工具"""
        
        for spec in composite_tool_specs:
            composite_tool = {
                "name": spec["name"],
                "description": spec["description"],
                "atomic_tools": spec["atomic_tools"],
                "combination_logic": spec.get("combination_logic", "sequential")
            }
            
            # 加入复合工具池
            self.composite_tools[spec["name"]] = composite_tool
            
            logger.info(f"创建复合工具: {spec['name']} - {spec['description']}")
    
    async def _execute_composite_tool(self, tool_name: str, context: Dict) -> Dict[str, Any]:
        """执行复合工具"""
        
        if tool_name not in self.composite_tools:
            raise Exception(f"复合工具不存在: {tool_name}")
        
        composite_tool = self.composite_tools[tool_name]
        atomic_tools = composite_tool["atomic_tools"]
        logic = composite_tool["combination_logic"]
        
        if logic == "parallel":
            return await self._execute_tools_parallel(atomic_tools, context)
        else:  # sequential
            return await self._execute_tools_sequential(atomic_tools, context)
    
    # === Pro模型质量控制和最终合成 ===
    
    async def _run_quality_control(self, results: Dict, plan: Dict) -> Dict[str, Any]:
        """运行Pro模型质量控制"""
        
        if not self.ai_client:
            return {
                "overall_score": 0.7,
                "quality_checks": ["fallback_validation"],
                "suggestions": ["AI客户端未初始化，跳过深度质量检查"]
            }
        
        content = results.get("content", "")
        concepts = results.get("concepts", [])
        analysis = results.get("analysis", {})
        
        try:
            response = await self._call_ai_with_model(
                model=self.pro_model,
                messages=[
                    {
                        "role": "system",
                        "content": """你是一个专业的内容质量控制专家。请深度分析处理结果的质量，并提供改进建议。

你需要评估：
1. 概念提取的完整性和准确性
2. 内容分析的深度和准确性  
3. 整体处理结果的逻辑一致性
4. 结果是否达到预期质量标准

请提供详细的质量评估和改进建议。"""
                    },
                    {
                        "role": "user",
                        "content": f"""请评估以下处理结果的质量：

原始内容长度：{len(content)} 字符

分析结果：
{json.dumps(analysis, ensure_ascii=False, indent=2)}

概念提取结果（前10个）：
{json.dumps(concepts[:10], ensure_ascii=False, indent=2)}

处理策略：{plan.get('strategy_name', '未知')}

请返回质量评估（JSON格式）：
{{
  "overall_score": 0.85,
  "detailed_scores": {{
    "concept_quality": 0.9,
    "analysis_quality": 0.8,
    "completeness": 0.85,
    "accuracy": 0.9
  }},
  "quality_issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"],
  "pass_criteria": true,
  "improvement_needed": false
}}"""
                    }
                ],
                max_tokens=800
            )
            
            try:
                # 清理可能的markdown标记
                clean_response = response.strip()
                if clean_response.startswith('```json'):
                    clean_response = clean_response[7:]
                if clean_response.endswith('```'):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                quality_results = json.loads(clean_response)
                
                # 确保有必要的字段
                if "overall_score" not in quality_results:
                    quality_results["overall_score"] = 0.8
                    
                logger.info(f"Pro模型质量评估完成，总分: {quality_results.get('overall_score', 0.8)}")
                return quality_results
                
            except json.JSONDecodeError:
                logger.warning("Pro模型质量评估JSON解析失败")
                return {
                    "overall_score": 0.75,
                    "quality_checks": ["pro_model_analysis"],
                    "suggestions": ["JSON解析失败，但Pro模型已分析"],
                    "raw_response": response
                }
                
        except Exception as e:
            logger.error(f"Pro模型质量控制失败: {e}")
            return {
                "overall_score": 0.7,
                "quality_checks": ["basic_validation"],
                "suggestions": [f"质量控制失败: {str(e)}"],
                "error": str(e)
            }
    
    async def _final_synthesis(self, results: Dict, quality_results: Dict, plan: Dict) -> str:
        """Pro模型最终合成"""
        
        content = results.get("content", "")
        concepts = results.get("concepts", [])
        analysis = results.get("analysis", {})
        
        # 如果质量分数太低，使用fallback
        quality_score = quality_results.get("overall_score", 0.8)
        if quality_score < 0.6 or not self.ai_client:
            logger.warning(f"质量分数过低({quality_score})或AI客户端不可用，使用fallback结构化")
            return self._create_fallback_structure(content, concepts, analysis)
        
        try:
            # 准备概念信息
            concept_info = []
            for concept in concepts[:15]:  # 最多15个概念用于合成
                term = concept.get('term', '')
                definition = concept.get('definition', '')
                if term:
                    if definition:
                        concept_info.append(f"- {term}: {definition}")
                    else:
                        concept_info.append(f"- {term}")
            
            concept_text = '\n'.join(concept_info) if concept_info else "暂无概念"
            
            # 获取处理建议
            suggestions = quality_results.get("suggestions", [])
            suggestions_text = '\n'.join([f"- {s}" for s in suggestions[:5]]) if suggestions else ""
            
            response = await self._call_ai_with_model(
                model=self.pro_model,
                messages=[
                    {
                        "role": "system",
                        "content": """你是一个专业的知识整理专家。请将所有处理结果整合为高质量的结构化Markdown文档。

要求：
1. 生成清晰的四部分结构
2. 为重要概念添加双链格式：[[概念名]]
3. 完整保留原始内容
4. 生成相关反向链接
5. 提取扩展知识点
6. 保持逻辑清晰和内容完整

输出格式：
# 标题

## 相关反向链接
- [[相关概念1]] - 关联说明
- [[相关概念2]] - 关联说明

## 相关概念
- [[概念A]]：定义
- [[概念B]]：定义

## 详细内容
原始输入内容（完全保留，不做任何修改）

## 扩展知识
- 扩展知识点1
- 扩展知识点2"""
                    },
                    {
                        "role": "user",
                        "content": f"""请整合以下信息，生成高质量的结构化Markdown文档：

原始内容：
{content}

内容分析：
- 主题：{analysis.get('main_topic', '未知')}
- 复杂度：{analysis.get('complexity', 'medium')}
- 内容类型：{analysis.get('content_type', 'general')}
- 关键主题：{', '.join(analysis.get('key_themes', []))}

提取的概念：
{concept_text}

质量评估分数：{quality_score}

质量改进建议：
{suggestions_text}

处理策略：{plan.get('strategy_name', '标准处理')}

请严格按照以下四部分结构生成文档：

1. **相关反向链接**：基于提取的概念，生成可能相关的主题和概念链接
2. **相关概念**：使用[[双链]]格式列出所有重要概念及其定义
3. **详细内容**：原始输入内容完全保留，不做任何修改或删减
4. **扩展知识**：从原始内容中识别出的可以深入学习的知识点和相关领域

注意：
- 标题应该基于内容分析的主题
- 反向链接应该基于核心概念生成相关主题
- 详细内容部分必须完全保留原始输入
- 扩展知识应该提取文章中涉及但未详细展开的知识点"""
                    }
                ],
                max_tokens=4000
            )
            
            logger.info(f"Pro模型最终合成完成，长度: {len(response)}")
            
            # 验证AI输出是否包含所有必需的部分
            required_sections = ["## 相关反向链接", "## 相关概念", "## 详细内容", "## 扩展知识"]
            missing_sections = []
            for section in required_sections:
                if section not in response:
                    missing_sections.append(section)
            
            if missing_sections:
                logger.warning(f"AI输出缺少以下部分: {missing_sections}, 使用fallback方法")
                return self._create_fallback_structure(content, concepts, analysis)
            
            # 验证详细内容部分是否包含原始内容
            if content.strip() not in response:
                logger.warning("AI输出没有完整保留原始内容，使用fallback方法")
                return self._create_fallback_structure(content, concepts, analysis)
            
            return response
            
        except Exception as e:
            logger.error(f"Pro模型最终合成失败: {e}")
            # 降级到fallback
            return self._create_fallback_structure(content, concepts, analysis)
    
    # === 向后兼容和降级机制 ===
    
    async def _fallback_to_simple_processing(self, content: str, content_type: str, metadata: Dict, options: Dict) -> Dict[str, Any]:
        """降级到简化处理"""
        logger.info("降级到SimpleKnowledgeProcessor处理")
        
        # 这里需要导入SimpleKnowledgeProcessor，但为了避免循环导入，我们先返回基础结果
        return {
            "success": False,
            "error": "AI编排处理失败，需要简化处理",
            "result": {
                "content": self._create_fallback_structure(content, [], {}),
                "concepts": [],
                "analysis": {"main_topic": "处理失败", "complexity": "unknown"}
            }
        }
    
    # === 辅助方法 ===
    
    def _record_execution_history(self, content: str, plan: Dict, results: Dict, final_content: str):
        """记录执行历史 - 增强版本"""
        current_time = time.time()
        success = "error" not in results
        
        # 传统内存记录 (保持兼容性)
        history_record = {
            "timestamp": current_time,
            "content_length": len(content),
            "strategy_name": plan.get("strategy_name", "unknown"),
            "tools_used": self._get_tools_used(plan),
            "success": success,
            "quality_score": results.get("quality_score", 0.8)
        }
        
        self.execution_history.append(history_record)
        
        # 保持历史记录数量
        if len(self.execution_history) > 50:
            self.execution_history = self.execution_history[-50:]
        
        # 新增：详细的策略历史记录
        if self.strategy_history:
            try:
                # 分析内容特征
                content_features = self._analyze_content_features(content)
                
                # 创建详细的执行记录
                execution_record = ExecutionRecord(
                    timestamp=current_time,
                    session_id=str(uuid.uuid4())[:8],
                    content_type=content_features.get("content_type", "unknown"),
                    content_length=len(content),
                    content_complexity=content_features.get("complexity", "medium"),
                    content_language=content_features.get("language", "mixed"),
                    technical_density=content_features.get("technical_density", 0.5),
                    content_hash=hashlib.md5(content.encode()).hexdigest()[:16],
                    strategy_name=plan.get("strategy_name", "unknown"),
                    strategy_version="1.0",
                    tools_selected=self._get_tools_used(plan),
                    tool_sequence=self._get_tool_sequence(plan),
                    success=success,
                    processing_time=plan.get("processing_time", 0.0),
                    quality_score=results.get("quality_score", 0.8),
                    concepts_extracted=len(results.get("concepts", [])),
                    links_created=len(results.get("links", [])),
                    ai_calls_count=self._count_ai_calls(plan),
                    token_usage=plan.get("token_usage", 0),
                    cost_estimate=plan.get("cost_estimate", 0.0),
                    fallback_triggered=plan.get("fallback_used", False),
                    error_type=results.get("error_type") if not success else None,
                    error_stage=results.get("error_stage") if not success else None,
                    error_details=results.get("error_details") if not success else None
                )
                
                # 记录到数据库
                record_id = self.strategy_history.record_execution(execution_record)
                logger.debug(f"策略执行历史已记录，ID: {record_id}")
                
            except Exception as e:
                logger.error(f"记录策略执行历史失败: {e}")
    
    def _get_tools_used(self, plan: Dict) -> List[str]:
        """获取使用的工具列表"""
        tools_used = []
        for phase in plan.get("execution_phases", []):
            tools_used.extend(phase.get("tools", []))
        return tools_used
    
    def _count_flash_calls(self, plan: Dict) -> int:
        """统计Flash模型调用次数"""
        count = 1  # 计划制定本身
        for tool_name in self._get_tools_used(plan):
            if tool_name in self.tool_registry:
                if self.tool_registry[tool_name].get("model") == self.flash_model:
                    count += 1
        return count
    
    def _count_pro_calls(self, plan: Dict) -> int:
        """统计Pro模型调用次数"""
        count = 0
        for tool_name in self._get_tools_used(plan):
            if tool_name in self.tool_registry:
                if self.tool_registry[tool_name].get("model") == self.pro_model:
                    count += 1
        return count
    
    async def _create_strategy_based_plan(self, content: str, content_type: str, 
                                        strategy_name: str, content_features: Dict[str, Any]) -> Dict[str, Any]:
        """基于策略优化器推荐的策略创建处理计划"""
        try:
            # 获取策略定义
            strategy_def = strategy_registry.get_strategy(strategy_name)
            if not strategy_def:
                logger.warning(f"策略定义不存在: {strategy_name}，使用AI规划")
                return await self._ai_create_processing_plan(content, content_type)
            
            logger.info(f"基于策略 '{strategy_name}' 创建处理计划")
            
            # 基于策略定义创建执行计划
            execution_phases = []
            tools_to_use = []
            
            # 分析阶段
            analysis_tools = [tool for tool in strategy_def.tools if tool.phase == "analysis"]
            if analysis_tools:
                phase_tools = [tool.name for tool in analysis_tools]
                execution_phases.append({
                    "phase": "analysis",
                    "tools": phase_tools,
                    "description": f"使用 {strategy_def.description} 进行内容分析"
                })
                tools_to_use.extend(phase_tools)
            
            # 提取阶段
            extraction_tools = [tool for tool in strategy_def.tools if tool.phase == "extraction"]
            if extraction_tools:
                phase_tools = [tool.name for tool in extraction_tools]
                execution_phases.append({
                    "phase": "extraction",
                    "tools": phase_tools,
                    "description": f"概念和信息提取"
                })
                tools_to_use.extend(phase_tools)
            
            # 增强阶段
            enhancement_tools = [tool for tool in strategy_def.tools if tool.phase == "enhancement"]
            if enhancement_tools:
                phase_tools = [tool.name for tool in enhancement_tools]
                execution_phases.append({
                    "phase": "enhancement",
                    "tools": phase_tools,
                    "description": f"内容增强和关系建立"
                })
                tools_to_use.extend(phase_tools)
            
            # 合成阶段
            synthesis_tools = [tool for tool in strategy_def.tools if tool.phase == "synthesis"]
            if synthesis_tools:
                phase_tools = [tool.name for tool in synthesis_tools]
                execution_phases.append({
                    "phase": "synthesis",
                    "tools": phase_tools,
                    "description": f"最终结构化和格式化"
                })
                tools_to_use.extend(phase_tools)
            
            # 如果没有找到合适的工具，添加基础工具
            if not tools_to_use:
                logger.warning(f"策略 {strategy_name} 没有可用工具，添加基础工具")
                execution_phases = [
                    {
                        "phase": "analysis",
                        "tools": ["basic_content_analyzer"],
                        "description": "基础内容分析"
                    },
                    {
                        "phase": "extraction", 
                        "tools": ["general_concept_extractor"],
                        "description": "通用概念提取"
                    },
                    {
                        "phase": "synthesis",
                        "tools": ["template_structurer"],
                        "description": "模板结构化"
                    }
                ]
                tools_to_use = ["basic_content_analyzer", "general_concept_extractor", "template_structurer"]
            
            # 创建处理计划
            processing_plan = {
                "strategy_name": strategy_name,
                "strategy_description": strategy_def.description,
                "optimization_method": "strategy_based",
                "content_analysis": content_features,
                "execution_phases": execution_phases,
                "tools_to_use": tools_to_use,
                "expected_quality_range": strategy_def.expected_quality_range,
                "expected_time_range": strategy_def.expected_time_range,
                "fallback_strategy": strategy_def.fallback_strategy,
                "priority": strategy_def.priority.value,
                "content_compatibility": {
                    "applicable_types": [t.value for t in strategy_def.applicable_content_types],
                    "complexity_range": [c.value for c in strategy_def.complexity_range]
                }
            }
            
            logger.info(f"策略化计划创建完成，将使用 {len(tools_to_use)} 个工具")
            return processing_plan
            
        except Exception as e:
            logger.error(f"基于策略创建计划失败: {e}，回退到AI规划")
            return await self._ai_create_processing_plan(content, content_type)
    
    def _analyze_content_features(self, content: str) -> Dict[str, Any]:
        """分析内容特征用于策略历史记录"""
        try:
            # 基础特征
            content_length = len(content)
            
            # 内容类型检测
            if "用户：" in content and "助手：" in content:
                content_type = "conversation"
            elif "# " in content or "## " in content:
                content_type = "markdown"
            elif "http" in content.lower():
                content_type = "url_content"
            else:
                content_type = "text"
            
            # 复杂度评估
            if content_length < 500:
                complexity = "simple"
            elif content_length < 2000:
                complexity = "medium"
            else:
                complexity = "complex"
            
            # 技术密度评估 (简化版本)
            technical_keywords = ["API", "函数", "代码", "算法", "数据库", "架构", "接口", "模块"]
            technical_count = sum(1 for keyword in technical_keywords if keyword in content)
            technical_density = min(technical_count / 10.0, 1.0)
            
            # 语言检测
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
            english_chars = len(re.findall(r'[a-zA-Z]', content))
            
            if chinese_chars > english_chars * 2:
                language = "chinese"
            elif english_chars > chinese_chars * 2:
                language = "english"
            else:
                language = "mixed"
            
            return {
                "content_type": content_type,
                "complexity": complexity,
                "technical_density": technical_density,
                "language": language,
                "length_category": self._get_length_category(content_length)
            }
            
        except Exception as e:
            logger.error(f"分析内容特征失败: {e}")
            return {
                "content_type": "unknown",
                "complexity": "medium",
                "technical_density": 0.5,
                "language": "mixed",
                "length_category": "medium"
            }
    
    def _get_length_category(self, length: int) -> str:
        """获取内容长度分类"""
        if length < 300:
            return "short"
        elif length < 1500:
            return "medium"
        elif length < 5000:
            return "long"
        else:
            return "very_long"
    
    def _get_tool_sequence(self, plan: Dict) -> List[str]:
        """获取工具执行序列"""
        try:
            tool_sequence = []
            for phase in plan.get("execution_phases", []):
                phase_tools = phase.get("tools", [])
                tool_sequence.extend(phase_tools)
            return tool_sequence
        except:
            return self._get_tools_used(plan)
    
    def _count_ai_calls(self, plan: Dict) -> int:
        """计算AI调用总次数"""
        try:
            flash_calls = self._count_flash_calls(plan)
            pro_calls = self._count_pro_calls(plan)
            return flash_calls + pro_calls
        except:
            return 0
    
    def _generate_enhanced_statistics(self, results: Dict) -> Dict[str, Any]:
        """生成增强的统计信息"""
        return {
            "processing_method": "AI智能编排",
            "tools_count": len([k for k in results.keys() if k.endswith("_result")]),
            "concept_count": len(results.get("concepts", [])),
            "has_analysis": "analysis" in results,
            "processing_quality": "enhanced"
        }

    def _format_available_tools(self) -> str:
        """格式化可用工具列表"""
        tools_by_phase = {}
        for tool_name, tool_config in self.tool_registry.items():
            phase = tool_config["phase"]
            if phase not in tools_by_phase:
                tools_by_phase[phase] = []
            tools_by_phase[phase].append({
                "name": tool_name,
                "description": tool_config["description"],
                "cost": tool_config["cost"]
            })
        
        formatted = []
        for phase, tools in tools_by_phase.items():
            formatted.append(f"\n{phase.upper()}阶段工具:")
            for tool in tools:
                formatted.append(f"  - {tool['name']}: {tool['description']} (成本: {tool['cost']})")
        
        return "\n".join(formatted)