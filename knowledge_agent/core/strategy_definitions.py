"""
策略定义和分类 - 定义具体的处理策略类型
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ProcessingComplexity(Enum):
    """处理复杂度级别"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ContentType(Enum):
    """内容类型"""
    CONVERSATION = "conversation"
    MARKDOWN = "markdown"
    URL_CONTENT = "url_content"
    TEXT = "text"
    CODE = "code"
    ACADEMIC = "academic"
    NEWS = "news"
    DOCUMENTATION = "documentation"


class StrategyPriority(Enum):
    """策略优先级"""
    SPEED = "speed"          # 速度优先
    QUALITY = "quality"      # 质量优先
    BALANCED = "balanced"    # 平衡
    COST = "cost"           # 成本优先


@dataclass
class ToolConfig:
    """工具配置"""
    name: str
    description: str
    model_required: str  # flash/pro/any
    phase: str          # analysis/extraction/enhancement/synthesis
    cost_level: str     # low/medium/high
    expected_time: float  # 预期执行时间(秒)
    reliability: float   # 可靠性评分(0-1)


@dataclass
class StrategyDefinition:
    """策略定义"""
    name: str
    description: str
    applicable_content_types: List[ContentType]
    complexity_range: List[ProcessingComplexity]
    priority: StrategyPriority
    tools: List[ToolConfig]
    expected_quality_range: tuple  # (min, max)
    expected_time_range: tuple     # (min_seconds, max_seconds)
    min_content_length: int
    max_content_length: int
    success_rate_threshold: float  # 最低成功率要求
    fallback_strategy: Optional[str]
    metadata: Dict[str, Any]


class StrategyRegistry:
    """策略注册表"""
    
    def __init__(self):
        self.strategies: Dict[str, StrategyDefinition] = {}
        self._initialize_predefined_strategies()
    
    def _initialize_predefined_strategies(self):
        """初始化预定义策略"""
        
        # 1. 轻量级快速处理策略
        self.strategies["lightweight_processing"] = StrategyDefinition(
            name="轻量级处理",
            description="适用于简单内容的快速处理，注重速度和效率",
            applicable_content_types=[
                ContentType.TEXT, ContentType.CONVERSATION
            ],
            complexity_range=[ProcessingComplexity.SIMPLE, ProcessingComplexity.MEDIUM],
            priority=StrategyPriority.SPEED,
            tools=[
                ToolConfig("basic_content_analyzer", "基础内容分析", "flash", "analysis", "low", 2.0, 0.9),
                ToolConfig("general_concept_extractor", "通用概念提取", "flash", "extraction", "medium", 3.0, 0.85),
                ToolConfig("template_structurer", "模板结构化", "flash", "synthesis", "low", 1.5, 0.9)
            ],
            expected_quality_range=(0.6, 0.8),
            expected_time_range=(5, 15),
            min_content_length=50,
            max_content_length=2000,
            success_rate_threshold=0.85,
            fallback_strategy="minimal_processing",
            metadata={
                "use_cases": ["短文本", "简单对话", "快速笔记"],
                "optimization_focus": "速度和成本",
                "typical_scenarios": ["日常记录", "简单问答"]
            }
        )
        
        # 2. 全面深度分析策略
        self.strategies["comprehensive_analysis"] = StrategyDefinition(
            name="全面分析",
            description="深度分析复杂内容，注重质量和完整性",
            applicable_content_types=[
                ContentType.ACADEMIC, ContentType.DOCUMENTATION, ContentType.MARKDOWN
            ],
            complexity_range=[ProcessingComplexity.COMPLEX, ProcessingComplexity.VERY_COMPLEX],
            priority=StrategyPriority.QUALITY,
            tools=[
                ToolConfig("content_type_detector", "内容类型检测", "flash", "analysis", "high", 8.0, 0.95),
                ToolConfig("academic_concept_extractor", "学术概念提取", "flash", "extraction", "high", 12.0, 0.9),
                ToolConfig("relationship_analyzer", "关系分析", "flash", "enhancement", "high", 15.0, 0.85),
                ToolConfig("content_quality_validator", "质量验证", "pro", "enhancement", "medium", 5.0, 0.9),
                ToolConfig("advanced_markdown_structurer", "高级结构化", "pro", "synthesis", "high", 10.0, 0.9)
            ],
            expected_quality_range=(0.8, 0.95),
            expected_time_range=(30, 90),
            min_content_length=1000,
            max_content_length=50000,
            success_rate_threshold=0.75,
            fallback_strategy="standard_processing",
            metadata={
                "use_cases": ["学术论文", "技术文档", "深度分析"],
                "optimization_focus": "质量和完整性",
                "typical_scenarios": ["研究文献", "技术规范", "复杂报告"]
            }
        )
        
        # 3. 对话专门化策略
        self.strategies["conversation_specialized"] = StrategyDefinition(
            name="对话专门化",
            description="专门处理对话记录，保持对话结构和语调",
            applicable_content_types=[ContentType.CONVERSATION],
            complexity_range=[ProcessingComplexity.SIMPLE, ProcessingComplexity.MEDIUM, ProcessingComplexity.COMPLEX],
            priority=StrategyPriority.BALANCED,
            tools=[
                ToolConfig("dialogue_parser", "对话解析器", "flash", "analysis", "medium", 4.0, 0.9),
                ToolConfig("general_concept_extractor", "通用概念提取", "flash", "extraction", "medium", 5.0, 0.85),
                ToolConfig("summary_generator", "摘要生成器", "flash", "enhancement", "medium", 8.0, 0.8),
                ToolConfig("specialized_formatter", "专门化格式化", "pro", "synthesis", "medium", 3.0, 0.85)
            ],
            expected_quality_range=(0.7, 0.9),
            expected_time_range=(10, 30),
            min_content_length=100,
            max_content_length=10000,
            success_rate_threshold=0.8,
            fallback_strategy="lightweight_processing",
            metadata={
                "use_cases": ["AI对话", "会议记录", "访谈记录"],
                "optimization_focus": "对话结构和语调保持",
                "typical_scenarios": ["客服对话", "技术讨论", "学习问答"]
            }
        )
        
        # 4. 代码和技术文档策略
        self.strategies["technical_specialized"] = StrategyDefinition(
            name="技术专门化",
            description="专门处理代码和技术文档，识别技术概念和API",
            applicable_content_types=[ContentType.CODE, ContentType.DOCUMENTATION],
            complexity_range=[ProcessingComplexity.MEDIUM, ProcessingComplexity.COMPLEX],
            priority=StrategyPriority.QUALITY,
            tools=[
                ToolConfig("complexity_assessor", "复杂度评估器", "flash", "analysis", "high", 10.0, 0.9),
                ToolConfig("technical_concept_extractor", "技术概念提取器", "flash", "extraction", "high", 8.0, 0.85),
                ToolConfig("code_block_extractor", "代码块提取器", "flash", "enhancement", "high", 12.0, 0.8),
                ToolConfig("advanced_markdown_structurer", "高级结构化", "pro", "synthesis", "medium", 5.0, 0.85)
            ],
            expected_quality_range=(0.75, 0.9),
            expected_time_range=(20, 60),
            min_content_length=200,
            max_content_length=20000,
            success_rate_threshold=0.75,
            fallback_strategy="comprehensive_analysis",
            metadata={
                "use_cases": ["代码文档", "API文档", "技术教程"],
                "optimization_focus": "技术准确性和结构化",
                "typical_scenarios": ["开发文档", "技术博客", "代码注释"]
            }
        )
        
        # 5. 标准平衡策略
        self.strategies["standard_processing"] = StrategyDefinition(
            name="标准处理",
            description="平衡质量和速度的通用处理策略",
            applicable_content_types=[t for t in ContentType],  # 支持所有类型
            complexity_range=[ProcessingComplexity.SIMPLE, ProcessingComplexity.MEDIUM, ProcessingComplexity.COMPLEX],
            priority=StrategyPriority.BALANCED,
            tools=[
                ToolConfig("basic_content_analyzer", "基础内容分析", "flash", "analysis", "medium", 5.0, 0.85),
                ToolConfig("general_concept_extractor", "通用概念提取", "flash", "extraction", "medium", 6.0, 0.8),
                ToolConfig("relationship_analyzer", "关系分析器", "flash", "enhancement", "medium", 4.0, 0.8),
                ToolConfig("advanced_markdown_structurer", "高级结构化", "pro", "synthesis", "medium", 4.0, 0.85)
            ],
            expected_quality_range=(0.65, 0.85),
            expected_time_range=(15, 35),
            min_content_length=100,
            max_content_length=15000,
            success_rate_threshold=0.8,
            fallback_strategy="lightweight_processing",
            metadata={
                "use_cases": ["通用文本", "混合内容", "中等复杂度"],
                "optimization_focus": "平衡性能",
                "typical_scenarios": ["新闻文章", "博客文章", "一般文档"]
            }
        )
        
        # 6. 最小化处理策略 (备用)
        self.strategies["minimal_processing"] = StrategyDefinition(
            name="最小处理",
            description="最基础的处理策略，确保基本功能可用",
            applicable_content_types=[t for t in ContentType],
            complexity_range=[t for t in ProcessingComplexity],
            priority=StrategyPriority.SPEED,
            tools=[
                ToolConfig("basic_content_analyzer", "基础内容分析", "flash", "analysis", "low", 1.0, 0.95),
                ToolConfig("regex_concept_extractor", "正则概念提取", "flash", "extraction", "low", 2.0, 0.9),
                ToolConfig("template_structurer", "模板结构化", "flash", "synthesis", "low", 1.0, 0.95)
            ],
            expected_quality_range=(0.4, 0.6),
            expected_time_range=(3, 8),
            min_content_length=1,
            max_content_length=100000,
            success_rate_threshold=0.9,
            fallback_strategy=None,  # 最后的备用策略
            metadata={
                "use_cases": ["紧急备用", "系统资源不足", "网络问题"],
                "optimization_focus": "可靠性和容错",
                "typical_scenarios": ["系统降级", "错误恢复"]
            }
        )
    
    def get_strategy(self, strategy_name: str) -> Optional[StrategyDefinition]:
        """获取策略定义"""
        return self.strategies.get(strategy_name)
    
    def get_applicable_strategies(self, content_type: ContentType, 
                                complexity: ProcessingComplexity,
                                content_length: int) -> List[StrategyDefinition]:
        """获取适用的策略列表"""
        applicable = []
        
        for strategy in self.strategies.values():
            # 检查内容类型适用性
            if content_type not in strategy.applicable_content_types:
                continue
            
            # 检查复杂度适用性
            if complexity not in strategy.complexity_range:
                continue
            
            # 检查内容长度适用性
            if not (strategy.min_content_length <= content_length <= strategy.max_content_length):
                continue
            
            applicable.append(strategy)
        
        return applicable
    
    def get_strategies_by_priority(self, priority: StrategyPriority) -> List[StrategyDefinition]:
        """根据优先级获取策略"""
        return [s for s in self.strategies.values() if s.priority == priority]
    
    def get_fallback_chain(self, strategy_name: str) -> List[str]:
        """获取策略的完整降级链"""
        chain = [strategy_name]
        current = strategy_name
        
        while current and current in self.strategies:
            strategy = self.strategies[current]
            if strategy.fallback_strategy and strategy.fallback_strategy not in chain:
                chain.append(strategy.fallback_strategy)
                current = strategy.fallback_strategy
            else:
                break
        
        return chain
    
    def add_custom_strategy(self, strategy: StrategyDefinition):
        """添加自定义策略"""
        self.strategies[strategy.name] = strategy
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """获取策略统计信息"""
        stats = {
            "total_strategies": len(self.strategies),
            "by_priority": {},
            "by_content_type": {},
            "by_complexity": {},
            "average_expected_quality": 0.0,
            "average_expected_time": 0.0
        }
        
        # 按优先级统计
        for priority in StrategyPriority:
            stats["by_priority"][priority.value] = len(self.get_strategies_by_priority(priority))
        
        # 按内容类型统计
        for content_type in ContentType:
            count = sum(1 for s in self.strategies.values() 
                       if content_type in s.applicable_content_types)
            stats["by_content_type"][content_type.value] = count
        
        # 按复杂度统计
        for complexity in ProcessingComplexity:
            count = sum(1 for s in self.strategies.values() 
                       if complexity in s.complexity_range)
            stats["by_complexity"][complexity.value] = count
        
        # 平均期望值
        if self.strategies:
            total_quality = sum(s.expected_quality_range[1] for s in self.strategies.values())
            total_time = sum(s.expected_time_range[1] for s in self.strategies.values())
            stats["average_expected_quality"] = total_quality / len(self.strategies)
            stats["average_expected_time"] = total_time / len(self.strategies)
        
        return stats
    
    def validate_strategy(self, strategy: StrategyDefinition) -> List[str]:
        """验证策略定义的完整性"""
        errors = []
        
        if not strategy.name:
            errors.append("策略名称不能为空")
        
        if not strategy.tools:
            errors.append("策略必须包含至少一个工具")
        
        if strategy.expected_quality_range[0] > strategy.expected_quality_range[1]:
            errors.append("质量范围最小值不能大于最大值")
        
        if strategy.expected_time_range[0] > strategy.expected_time_range[1]:
            errors.append("时间范围最小值不能大于最大值")
        
        if strategy.min_content_length > strategy.max_content_length:
            errors.append("内容长度最小值不能大于最大值")
        
        if not (0 <= strategy.success_rate_threshold <= 1):
            errors.append("成功率阈值必须在0-1之间")
        
        # 检查工具配置
        required_phases = {"analysis", "extraction", "synthesis"}
        tool_phases = {tool.phase for tool in strategy.tools}
        missing_phases = required_phases - tool_phases
        
        if missing_phases:
            errors.append(f"策略缺少必需的处理阶段: {missing_phases}")
        
        return errors
    
    def get_recommended_strategy(self, content_features: Dict[str, Any]) -> Optional[str]:
        """基于内容特征推荐策略"""
        try:
            content_type = ContentType(content_features.get("content_type", "text"))
            complexity = ProcessingComplexity(content_features.get("complexity", "medium"))
            content_length = content_features.get("content_length", 1000)
            priority = StrategyPriority(content_features.get("priority", "balanced"))
            
            # 获取适用策略
            applicable = self.get_applicable_strategies(content_type, complexity, content_length)
            
            if not applicable:
                return "standard_processing"  # 默认策略
            
            # 按优先级过滤
            preferred = [s for s in applicable if s.priority == priority]
            if not preferred:
                preferred = applicable
            
            # 选择最匹配的策略
            best_strategy = max(preferred, key=lambda s: self._calculate_match_score(s, content_features))
            
            return best_strategy.name
            
        except (ValueError, KeyError):
            return "standard_processing"
    
    def _calculate_match_score(self, strategy: StrategyDefinition, 
                             content_features: Dict[str, Any]) -> float:
        """计算策略与内容特征的匹配分数"""
        score = 0.0
        
        # 内容长度匹配度
        length = content_features.get("content_length", 1000)
        length_range = strategy.max_content_length - strategy.min_content_length
        if length_range > 0:
            length_fit = 1.0 - abs(length - (strategy.min_content_length + length_range / 2)) / length_range
            score += max(0, length_fit) * 0.3
        
        # 复杂度匹配度
        complexity = content_features.get("complexity", "medium")
        if ProcessingComplexity(complexity) in strategy.complexity_range:
            score += 0.4
        
        # 优先级匹配度
        priority = content_features.get("priority", "balanced")
        if strategy.priority.value == priority:
            score += 0.3
        
        return score


# 全局策略注册表实例
strategy_registry = StrategyRegistry()