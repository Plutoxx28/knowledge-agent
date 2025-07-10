"""
策略学习机制 - 基于历史性能的持续学习和优化
"""
import logging
import time
import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
from datetime import datetime, timedelta

from .strategy_history import StrategyHistoryDB
from .history_analyzer import HistoryAnalyzer
from .strategy_evaluator import StrategyEvaluator
from .strategy_optimizer import StrategyOptimizer
from .strategy_definitions import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class LearningReport:
    """学习报告"""
    timestamp: float
    learning_period: str
    processed_executions: int
    identified_patterns: List[Dict[str, Any]]
    optimization_adjustments: Dict[str, Any]
    performance_improvements: Dict[str, float]
    recommendations: List[str]
    next_learning_schedule: float


@dataclass
class PerformancePattern:
    """性能模式"""
    pattern_type: str  # success_pattern, failure_pattern, optimization_opportunity
    pattern_description: str
    confidence_score: float
    frequency: int
    impact_score: float
    conditions: Dict[str, Any]
    recommended_actions: List[str]


class StrategyLearner:
    """策略学习器"""
    
    def __init__(self, 
                 history_db: Optional[StrategyHistoryDB] = None,
                 strategy_optimizer: Optional[StrategyOptimizer] = None):
        self.history_db = history_db or StrategyHistoryDB()
        self.history_analyzer = HistoryAnalyzer(self.history_db)
        self.strategy_evaluator = StrategyEvaluator(self.history_db)
        self.strategy_optimizer = strategy_optimizer or StrategyOptimizer(self.history_db)
        self.strategy_registry = StrategyRegistry()
        
        # 学习配置
        self.learning_config = {
            "min_executions_for_learning": 10,    # 最少执行次数才开始学习
            "learning_interval_hours": 24,         # 学习间隔（小时）
            "pattern_confidence_threshold": 0.7,   # 模式识别置信度阈值
            "adaptation_rate": 0.1,               # 适应速度
            "performance_memory_window": 100,      # 性能记忆窗口
            "pattern_detection_window": 50,       # 模式检测窗口
            "min_pattern_frequency": 3            # 最小模式频率
        }
        
        # 学习状态
        self.learning_state = {
            "last_learning_time": 0,
            "total_learning_cycles": 0,
            "identified_patterns": [],
            "performance_baselines": {},
            "adaptation_history": deque(maxlen=20)
        }
        
        # 性能基线
        self.performance_baselines = {}
        
        # 学习历史
        self.learning_history = deque(maxlen=10)
    
    async def continuous_learning_cycle(self) -> LearningReport:
        """持续学习周期"""
        try:
            learning_start_time = time.time()
            logger.info("开始策略学习周期")
            
            # 检查是否需要学习
            if not self._should_perform_learning():
                return self._create_skip_report("学习间隔未到或数据不足")
            
            # 收集最新执行数据
            recent_executions = self._collect_recent_executions()
            
            if len(recent_executions) < self.learning_config["min_executions_for_learning"]:
                return self._create_skip_report(f"执行数据不足：{len(recent_executions)}")
            
            # 第一阶段：模式识别
            identified_patterns = await self._identify_performance_patterns(recent_executions)
            
            # 第二阶段：性能基线更新
            self._update_performance_baselines(recent_executions)
            
            # 第三阶段：策略优化调整
            optimization_adjustments = await self._perform_strategy_optimization(identified_patterns)
            
            # 第四阶段：评估改进效果
            performance_improvements = self._evaluate_performance_improvements(recent_executions)
            
            # 第五阶段：生成学习建议
            recommendations = self._generate_learning_recommendations(
                identified_patterns, optimization_adjustments, performance_improvements
            )
            
            # 更新学习状态
            self._update_learning_state(identified_patterns, optimization_adjustments)
            
            # 生成学习报告
            learning_report = LearningReport(
                timestamp=learning_start_time,
                learning_period="24h",
                processed_executions=len(recent_executions),
                identified_patterns=[pattern.__dict__ for pattern in identified_patterns],
                optimization_adjustments=optimization_adjustments,
                performance_improvements=performance_improvements,
                recommendations=recommendations,
                next_learning_schedule=learning_start_time + (self.learning_config["learning_interval_hours"] * 3600)
            )
            
            self.learning_history.append(learning_report)
            
            logger.info(f"学习周期完成，识别到 {len(identified_patterns)} 个模式，"
                       f"进行了 {len(optimization_adjustments)} 项优化调整")
            
            return learning_report
            
        except Exception as e:
            logger.error(f"策略学习周期失败: {e}")
            return self._create_error_report(str(e))
    
    async def _identify_performance_patterns(self, executions: List[Dict[str, Any]]) -> List[PerformancePattern]:
        """识别性能模式"""
        patterns = []
        
        try:
            # 成功模式识别
            success_patterns = self._identify_success_patterns(executions)
            patterns.extend(success_patterns)
            
            # 失败模式识别
            failure_patterns = self._identify_failure_patterns(executions)
            patterns.extend(failure_patterns)
            
            # 优化机会识别
            optimization_patterns = self._identify_optimization_opportunities(executions)
            patterns.extend(optimization_patterns)
            
            # 时间模式识别
            temporal_patterns = self._identify_temporal_patterns(executions)
            patterns.extend(temporal_patterns)
            
            # 内容类型模式识别
            content_patterns = self._identify_content_type_patterns(executions)
            patterns.extend(content_patterns)
            
            # 过滤低置信度模式
            high_confidence_patterns = [
                p for p in patterns 
                if p.confidence_score >= self.learning_config["pattern_confidence_threshold"]
            ]
            
            logger.info(f"识别到 {len(patterns)} 个模式，{len(high_confidence_patterns)} 个高置信度模式")
            
            return high_confidence_patterns
            
        except Exception as e:
            logger.error(f"模式识别失败: {e}")
            return []
    
    def _identify_success_patterns(self, executions: List[Dict[str, Any]]) -> List[PerformancePattern]:
        """识别成功模式"""
        patterns = []
        
        # 找出高质量的成功案例
        high_quality_executions = [
            ex for ex in executions 
            if ex.get("success", False) and ex.get("quality_score", 0) > 0.8
        ]
        
        if len(high_quality_executions) < self.learning_config["min_pattern_frequency"]:
            return patterns
        
        # 分析成功案例的共同特征
        
        # 策略成功模式
        strategy_success = defaultdict(list)
        for ex in high_quality_executions:
            strategy = ex.get("strategy_name", "unknown")
            strategy_success[strategy].append(ex)
        
        for strategy, cases in strategy_success.items():
            if len(cases) >= self.learning_config["min_pattern_frequency"]:
                avg_quality = np.mean([case["quality_score"] for case in cases])
                avg_time = np.mean([case.get("processing_time", 0) for case in cases])
                
                pattern = PerformancePattern(
                    pattern_type="success_pattern",
                    pattern_description=f"策略 '{strategy}' 在相似内容上表现优异",
                    confidence_score=min(len(cases) / 10.0, 1.0),  # 基于样本数量的置信度
                    frequency=len(cases),
                    impact_score=avg_quality,
                    conditions={
                        "strategy_name": strategy,
                        "avg_quality": avg_quality,
                        "avg_processing_time": avg_time,
                        "sample_count": len(cases)
                    },
                    recommended_actions=[
                        f"优先推荐策略 '{strategy}' 用于类似内容",
                        f"可以提高策略 '{strategy}' 的选择权重"
                    ]
                )
                patterns.append(pattern)
        
        # 内容类型成功模式
        content_type_success = defaultdict(list)
        for ex in high_quality_executions:
            content_type = ex.get("content_type", "unknown")
            content_type_success[content_type].append(ex)
        
        for content_type, cases in content_type_success.items():
            if len(cases) >= self.learning_config["min_pattern_frequency"]:
                # 找出该内容类型最成功的策略
                strategy_performance = defaultdict(list)
                for case in cases:
                    strategy = case.get("strategy_name", "unknown")
                    strategy_performance[strategy].append(case["quality_score"])
                
                best_strategy = max(strategy_performance.items(), 
                                  key=lambda x: np.mean(x[1]))
                
                pattern = PerformancePattern(
                    pattern_type="success_pattern",
                    pattern_description=f"内容类型 '{content_type}' 使用策略 '{best_strategy[0]}' 效果最佳",
                    confidence_score=0.8,
                    frequency=len(cases),
                    impact_score=np.mean(best_strategy[1]),
                    conditions={
                        "content_type": content_type,
                        "best_strategy": best_strategy[0],
                        "avg_quality": np.mean(best_strategy[1])
                    },
                    recommended_actions=[
                        f"对 '{content_type}' 类型内容优先推荐策略 '{best_strategy[0]}'"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _identify_failure_patterns(self, executions: List[Dict[str, Any]]) -> List[PerformancePattern]:
        """识别失败模式"""
        patterns = []
        
        # 找出失败或低质量的案例
        poor_executions = [
            ex for ex in executions 
            if not ex.get("success", True) or ex.get("quality_score", 1.0) < 0.5
        ]
        
        if len(poor_executions) < self.learning_config["min_pattern_frequency"]:
            return patterns
        
        # 分析失败案例的共同特征
        
        # 策略失败模式
        strategy_failures = defaultdict(list)
        for ex in poor_executions:
            strategy = ex.get("strategy_name", "unknown")
            strategy_failures[strategy].append(ex)
        
        for strategy, cases in strategy_failures.items():
            if len(cases) >= self.learning_config["min_pattern_frequency"]:
                failure_rate = len(cases) / len([e for e in executions if e.get("strategy_name") == strategy])
                
                if failure_rate > 0.3:  # 失败率超过30%
                    pattern = PerformancePattern(
                        pattern_type="failure_pattern",
                        pattern_description=f"策略 '{strategy}' 失败率过高 ({failure_rate:.1%})",
                        confidence_score=min(failure_rate * 2, 1.0),
                        frequency=len(cases),
                        impact_score=1.0 - failure_rate,
                        conditions={
                            "strategy_name": strategy,
                            "failure_rate": failure_rate,
                            "failure_count": len(cases)
                        },
                        recommended_actions=[
                            f"降低策略 '{strategy}' 的选择权重",
                            f"调查策略 '{strategy}' 失败的具体原因",
                            "考虑优化或替换该策略"
                        ]
                    )
                    patterns.append(pattern)
        
        # 内容复杂度失败模式
        complexity_failures = defaultdict(list)
        for ex in poor_executions:
            complexity = ex.get("content_complexity", "unknown")
            complexity_failures[complexity].append(ex)
        
        for complexity, cases in complexity_failures.items():
            if len(cases) >= self.learning_config["min_pattern_frequency"]:
                total_complexity_cases = [e for e in executions if e.get("content_complexity") == complexity]
                failure_rate = len(cases) / len(total_complexity_cases) if total_complexity_cases else 0
                
                if failure_rate > 0.4:  # 某复杂度失败率过高
                    pattern = PerformancePattern(
                        pattern_type="failure_pattern",
                        pattern_description=f"复杂度 '{complexity}' 内容处理失败率高 ({failure_rate:.1%})",
                        confidence_score=0.7,
                        frequency=len(cases),
                        impact_score=1.0 - failure_rate,
                        conditions={
                            "content_complexity": complexity,
                            "failure_rate": failure_rate
                        },
                        recommended_actions=[
                            f"为 '{complexity}' 复杂度内容开发专门的处理策略",
                            "提高该复杂度内容的质量阈值"
                        ]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _identify_optimization_opportunities(self, executions: List[Dict[str, Any]]) -> List[PerformancePattern]:
        """识别优化机会"""
        patterns = []
        
        # 处理时间优化机会
        slow_executions = [ex for ex in executions if ex.get("processing_time", 0) > 30]
        if len(slow_executions) >= self.learning_config["min_pattern_frequency"]:
            avg_time = np.mean([ex["processing_time"] for ex in slow_executions])
            
            # 找出导致缓慢的主要策略
            strategy_times = defaultdict(list)
            for ex in slow_executions:
                strategy = ex.get("strategy_name", "unknown")
                strategy_times[strategy].append(ex["processing_time"])
            
            if strategy_times:
                slowest_strategy = max(strategy_times.items(), key=lambda x: np.mean(x[1]))
                
                pattern = PerformancePattern(
                    pattern_type="optimization_opportunity",
                    pattern_description=f"策略 '{slowest_strategy[0]}' 处理时间过长 (平均 {np.mean(slowest_strategy[1]):.1f}秒)",
                    confidence_score=0.8,
                    frequency=len(slow_executions),
                    impact_score=avg_time / 60.0,  # 时间影响分数
                    conditions={
                        "strategy_name": slowest_strategy[0],
                        "avg_processing_time": np.mean(slowest_strategy[1]),
                        "slow_execution_count": len(slow_executions)
                    },
                    recommended_actions=[
                        f"优化策略 '{slowest_strategy[0]}' 的工具选择",
                        "考虑并行处理或工具替换",
                        "为时间敏感任务降低该策略优先级"
                    ]
                )
                patterns.append(pattern)
        
        # 质量提升机会
        medium_quality_executions = [
            ex for ex in executions 
            if ex.get("success", True) and 0.6 <= ex.get("quality_score", 0) <= 0.75
        ]
        
        if len(medium_quality_executions) >= self.learning_config["min_pattern_frequency"]:
            # 找出可能提升质量的策略调整
            pattern = PerformancePattern(
                pattern_type="optimization_opportunity",
                pattern_description=f"有 {len(medium_quality_executions)} 个案例质量中等，存在提升空间",
                confidence_score=0.6,
                frequency=len(medium_quality_executions),
                impact_score=0.5,
                conditions={
                    "medium_quality_count": len(medium_quality_executions),
                    "avg_quality": np.mean([ex["quality_score"] for ex in medium_quality_executions])
                },
                recommended_actions=[
                    "考虑为中等质量案例增加质量增强步骤",
                    "分析高质量案例的成功因素并应用"
                ]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _identify_temporal_patterns(self, executions: List[Dict[str, Any]]) -> List[PerformancePattern]:
        """识别时间模式"""
        patterns = []
        
        # 按时间段分组分析性能
        time_buckets = defaultdict(list)
        
        for ex in executions:
            timestamp = ex.get("timestamp", time.time())
            hour = datetime.fromtimestamp(timestamp).hour
            
            if 6 <= hour < 12:
                time_bucket = "morning"
            elif 12 <= hour < 18:
                time_bucket = "afternoon"
            elif 18 <= hour < 24:
                time_bucket = "evening"
            else:
                time_bucket = "night"
            
            time_buckets[time_bucket].append(ex)
        
        # 找出性能差异显著的时间段
        bucket_performance = {}
        for bucket, cases in time_buckets.items():
            if len(cases) >= 3:
                avg_quality = np.mean([case.get("quality_score", 0) for case in cases])
                success_rate = np.mean([case.get("success", False) for case in cases])
                bucket_performance[bucket] = {"quality": avg_quality, "success_rate": success_rate, "count": len(cases)}
        
        if len(bucket_performance) >= 2:
            # 找出最佳和最差时间段
            best_bucket = max(bucket_performance.items(), key=lambda x: x[1]["quality"])
            worst_bucket = min(bucket_performance.items(), key=lambda x: x[1]["quality"])
            
            quality_diff = best_bucket[1]["quality"] - worst_bucket[1]["quality"]
            
            if quality_diff > 0.1:  # 质量差异超过0.1
                pattern = PerformancePattern(
                    pattern_type="temporal_pattern",
                    pattern_description=f"时间段性能差异：{best_bucket[0]} 表现最佳，{worst_bucket[0]} 表现最差",
                    confidence_score=0.7,
                    frequency=sum(p["count"] for p in bucket_performance.values()),
                    impact_score=quality_diff,
                    conditions={
                        "best_time": best_bucket[0],
                        "worst_time": worst_bucket[0],
                        "quality_difference": quality_diff,
                        "performance_by_time": bucket_performance
                    },
                    recommended_actions=[
                        f"在 {best_bucket[0]} 时段优先处理重要内容",
                        f"在 {worst_bucket[0]} 时段考虑使用更可靠的策略"
                    ]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _identify_content_type_patterns(self, executions: List[Dict[str, Any]]) -> List[PerformancePattern]:
        """识别内容类型模式"""
        patterns = []
        
        # 按内容类型分组分析
        content_type_performance = defaultdict(lambda: {"cases": [], "strategies": defaultdict(list)})
        
        for ex in executions:
            content_type = ex.get("content_type", "unknown")
            strategy = ex.get("strategy_name", "unknown")
            
            content_type_performance[content_type]["cases"].append(ex)
            content_type_performance[content_type]["strategies"][strategy].append(ex)
        
        for content_type, data in content_type_performance.items():
            cases = data["cases"]
            strategies = data["strategies"]
            
            if len(cases) >= self.learning_config["min_pattern_frequency"]:
                # 找出该内容类型的最佳策略组合
                strategy_scores = {}
                for strategy, strategy_cases in strategies.items():
                    if len(strategy_cases) >= 2:
                        avg_quality = np.mean([case.get("quality_score", 0) for case in strategy_cases])
                        success_rate = np.mean([case.get("success", False) for case in strategy_cases])
                        strategy_scores[strategy] = avg_quality * 0.7 + success_rate * 0.3
                
                if len(strategy_scores) >= 2:
                    best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
                    worst_strategy = min(strategy_scores.items(), key=lambda x: x[1])
                    
                    score_diff = best_strategy[1] - worst_strategy[1]
                    
                    if score_diff > 0.15:  # 策略差异显著
                        pattern = PerformancePattern(
                            pattern_type="content_type_pattern",
                            pattern_description=f"内容类型 '{content_type}' 的策略选择显著影响性能",
                            confidence_score=0.8,
                            frequency=len(cases),
                            impact_score=score_diff,
                            conditions={
                                "content_type": content_type,
                                "best_strategy": best_strategy[0],
                                "worst_strategy": worst_strategy[0],
                                "performance_difference": score_diff,
                                "strategy_scores": dict(strategy_scores)
                            },
                            recommended_actions=[
                                f"对 '{content_type}' 类型内容强烈推荐策略 '{best_strategy[0]}'",
                                f"避免对 '{content_type}' 类型内容使用策略 '{worst_strategy[0]}'"
                            ]
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _perform_strategy_optimization(self, patterns: List[PerformancePattern]) -> Dict[str, Any]:
        """执行策略优化"""
        optimization_adjustments = {
            "weight_adjustments": {},
            "threshold_adjustments": {},
            "strategy_priorities": {},
            "new_rules": []
        }
        
        try:
            for pattern in patterns:
                if pattern.pattern_type == "success_pattern":
                    # 提高成功策略的权重
                    strategy = pattern.conditions.get("strategy_name")
                    if strategy:
                        current_weight = self.strategy_optimizer.optimization_config.get("performance_weight", 0.4)
                        adjustment = pattern.impact_score * 0.1
                        optimization_adjustments["weight_adjustments"][f"{strategy}_performance"] = adjustment
                
                elif pattern.pattern_type == "failure_pattern":
                    # 降低失败策略的权重
                    strategy = pattern.conditions.get("strategy_name")
                    if strategy:
                        failure_rate = pattern.conditions.get("failure_rate", 0)
                        penalty = failure_rate * 0.2
                        optimization_adjustments["weight_adjustments"][f"{strategy}_penalty"] = -penalty
                
                elif pattern.pattern_type == "optimization_opportunity":
                    # 调整优化参数
                    if "processing_time" in pattern.pattern_description:
                        # 时间优化：提高速度权重
                        optimization_adjustments["threshold_adjustments"]["time_weight_increase"] = 0.05
                
                elif pattern.pattern_type == "content_type_pattern":
                    # 内容类型特定优化
                    content_type = pattern.conditions.get("content_type")
                    best_strategy = pattern.conditions.get("best_strategy")
                    if content_type and best_strategy:
                        rule = f"content_type:{content_type} -> prefer_strategy:{best_strategy}"
                        optimization_adjustments["new_rules"].append(rule)
            
            # 应用优化调整
            await self._apply_optimization_adjustments(optimization_adjustments)
            
            return optimization_adjustments
            
        except Exception as e:
            logger.error(f"策略优化执行失败: {e}")
            return optimization_adjustments
    
    async def _apply_optimization_adjustments(self, adjustments: Dict[str, Any]):
        """应用优化调整"""
        try:
            # 应用权重调整
            weight_adjustments = adjustments.get("weight_adjustments", {})
            if weight_adjustments:
                self.strategy_optimizer.adapt_strategy_weights({"weight_changes": weight_adjustments})
            
            # 应用阈值调整
            threshold_adjustments = adjustments.get("threshold_adjustments", {})
            for adjustment_type, value in threshold_adjustments.items():
                if adjustment_type == "time_weight_increase":
                    current_weight = self.strategy_optimizer.optimization_config.get("processing_time", 0.15)
                    new_weight = min(current_weight + value, 0.3)  # 限制最大权重
                    self.strategy_optimizer.optimization_config["processing_time"] = new_weight
            
            logger.info(f"应用了 {len(adjustments)} 项优化调整")
            
        except Exception as e:
            logger.error(f"应用优化调整失败: {e}")
    
    def _collect_recent_executions(self) -> List[Dict[str, Any]]:
        """收集最近的执行数据"""
        try:
            # 获取最近24小时的执行记录
            recent_limit = self.learning_config["pattern_detection_window"]
            executions = self.history_db.get_execution_history(limit=recent_limit)
            
            # 过滤最近24小时内的记录
            current_time = time.time()
            time_threshold = current_time - (24 * 3600)  # 24小时前
            
            recent_executions = [
                ex for ex in executions 
                if ex.get("timestamp", 0) > time_threshold
            ]
            
            logger.info(f"收集到 {len(recent_executions)} 条最近执行记录")
            return recent_executions
            
        except Exception as e:
            logger.error(f"收集最近执行数据失败: {e}")
            return []
    
    def _should_perform_learning(self) -> bool:
        """判断是否应该执行学习"""
        current_time = time.time()
        
        # 检查时间间隔
        last_learning = self.learning_state["last_learning_time"]
        interval_hours = self.learning_config["learning_interval_hours"]
        
        if current_time - last_learning < (interval_hours * 3600):
            return False
        
        # 检查数据量
        recent_executions = self.history_db.get_execution_history(limit=50)
        if len(recent_executions) < self.learning_config["min_executions_for_learning"]:
            return False
        
        return True
    
    def _update_performance_baselines(self, executions: List[Dict[str, Any]]):
        """更新性能基线"""
        try:
            # 按策略分组计算基线
            strategy_metrics = defaultdict(lambda: {"quality": [], "time": [], "success": []})
            
            for ex in executions:
                strategy = ex.get("strategy_name", "unknown")
                if ex.get("success", False):
                    strategy_metrics[strategy]["quality"].append(ex.get("quality_score", 0))
                    strategy_metrics[strategy]["time"].append(ex.get("processing_time", 0))
                    strategy_metrics[strategy]["success"].append(1)
                else:
                    strategy_metrics[strategy]["success"].append(0)
            
            # 计算新的基线
            for strategy, metrics in strategy_metrics.items():
                if len(metrics["quality"]) >= 3:  # 至少3个样本
                    baseline = {
                        "avg_quality": np.mean(metrics["quality"]),
                        "avg_time": np.mean(metrics["time"]) if metrics["time"] else 0,
                        "success_rate": np.mean(metrics["success"]),
                        "last_updated": time.time(),
                        "sample_count": len(metrics["quality"])
                    }
                    self.performance_baselines[strategy] = baseline
            
            logger.debug(f"更新了 {len(self.performance_baselines)} 个策略的性能基线")
            
        except Exception as e:
            logger.error(f"更新性能基线失败: {e}")
    
    def _evaluate_performance_improvements(self, recent_executions: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估性能改进"""
        improvements = {}
        
        try:
            # 与历史基线比较
            current_metrics = defaultdict(lambda: {"quality": [], "time": [], "success": []})
            
            for ex in recent_executions:
                strategy = ex.get("strategy_name", "unknown")
                if ex.get("success", False):
                    current_metrics[strategy]["quality"].append(ex.get("quality_score", 0))
                    current_metrics[strategy]["time"].append(ex.get("processing_time", 0))
                    current_metrics[strategy]["success"].append(1)
                else:
                    current_metrics[strategy]["success"].append(0)
            
            # 计算改进幅度
            for strategy, metrics in current_metrics.items():
                if strategy in self.performance_baselines and len(metrics["quality"]) >= 3:
                    baseline = self.performance_baselines[strategy]
                    
                    current_quality = np.mean(metrics["quality"])
                    current_success = np.mean(metrics["success"])
                    
                    quality_improvement = current_quality - baseline["avg_quality"]
                    success_improvement = current_success - baseline["success_rate"]
                    
                    overall_improvement = (quality_improvement + success_improvement) / 2
                    improvements[strategy] = overall_improvement
            
            return improvements
            
        except Exception as e:
            logger.error(f"评估性能改进失败: {e}")
            return improvements
    
    def _generate_learning_recommendations(self, patterns: List[PerformancePattern],
                                         adjustments: Dict[str, Any],
                                         improvements: Dict[str, float]) -> List[str]:
        """生成学习建议"""
        recommendations = []
        
        # 基于模式的建议
        high_impact_patterns = [p for p in patterns if p.impact_score > 0.5]
        if high_impact_patterns:
            recommendations.append(f"发现 {len(high_impact_patterns)} 个高影响模式，建议重点关注")
        
        # 基于改进的建议
        improved_strategies = [s for s, imp in improvements.items() if imp > 0.1]
        if improved_strategies:
            recommendations.append(f"策略 {', '.join(improved_strategies)} 性能显著提升，可增加使用频率")
        
        degraded_strategies = [s for s, imp in improvements.items() if imp < -0.1]
        if degraded_strategies:
            recommendations.append(f"策略 {', '.join(degraded_strategies)} 性能下降，需要调查原因")
        
        # 基于调整的建议
        if adjustments.get("weight_adjustments"):
            recommendations.append("已调整策略权重，建议观察后续表现")
        
        if adjustments.get("new_rules"):
            recommendations.append(f"新增 {len(adjustments['new_rules'])} 条优化规则")
        
        return recommendations
    
    def _update_learning_state(self, patterns: List[PerformancePattern],
                             adjustments: Dict[str, Any]):
        """更新学习状态"""
        self.learning_state["last_learning_time"] = time.time()
        self.learning_state["total_learning_cycles"] += 1
        self.learning_state["identified_patterns"] = [p.__dict__ for p in patterns]
        self.learning_state["adaptation_history"].append({
            "timestamp": time.time(),
            "adjustments": adjustments,
            "pattern_count": len(patterns)
        })
    
    def _create_skip_report(self, reason: str) -> LearningReport:
        """创建跳过报告"""
        return LearningReport(
            timestamp=time.time(),
            learning_period="skipped",
            processed_executions=0,
            identified_patterns=[],
            optimization_adjustments={},
            performance_improvements={},
            recommendations=[f"跳过学习周期: {reason}"],
            next_learning_schedule=time.time() + (self.learning_config["learning_interval_hours"] * 3600)
        )
    
    def _create_error_report(self, error_message: str) -> LearningReport:
        """创建错误报告"""
        return LearningReport(
            timestamp=time.time(),
            learning_period="error",
            processed_executions=0,
            identified_patterns=[],
            optimization_adjustments={},
            performance_improvements={},
            recommendations=[f"学习周期出错: {error_message}"],
            next_learning_schedule=time.time() + (self.learning_config["learning_interval_hours"] * 3600)
        )
    
    def get_learning_status(self) -> Dict[str, Any]:
        """获取学习状态"""
        return {
            "learning_state": self.learning_state.copy(),
            "performance_baselines": self.performance_baselines.copy(),
            "recent_learning_history": list(self.learning_history)[-5:],
            "next_learning_time": self.learning_state["last_learning_time"] + (self.learning_config["learning_interval_hours"] * 3600),
            "learning_config": self.learning_config.copy()
        }
    
    async def force_learning_cycle(self) -> LearningReport:
        """强制执行学习周期（用于测试和调试）"""
        logger.info("强制执行学习周期")
        # 临时重置时间检查
        original_last_time = self.learning_state["last_learning_time"]
        self.learning_state["last_learning_time"] = 0
        
        try:
            report = await self.continuous_learning_cycle()
            return report
        finally:
            # 如果强制学习失败，恢复原始时间
            if not hasattr(report, 'timestamp'):
                self.learning_state["last_learning_time"] = original_last_time