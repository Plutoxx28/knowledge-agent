"""
策略性能评估框架 - 评估和比较不同策略的性能
"""
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from .strategy_history import StrategyHistoryDB
from .history_analyzer import HistoryAnalyzer
from .strategy_definitions import StrategyRegistry, StrategyDefinition, ProcessingComplexity, ContentType

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    success_rate: float
    avg_quality_score: float
    avg_processing_time: float
    avg_cost: float
    reliability_score: float
    efficiency_score: float
    user_satisfaction: float
    total_executions: int
    failure_rate: float
    fallback_rate: float


@dataclass
class StrategyEvaluation:
    """策略评估结果"""
    strategy_name: str
    performance_metrics: PerformanceMetrics
    trend_analysis: Dict[str, Any]
    ranking_score: float
    category: str  # excellent/good/acceptable/poor/failing
    recommendations: List[str]
    confidence_level: float
    last_evaluation: float


class StrategyEvaluator:
    """策略性能评估器"""
    
    def __init__(self, history_db: Optional[StrategyHistoryDB] = None,
                 strategy_registry: Optional[StrategyRegistry] = None):
        self.history_db = history_db or StrategyHistoryDB()
        self.history_analyzer = HistoryAnalyzer(self.history_db)
        self.strategy_registry = strategy_registry or StrategyRegistry()
        
        # 评估权重配置
        self.evaluation_weights = {
            "success_rate": 0.25,      # 成功率权重
            "quality_score": 0.25,     # 质量分数权重
            "processing_time": 0.15,   # 处理时间权重
            "cost_efficiency": 0.10,   # 成本效率权重
            "reliability": 0.15,       # 可靠性权重
            "trend": 0.10             # 趋势权重
        }
        
        # 性能阈值配置
        self.performance_thresholds = {
            "excellent": {"success_rate": 0.9, "quality": 0.85, "efficiency": 0.8},
            "good": {"success_rate": 0.8, "quality": 0.75, "efficiency": 0.7},
            "acceptable": {"success_rate": 0.7, "quality": 0.65, "efficiency": 0.6},
            "poor": {"success_rate": 0.5, "quality": 0.5, "efficiency": 0.4}
        }
    
    def evaluate_strategy(self, strategy_name: str, 
                         time_window: str = "30d",
                         content_type: Optional[str] = None) -> StrategyEvaluation:
        """评估单个策略的性能"""
        try:
            # 获取策略定义
            strategy_def = self.strategy_registry.get_strategy(strategy_name)
            if not strategy_def:
                logger.warning(f"未找到策略定义: {strategy_name}")
                return self._create_empty_evaluation(strategy_name)
            
            # 获取历史性能数据
            performance_data = self.history_db.get_strategy_performance(
                strategy_name, content_type, time_window
            )
            
            if not performance_data or performance_data.get("total_executions", 0) < 3:
                return self._create_insufficient_data_evaluation(strategy_name, strategy_def)
            
            # 计算性能指标
            metrics = self._calculate_performance_metrics(performance_data, strategy_def)
            
            # 趋势分析
            trend_analysis = self.history_analyzer._calculate_strategy_trend(strategy_name, time_window)
            
            # 计算排名分数
            ranking_score = self._calculate_ranking_score(metrics, trend_analysis)
            
            # 性能分类
            category = self._categorize_performance(metrics)
            
            # 生成建议
            recommendations = self._generate_recommendations(strategy_name, metrics, trend_analysis, strategy_def)
            
            # 计算置信度
            confidence_level = self._calculate_confidence_level(performance_data["total_executions"], trend_analysis)
            
            return StrategyEvaluation(
                strategy_name=strategy_name,
                performance_metrics=metrics,
                trend_analysis=trend_analysis,
                ranking_score=ranking_score,
                category=category,
                recommendations=recommendations,
                confidence_level=confidence_level,
                last_evaluation=time.time()
            )
            
        except Exception as e:
            logger.error(f"评估策略 {strategy_name} 失败: {e}")
            return self._create_error_evaluation(strategy_name, str(e))
    
    def evaluate_all_strategies(self, time_window: str = "30d") -> Dict[str, StrategyEvaluation]:
        """评估所有策略的性能"""
        try:
            evaluations = {}
            
            # 获取所有有历史数据的策略
            rankings = self.history_db.get_strategy_rankings(time_window=time_window)
            
            for ranking_data in rankings:
                strategy_name = ranking_data["strategy_name"]
                evaluation = self.evaluate_strategy(strategy_name, time_window)
                evaluations[strategy_name] = evaluation
            
            # 添加没有历史数据的策略
            for strategy_name in self.strategy_registry.strategies.keys():
                if strategy_name not in evaluations:
                    evaluation = self.evaluate_strategy(strategy_name, time_window)
                    evaluations[strategy_name] = evaluation
            
            return evaluations
            
        except Exception as e:
            logger.error(f"评估所有策略失败: {e}")
            return {}
    
    def compare_strategies(self, strategy_names: List[str], 
                         time_window: str = "30d",
                         content_type: Optional[str] = None) -> Dict[str, Any]:
        """比较多个策略的性能"""
        try:
            evaluations = {}
            
            # 评估每个策略
            for strategy_name in strategy_names:
                evaluations[strategy_name] = self.evaluate_strategy(
                    strategy_name, time_window, content_type
                )
            
            # 生成比较报告
            comparison = {
                "evaluations": evaluations,
                "rankings": self._rank_strategies(evaluations),
                "performance_comparison": self._compare_performance_metrics(evaluations),
                "recommendations": self._generate_comparison_recommendations(evaluations),
                "best_strategy": self._identify_best_strategy(evaluations),
                "analysis_summary": self._generate_comparison_summary(evaluations)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"比较策略失败: {e}")
            return {"error": str(e)}
    
    def evaluate_strategy_for_content(self, content_features: Dict[str, Any]) -> Dict[str, Any]:
        """为特定内容特征评估策略适用性"""
        try:
            # 获取内容特征
            content_type = ContentType(content_features.get("content_type", "text"))
            complexity = ProcessingComplexity(content_features.get("complexity", "medium"))
            content_length = content_features.get("content_length", 1000)
            
            # 获取适用的策略
            applicable_strategies = self.strategy_registry.get_applicable_strategies(
                content_type, complexity, content_length
            )
            
            if not applicable_strategies:
                return {
                    "status": "no_applicable_strategies",
                    "message": "没有找到适用的策略",
                    "fallback_recommendation": "standard_processing"
                }
            
            # 评估每个适用策略
            strategy_evaluations = {}
            for strategy in applicable_strategies:
                evaluation = self.evaluate_strategy(strategy.name)
                
                # 计算内容适配分数
                compatibility_score = self._calculate_content_compatibility(strategy, content_features)
                
                strategy_evaluations[strategy.name] = {
                    "evaluation": evaluation,
                    "compatibility_score": compatibility_score,
                    "overall_score": evaluation.ranking_score * 0.7 + compatibility_score * 0.3
                }
            
            # 排序推荐
            sorted_strategies = sorted(
                strategy_evaluations.items(),
                key=lambda x: x[1]["overall_score"],
                reverse=True
            )
            
            return {
                "status": "success",
                "content_features": content_features,
                "applicable_strategies_count": len(applicable_strategies),
                "strategy_evaluations": strategy_evaluations,
                "ranked_recommendations": [
                    {
                        "strategy_name": name,
                        "overall_score": data["overall_score"],
                        "performance_score": data["evaluation"].ranking_score,
                        "compatibility_score": data["compatibility_score"],
                        "confidence": data["evaluation"].confidence_level,
                        "category": data["evaluation"].category
                    }
                    for name, data in sorted_strategies
                ],
                "top_recommendation": sorted_strategies[0][0] if sorted_strategies else None
            }
            
        except Exception as e:
            logger.error(f"为内容评估策略失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_performance_metrics(self, performance_data: Dict[str, Any], 
                                     strategy_def: StrategyDefinition) -> PerformanceMetrics:
        """计算性能指标"""
        success_rate = performance_data.get("success_rate", 0.0)
        avg_quality = performance_data.get("avg_quality_score", 0.0)
        avg_time = performance_data.get("avg_processing_time", 0.0)
        avg_cost = performance_data.get("avg_cost", 0.0)
        total_executions = performance_data.get("total_executions", 0)
        fallback_rate = performance_data.get("fallback_rate", 0.0)
        
        # 计算可靠性分数 (基于成功率和期望成功率的比较)
        expected_success_rate = strategy_def.success_rate_threshold
        reliability_score = min(success_rate / expected_success_rate, 1.0) if expected_success_rate > 0 else 0.0
        
        # 计算效率分数 (基于时间和期望时间的比较)
        expected_time = np.mean(strategy_def.expected_time_range)
        efficiency_score = max(0, min(expected_time / (avg_time + 1), 1.0))
        
        # 用户满意度 (基于质量和期望质量的比较)
        expected_quality = np.mean(strategy_def.expected_quality_range)
        user_satisfaction = min(avg_quality / expected_quality, 1.0) if expected_quality > 0 else 0.0
        
        return PerformanceMetrics(
            success_rate=success_rate,
            avg_quality_score=avg_quality,
            avg_processing_time=avg_time,
            avg_cost=avg_cost,
            reliability_score=reliability_score,
            efficiency_score=efficiency_score,
            user_satisfaction=user_satisfaction,
            total_executions=total_executions,
            failure_rate=1.0 - success_rate,
            fallback_rate=fallback_rate
        )
    
    def _calculate_ranking_score(self, metrics: PerformanceMetrics, 
                               trend_analysis: Dict[str, Any]) -> float:
        """计算排名分数"""
        # 标准化各项指标到0-1范围
        normalized_success = metrics.success_rate
        normalized_quality = metrics.avg_quality_score
        normalized_time = metrics.efficiency_score
        normalized_cost = max(0, 1.0 - metrics.avg_cost / 10.0)  # 假设成本上限为10
        normalized_reliability = metrics.reliability_score
        
        # 趋势调整因子
        trend_factor = 1.0
        if trend_analysis.get("trend") == "improving":
            trend_factor = 1.0 + trend_analysis.get("confidence", 0.0) * 0.1
        elif trend_analysis.get("trend") == "declining":
            trend_factor = 1.0 - trend_analysis.get("confidence", 0.0) * 0.1
        
        # 加权计算总分
        weighted_score = (
            normalized_success * self.evaluation_weights["success_rate"] +
            normalized_quality * self.evaluation_weights["quality_score"] +
            normalized_time * self.evaluation_weights["processing_time"] +
            normalized_cost * self.evaluation_weights["cost_efficiency"] +
            normalized_reliability * self.evaluation_weights["reliability"]
        )
        
        # 应用趋势调整
        final_score = weighted_score * trend_factor * self.evaluation_weights["trend"] + \
                     weighted_score * (1 - self.evaluation_weights["trend"])
        
        return min(max(final_score, 0.0), 1.0)
    
    def _categorize_performance(self, metrics: PerformanceMetrics) -> str:
        """对性能进行分类"""
        success_rate = metrics.success_rate
        quality_score = metrics.avg_quality_score
        efficiency = metrics.efficiency_score
        
        for category, thresholds in self.performance_thresholds.items():
            if (success_rate >= thresholds["success_rate"] and
                quality_score >= thresholds["quality"] and
                efficiency >= thresholds["efficiency"]):
                return category
        
        return "failing"
    
    def _generate_recommendations(self, strategy_name: str, 
                                metrics: PerformanceMetrics,
                                trend_analysis: Dict[str, Any],
                                strategy_def: StrategyDefinition) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于性能指标的建议
        if metrics.success_rate < 0.7:
            recommendations.append(f"成功率偏低 ({metrics.success_rate:.1%})，需要检查失败原因")
        
        if metrics.avg_quality_score < 0.6:
            recommendations.append(f"质量分数偏低 ({metrics.avg_quality_score:.2f})，考虑优化处理流程")
        
        if metrics.efficiency_score < 0.5:
            recommendations.append("处理效率低下，考虑优化工具选择或并行处理")
        
        if metrics.fallback_rate > 0.2:
            recommendations.append(f"降级率过高 ({metrics.fallback_rate:.1%})，需要提高策略稳定性")
        
        # 基于趋势的建议
        if trend_analysis.get("trend") == "declining":
            recommendations.append("性能呈下降趋势，需要立即调查原因")
        elif trend_analysis.get("trend") == "improving":
            recommendations.append("性能持续改善，可以考虑扩大使用范围")
        
        # 基于策略定义的建议
        if metrics.total_executions < 10:
            recommendations.append("执行样本不足，需要更多数据来准确评估")
        
        # 与期望值比较的建议
        expected_quality = np.mean(strategy_def.expected_quality_range)
        if metrics.avg_quality_score < expected_quality * 0.9:
            recommendations.append(f"质量低于预期 (期望: {expected_quality:.2f})，需要调优")
        
        return recommendations
    
    def _calculate_confidence_level(self, total_executions: int, 
                                  trend_analysis: Dict[str, Any]) -> float:
        """计算评估置信度"""
        # 基于样本数量的基础置信度
        sample_confidence = min(total_executions / 20.0, 1.0)  # 20个样本达到满置信度
        
        # 基于趋势稳定性的调整
        trend_confidence = trend_analysis.get("confidence", 0.5)
        
        # 综合置信度
        overall_confidence = sample_confidence * 0.7 + trend_confidence * 0.3
        
        return min(max(overall_confidence, 0.1), 1.0)
    
    def _calculate_content_compatibility(self, strategy: StrategyDefinition, 
                                       content_features: Dict[str, Any]) -> float:
        """计算策略与内容的兼容性分数"""
        score = 0.0
        
        # 内容类型匹配
        content_type = ContentType(content_features.get("content_type", "text"))
        if content_type in strategy.applicable_content_types:
            score += 0.3
        
        # 复杂度匹配
        complexity = ProcessingComplexity(content_features.get("complexity", "medium"))
        if complexity in strategy.complexity_range:
            score += 0.3
        
        # 内容长度适配
        content_length = content_features.get("content_length", 1000)
        if strategy.min_content_length <= content_length <= strategy.max_content_length:
            # 计算在范围内的适配度
            range_size = strategy.max_content_length - strategy.min_content_length
            if range_size > 0:
                optimal_length = (strategy.min_content_length + strategy.max_content_length) / 2
                distance = abs(content_length - optimal_length)
                length_score = max(0, 1.0 - (distance / range_size))
                score += length_score * 0.2
            else:
                score += 0.2
        
        # 优先级匹配
        priority = content_features.get("priority", "balanced")
        if strategy.priority.value == priority:
            score += 0.2
        
        return min(score, 1.0)
    
    def _rank_strategies(self, evaluations: Dict[str, StrategyEvaluation]) -> List[Dict[str, Any]]:
        """对策略进行排名"""
        ranked = []
        
        for name, evaluation in evaluations.items():
            ranked.append({
                "strategy_name": name,
                "ranking_score": evaluation.ranking_score,
                "category": evaluation.category,
                "confidence": evaluation.confidence_level
            })
        
        return sorted(ranked, key=lambda x: x["ranking_score"], reverse=True)
    
    def _compare_performance_metrics(self, evaluations: Dict[str, StrategyEvaluation]) -> Dict[str, Any]:
        """比较性能指标"""
        if not evaluations:
            return {}
        
        metrics_comparison = {
            "success_rate": {},
            "quality_score": {},
            "processing_time": {},
            "efficiency": {}
        }
        
        for name, evaluation in evaluations.items():
            metrics = evaluation.performance_metrics
            metrics_comparison["success_rate"][name] = metrics.success_rate
            metrics_comparison["quality_score"][name] = metrics.avg_quality_score
            metrics_comparison["processing_time"][name] = metrics.avg_processing_time
            metrics_comparison["efficiency"][name] = metrics.efficiency_score
        
        # 计算最佳表现
        for metric_name, values in metrics_comparison.items():
            if values:
                if metric_name == "processing_time":  # 时间越短越好
                    best = min(values.items(), key=lambda x: x[1])
                else:  # 其他指标越高越好
                    best = max(values.items(), key=lambda x: x[1])
                metrics_comparison[f"best_{metric_name}"] = best
        
        return metrics_comparison
    
    def _generate_comparison_recommendations(self, evaluations: Dict[str, StrategyEvaluation]) -> List[str]:
        """生成比较建议"""
        recommendations = []
        
        if not evaluations:
            return recommendations
        
        # 找出最佳和最差策略
        best_strategy = max(evaluations.items(), key=lambda x: x[1].ranking_score)
        worst_strategy = min(evaluations.items(), key=lambda x: x[1].ranking_score)
        
        recommendations.append(f"推荐使用策略: {best_strategy[0]} (得分: {best_strategy[1].ranking_score:.3f})")
        
        if worst_strategy[1].ranking_score < 0.5:
            recommendations.append(f"建议停用策略: {worst_strategy[0]} (得分: {worst_strategy[1].ranking_score:.3f})")
        
        # 分析类别分布
        categories = [eval.category for eval in evaluations.values()]
        excellent_count = categories.count("excellent")
        failing_count = categories.count("failing")
        
        if excellent_count > 0:
            recommendations.append(f"有 {excellent_count} 个优秀策略可供选择")
        
        if failing_count > 0:
            recommendations.append(f"有 {failing_count} 个策略需要立即优化或停用")
        
        return recommendations
    
    def _identify_best_strategy(self, evaluations: Dict[str, StrategyEvaluation]) -> Optional[Dict[str, Any]]:
        """识别最佳策略"""
        if not evaluations:
            return None
        
        best_strategy = max(evaluations.items(), key=lambda x: x[1].ranking_score)
        
        return {
            "strategy_name": best_strategy[0],
            "ranking_score": best_strategy[1].ranking_score,
            "category": best_strategy[1].category,
            "confidence": best_strategy[1].confidence_level,
            "performance_summary": {
                "success_rate": best_strategy[1].performance_metrics.success_rate,
                "quality_score": best_strategy[1].performance_metrics.avg_quality_score,
                "efficiency": best_strategy[1].performance_metrics.efficiency_score
            }
        }
    
    def _generate_comparison_summary(self, evaluations: Dict[str, StrategyEvaluation]) -> Dict[str, Any]:
        """生成比较摘要"""
        if not evaluations:
            return {}
        
        scores = [eval.ranking_score for eval in evaluations.values()]
        categories = [eval.category for eval in evaluations.values()]
        
        return {
            "total_strategies": len(evaluations),
            "average_score": np.mean(scores),
            "score_variance": np.var(scores),
            "category_distribution": {cat: categories.count(cat) for cat in set(categories)},
            "performance_range": {
                "highest_score": max(scores),
                "lowest_score": min(scores),
                "score_spread": max(scores) - min(scores)
            }
        }
    
    def _create_empty_evaluation(self, strategy_name: str) -> StrategyEvaluation:
        """创建空的评估结果"""
        return StrategyEvaluation(
            strategy_name=strategy_name,
            performance_metrics=PerformanceMetrics(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1.0, 0.0
            ),
            trend_analysis={"trend": "no_data", "confidence": 0.0},
            ranking_score=0.0,
            category="no_data",
            recommendations=["策略定义不存在"],
            confidence_level=0.0,
            last_evaluation=time.time()
        )
    
    def _create_insufficient_data_evaluation(self, strategy_name: str, 
                                           strategy_def: StrategyDefinition) -> StrategyEvaluation:
        """创建数据不足的评估结果"""
        return StrategyEvaluation(
            strategy_name=strategy_name,
            performance_metrics=PerformanceMetrics(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1.0, 0.0
            ),
            trend_analysis={"trend": "insufficient_data", "confidence": 0.0},
            ranking_score=0.0,
            category="insufficient_data",
            recommendations=[f"需要更多执行数据来评估策略 {strategy_name}"],
            confidence_level=0.0,
            last_evaluation=time.time()
        )
    
    def _create_error_evaluation(self, strategy_name: str, error_message: str) -> StrategyEvaluation:
        """创建错误的评估结果"""
        return StrategyEvaluation(
            strategy_name=strategy_name,
            performance_metrics=PerformanceMetrics(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1.0, 0.0
            ),
            trend_analysis={"trend": "error", "confidence": 0.0},
            ranking_score=0.0,
            category="error",
            recommendations=[f"评估出错: {error_message}"],
            confidence_level=0.0,
            last_evaluation=time.time()
        )