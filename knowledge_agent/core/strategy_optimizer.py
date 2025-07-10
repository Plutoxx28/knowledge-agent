"""
智能策略选择器 - 基于历史数据的智能策略优化决策引擎
"""
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from .strategy_history import StrategyHistoryDB
from .history_analyzer import HistoryAnalyzer
from .strategy_evaluator import StrategyEvaluator
from .strategy_definitions import StrategyRegistry, ContentType, ProcessingComplexity, StrategyPriority

logger = logging.getLogger(__name__)


@dataclass
class StrategyRecommendation:
    """策略推荐结果"""
    strategy_name: str
    confidence_score: float
    expected_success_rate: float
    expected_quality: float
    expected_time: float
    reasoning: str
    fallback_strategies: List[str]
    risk_assessment: str
    optimization_notes: List[str]


@dataclass
class OptimizationContext:
    """优化上下文"""
    content_features: Dict[str, Any]
    user_preferences: Dict[str, Any]
    system_constraints: Dict[str, Any]
    historical_context: Dict[str, Any]
    time_constraints: Optional[float]
    quality_requirements: Optional[float]


class StrategyOptimizer:
    """智能策略优化器"""
    
    def __init__(self, 
                 history_db: Optional[StrategyHistoryDB] = None,
                 strategy_registry: Optional[StrategyRegistry] = None):
        self.history_db = history_db or StrategyHistoryDB()
        self.history_analyzer = HistoryAnalyzer(self.history_db)
        self.strategy_evaluator = StrategyEvaluator(self.history_db)
        self.strategy_registry = strategy_registry or StrategyRegistry()
        
        # 优化配置
        self.optimization_config = {
            "min_historical_samples": 3,      # 最少历史样本数
            "confidence_threshold": 0.6,      # 置信度阈值
            "performance_weight": 0.4,        # 历史性能权重
            "compatibility_weight": 0.3,      # 兼容性权重
            "trend_weight": 0.2,              # 趋势权重
            "novelty_penalty": 0.1,           # 新策略惩罚权重
            "fallback_chain_length": 3        # 降级链长度
        }
        
        # 学习参数
        self.learning_params = {
            "exploration_rate": 0.1,          # 探索率
            "exploitation_decay": 0.95,       # 利用衰减率
            "adaptation_speed": 0.1,           # 适应速度
            "performance_memory": 0.8          # 性能记忆权重
        }
        
        # 缓存策略评估结果
        self._evaluation_cache = {}
        self._cache_ttl = 300  # 5分钟缓存
    
    def select_optimal_strategy(self, content_features: Dict[str, Any],
                              context: Optional[OptimizationContext] = None) -> StrategyRecommendation:
        """选择最优策略"""
        try:
            logger.info(f"开始策略优化选择，内容特征: {content_features}")
            
            # 创建优化上下文
            if context is None:
                context = self._create_default_context(content_features)
            
            # 第一阶段：候选策略筛选
            candidate_strategies = self._identify_candidate_strategies(content_features, context)
            
            if not candidate_strategies:
                logger.warning("没有找到候选策略，使用默认策略")
                return self._create_fallback_recommendation()
            
            # 第二阶段：历史性能分析
            performance_scores = self._analyze_historical_performance(candidate_strategies, content_features)
            
            # 第三阶段：兼容性评估
            compatibility_scores = self._evaluate_compatibility(candidate_strategies, content_features, context)
            
            # 第四阶段：趋势分析
            trend_scores = self._analyze_performance_trends(candidate_strategies)
            
            # 第五阶段：综合决策
            final_scores = self._calculate_final_scores(
                candidate_strategies, performance_scores, compatibility_scores, trend_scores
            )
            
            # 选择最优策略
            optimal_strategy = self._select_best_strategy(final_scores, context)
            
            # 生成推荐结果
            recommendation = self._generate_recommendation(
                optimal_strategy, final_scores, content_features, context
            )
            
            # 记录决策过程
            self._log_decision_process(recommendation, final_scores)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"策略优化选择失败: {e}")
            return self._create_error_recommendation(str(e))
    
    def optimize_strategy_selection(self, content_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量优化策略选择"""
        try:
            optimization_results = {
                "total_content": len(content_batch),
                "strategy_assignments": {},
                "optimization_summary": {},
                "performance_predictions": {},
                "resource_allocation": {}
            }
            
            # 分析内容批次特征
            batch_features = self._analyze_batch_features(content_batch)
            
            # 为每个内容选择策略
            strategy_assignments = {}
            for i, content_features in enumerate(content_batch):
                recommendation = self.select_optimal_strategy(content_features)
                strategy_assignments[f"content_{i}"] = {
                    "strategy": recommendation.strategy_name,
                    "confidence": recommendation.confidence_score,
                    "content_features": content_features
                }
            
            optimization_results["strategy_assignments"] = strategy_assignments
            
            # 生成优化摘要
            optimization_results["optimization_summary"] = self._generate_batch_optimization_summary(
                strategy_assignments, batch_features
            )
            
            # 预测性能
            optimization_results["performance_predictions"] = self._predict_batch_performance(
                strategy_assignments
            )
            
            # 资源分配建议
            optimization_results["resource_allocation"] = self._suggest_resource_allocation(
                strategy_assignments
            )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"批量策略优化失败: {e}")
            return {"error": str(e)}
    
    def adapt_strategy_weights(self, recent_performance: Dict[str, Any]) -> Dict[str, Any]:
        """基于最近性能自适应调整策略权重"""
        try:
            adaptation_results = {
                "previous_weights": self.optimization_config.copy(),
                "performance_analysis": {},
                "weight_adjustments": {},
                "new_weights": {},
                "adaptation_impact": {}
            }
            
            # 分析最近性能
            performance_analysis = self._analyze_recent_performance(recent_performance)
            adaptation_results["performance_analysis"] = performance_analysis
            
            # 计算权重调整
            weight_adjustments = self._calculate_weight_adjustments(performance_analysis)
            adaptation_results["weight_adjustments"] = weight_adjustments
            
            # 应用调整
            self._apply_weight_adjustments(weight_adjustments)
            adaptation_results["new_weights"] = self.optimization_config.copy()
            
            # 评估调整影响
            impact_assessment = self._assess_adaptation_impact(weight_adjustments)
            adaptation_results["adaptation_impact"] = impact_assessment
            
            logger.info(f"策略权重自适应调整完成: {weight_adjustments}")
            
            return adaptation_results
            
        except Exception as e:
            logger.error(f"策略权重自适应调整失败: {e}")
            return {"error": str(e)}
    
    def _identify_candidate_strategies(self, content_features: Dict[str, Any], 
                                     context: OptimizationContext) -> List[str]:
        """识别候选策略"""
        try:
            # 基于内容特征筛选适用策略
            content_type = ContentType(content_features.get("content_type", "text"))
            complexity = ProcessingComplexity(content_features.get("complexity", "medium"))
            content_length = content_features.get("content_length", 1000)
            
            applicable_strategies = self.strategy_registry.get_applicable_strategies(
                content_type, complexity, content_length
            )
            
            candidate_names = [s.name for s in applicable_strategies]
            
            # 添加用户偏好策略
            preferred_strategies = context.user_preferences.get("preferred_strategies", [])
            for strategy in preferred_strategies:
                if strategy not in candidate_names and strategy in self.strategy_registry.strategies:
                    candidate_names.append(strategy)
            
            # 确保至少有基础策略
            if not candidate_names:
                candidate_names = ["standard_processing", "minimal_processing"]
            
            logger.debug(f"识别到候选策略: {candidate_names}")
            return candidate_names
            
        except Exception as e:
            logger.error(f"识别候选策略失败: {e}")
            return ["standard_processing"]
    
    def _analyze_historical_performance(self, candidate_strategies: List[str],
                                      content_features: Dict[str, Any]) -> Dict[str, float]:
        """分析历史性能"""
        performance_scores = {}
        
        for strategy_name in candidate_strategies:
            try:
                # 获取相似内容的历史性能
                similar_cases = self.history_db.find_similar_executions(content_features, limit=20)
                strategy_cases = [case for case in similar_cases if case["strategy_name"] == strategy_name]
                
                if len(strategy_cases) >= self.optimization_config["min_historical_samples"]:
                    # 计算历史性能分数
                    success_rates = [case["success"] for case in strategy_cases]
                    quality_scores = [case["quality_score"] for case in strategy_cases]
                    processing_times = [case["processing_time"] for case in strategy_cases]
                    
                    avg_success_rate = np.mean(success_rates)
                    avg_quality = np.mean(quality_scores)
                    avg_time = np.mean(processing_times)
                    
                    # 计算综合性能分数
                    performance_score = (
                        avg_success_rate * 0.4 +
                        avg_quality * 0.4 +
                        max(0, 1.0 - avg_time / 60.0) * 0.2  # 时间权重，假设60秒为基准
                    )
                    
                    # 考虑样本数量的置信度调整
                    confidence_factor = min(len(strategy_cases) / 10.0, 1.0)
                    performance_scores[strategy_name] = performance_score * confidence_factor
                    
                else:
                    # 历史数据不足，使用策略定义的期望值
                    strategy_def = self.strategy_registry.get_strategy(strategy_name)
                    if strategy_def:
                        expected_quality = np.mean(strategy_def.expected_quality_range)
                        expected_success = strategy_def.success_rate_threshold
                        performance_scores[strategy_name] = (expected_quality + expected_success) / 2 * 0.5  # 降低权重
                    else:
                        performance_scores[strategy_name] = 0.3  # 默认较低分数
                        
            except Exception as e:
                logger.error(f"分析策略 {strategy_name} 历史性能失败: {e}")
                performance_scores[strategy_name] = 0.3
        
        return performance_scores
    
    def _evaluate_compatibility(self, candidate_strategies: List[str],
                              content_features: Dict[str, Any],
                              context: OptimizationContext) -> Dict[str, float]:
        """评估兼容性"""
        compatibility_scores = {}
        
        for strategy_name in candidate_strategies:
            try:
                strategy_def = self.strategy_registry.get_strategy(strategy_name)
                if not strategy_def:
                    compatibility_scores[strategy_name] = 0.0
                    continue
                
                score = 0.0
                
                # 内容类型兼容性
                content_type = ContentType(content_features.get("content_type", "text"))
                if content_type in strategy_def.applicable_content_types:
                    score += 0.3
                
                # 复杂度兼容性
                complexity = ProcessingComplexity(content_features.get("complexity", "medium"))
                if complexity in strategy_def.complexity_range:
                    score += 0.3
                
                # 优先级匹配
                user_priority = context.user_preferences.get("priority", "balanced")
                if strategy_def.priority.value == user_priority:
                    score += 0.2
                
                # 系统约束兼容性
                if context.time_constraints:
                    expected_time = np.mean(strategy_def.expected_time_range)
                    if expected_time <= context.time_constraints:
                        score += 0.1
                    else:
                        score -= 0.1
                
                if context.quality_requirements:
                    expected_quality = np.mean(strategy_def.expected_quality_range)
                    if expected_quality >= context.quality_requirements:
                        score += 0.1
                    else:
                        score -= 0.1
                
                compatibility_scores[strategy_name] = max(0.0, min(1.0, score))
                
            except Exception as e:
                logger.error(f"评估策略 {strategy_name} 兼容性失败: {e}")
                compatibility_scores[strategy_name] = 0.5
        
        return compatibility_scores
    
    def _analyze_performance_trends(self, candidate_strategies: List[str]) -> Dict[str, float]:
        """分析性能趋势"""
        trend_scores = {}
        
        for strategy_name in candidate_strategies:
            try:
                trend_analysis = self.history_analyzer._calculate_strategy_trend(strategy_name, "14d")
                
                trend = trend_analysis.get("trend", "stable")
                confidence = trend_analysis.get("confidence", 0.0)
                
                if trend == "improving":
                    trend_score = 0.8 + confidence * 0.2
                elif trend == "stable":
                    trend_score = 0.6
                elif trend == "declining":
                    trend_score = 0.4 - confidence * 0.2
                else:  # insufficient_data or error
                    trend_score = 0.5
                
                trend_scores[strategy_name] = max(0.0, min(1.0, trend_score))
                
            except Exception as e:
                logger.error(f"分析策略 {strategy_name} 趋势失败: {e}")
                trend_scores[strategy_name] = 0.5
        
        return trend_scores
    
    def _calculate_final_scores(self, candidate_strategies: List[str],
                              performance_scores: Dict[str, float],
                              compatibility_scores: Dict[str, float],
                              trend_scores: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """计算最终分数"""
        final_scores = {}
        
        for strategy_name in candidate_strategies:
            perf_score = performance_scores.get(strategy_name, 0.0)
            compat_score = compatibility_scores.get(strategy_name, 0.0)
            trend_score = trend_scores.get(strategy_name, 0.5)
            
            # 加权计算最终分数
            weighted_score = (
                perf_score * self.optimization_config["performance_weight"] +
                compat_score * self.optimization_config["compatibility_weight"] +
                trend_score * self.optimization_config["trend_weight"]
            )
            
            # 新策略惩罚（如果历史数据不足）
            if perf_score < 0.4:  # 可能是新策略或数据不足
                weighted_score *= (1 - self.optimization_config["novelty_penalty"])
            
            final_scores[strategy_name] = {
                "final_score": weighted_score,
                "performance_score": perf_score,
                "compatibility_score": compat_score,
                "trend_score": trend_score,
                "components": {
                    "performance_weighted": perf_score * self.optimization_config["performance_weight"],
                    "compatibility_weighted": compat_score * self.optimization_config["compatibility_weight"],
                    "trend_weighted": trend_score * self.optimization_config["trend_weight"]
                }
            }
        
        return final_scores
    
    def _select_best_strategy(self, final_scores: Dict[str, Dict[str, Any]],
                            context: OptimizationContext) -> str:
        """选择最佳策略"""
        if not final_scores:
            return "standard_processing"
        
        # 按最终分数排序
        sorted_strategies = sorted(
            final_scores.items(),
            key=lambda x: x[1]["final_score"],
            reverse=True
        )
        
        best_strategy = sorted_strategies[0][0]
        best_score = sorted_strategies[0][1]["final_score"]
        
        # 检查置信度阈值
        if best_score < self.optimization_config["confidence_threshold"]:
            logger.warning(f"最佳策略 {best_strategy} 分数过低 ({best_score:.3f})，使用标准策略")
            return "standard_processing"
        
        # 探索 vs 利用策略
        if self._should_explore():
            # 偶尔选择次优策略进行探索
            if len(sorted_strategies) > 1:
                second_best = sorted_strategies[1][0]
                second_score = sorted_strategies[1][1]["final_score"]
                if second_score > best_score * 0.8:  # 次优策略不能太差
                    logger.info(f"执行探索策略，选择 {second_best} 而不是 {best_strategy}")
                    return second_best
        
        return best_strategy
    
    def _generate_recommendation(self, strategy_name: str,
                               final_scores: Dict[str, Dict[str, Any]],
                               content_features: Dict[str, Any],
                               context: OptimizationContext) -> StrategyRecommendation:
        """生成策略推荐"""
        try:
            strategy_score = final_scores.get(strategy_name, {})
            
            # 获取策略定义
            strategy_def = self.strategy_registry.get_strategy(strategy_name)
            
            # 预测性能
            expected_success_rate = strategy_score.get("performance_score", 0.7)
            expected_quality = np.mean(strategy_def.expected_quality_range) if strategy_def else 0.7
            expected_time = np.mean(strategy_def.expected_time_range) if strategy_def else 15.0
            
            # 生成推理说明
            reasoning = self._generate_reasoning(strategy_name, strategy_score, content_features)
            
            # 生成降级链
            fallback_strategies = self.strategy_registry.get_fallback_chain(strategy_name)
            fallback_strategies = fallback_strategies[1:self.optimization_config["fallback_chain_length"]+1]
            
            # 风险评估
            risk_assessment = self._assess_risk(strategy_name, strategy_score, context)
            
            # 优化注释
            optimization_notes = self._generate_optimization_notes(strategy_name, strategy_score)
            
            return StrategyRecommendation(
                strategy_name=strategy_name,
                confidence_score=strategy_score.get("final_score", 0.5),
                expected_success_rate=expected_success_rate,
                expected_quality=expected_quality,
                expected_time=expected_time,
                reasoning=reasoning,
                fallback_strategies=fallback_strategies,
                risk_assessment=risk_assessment,
                optimization_notes=optimization_notes
            )
            
        except Exception as e:
            logger.error(f"生成策略推荐失败: {e}")
            return self._create_fallback_recommendation()
    
    def _create_default_context(self, content_features: Dict[str, Any]) -> OptimizationContext:
        """创建默认优化上下文"""
        return OptimizationContext(
            content_features=content_features,
            user_preferences={"priority": "balanced"},
            system_constraints={},
            historical_context={},
            time_constraints=None,
            quality_requirements=None
        )
    
    def _should_explore(self) -> bool:
        """判断是否应该进行探索"""
        exploration_rate = self.learning_params["exploration_rate"]
        return np.random.random() < exploration_rate
    
    def _generate_reasoning(self, strategy_name: str, 
                          strategy_score: Dict[str, Any],
                          content_features: Dict[str, Any]) -> str:
        """生成推理说明"""
        reasoning_parts = []
        
        # 基于分数组成的推理
        perf_score = strategy_score.get("performance_score", 0.0)
        compat_score = strategy_score.get("compatibility_score", 0.0)
        trend_score = strategy_score.get("trend_score", 0.0)
        
        if perf_score > 0.7:
            reasoning_parts.append(f"历史性能优秀 ({perf_score:.2f})")
        elif perf_score > 0.5:
            reasoning_parts.append(f"历史性能良好 ({perf_score:.2f})")
        else:
            reasoning_parts.append("基于策略定义评估")
        
        if compat_score > 0.8:
            reasoning_parts.append("高度兼容当前内容")
        elif compat_score > 0.6:
            reasoning_parts.append("较好兼容当前内容")
        
        if trend_score > 0.7:
            reasoning_parts.append("性能趋势向好")
        elif trend_score < 0.5:
            reasoning_parts.append("性能趋势需要关注")
        
        # 基于内容特征的推理
        content_type = content_features.get("content_type", "")
        if content_type == "conversation":
            reasoning_parts.append("适合对话内容处理")
        elif content_type == "academic":
            reasoning_parts.append("适合学术内容深度分析")
        
        return "、".join(reasoning_parts) if reasoning_parts else "综合评估结果"
    
    def _assess_risk(self, strategy_name: str, 
                    strategy_score: Dict[str, Any],
                    context: OptimizationContext) -> str:
        """评估风险"""
        final_score = strategy_score.get("final_score", 0.5)
        perf_score = strategy_score.get("performance_score", 0.5)
        
        if final_score > 0.8 and perf_score > 0.7:
            return "低风险"
        elif final_score > 0.6:
            return "中等风险"
        elif final_score > 0.4:
            return "较高风险"
        else:
            return "高风险"
    
    def _generate_optimization_notes(self, strategy_name: str,
                                   strategy_score: Dict[str, Any]) -> List[str]:
        """生成优化注释"""
        notes = []
        
        final_score = strategy_score.get("final_score", 0.5)
        
        if final_score < 0.6:
            notes.append("建议监控执行结果，准备降级")
        
        if strategy_score.get("performance_score", 0.5) < 0.5:
            notes.append("历史数据不足，基于策略定义推荐")
        
        if strategy_score.get("trend_score", 0.5) < 0.5:
            notes.append("性能下降趋势，需要关注")
        
        return notes
    
    def _create_fallback_recommendation(self) -> StrategyRecommendation:
        """创建降级推荐"""
        return StrategyRecommendation(
            strategy_name="standard_processing",
            confidence_score=0.5,
            expected_success_rate=0.8,
            expected_quality=0.7,
            expected_time=20.0,
            reasoning="使用标准策略作为安全选择",
            fallback_strategies=["lightweight_processing", "minimal_processing"],
            risk_assessment="低风险",
            optimization_notes=["标准策略，适用于大多数情况"]
        )
    
    def _create_error_recommendation(self, error_message: str) -> StrategyRecommendation:
        """创建错误推荐"""
        return StrategyRecommendation(
            strategy_name="minimal_processing",
            confidence_score=0.3,
            expected_success_rate=0.9,
            expected_quality=0.5,
            expected_time=5.0,
            reasoning=f"优化过程出错，使用最小策略: {error_message}",
            fallback_strategies=[],
            risk_assessment="低风险",
            optimization_notes=["错误恢复模式"]
        )
    
    def _log_decision_process(self, recommendation: StrategyRecommendation,
                            final_scores: Dict[str, Dict[str, Any]]):
        """记录决策过程"""
        logger.info(f"策略优化决策完成:")
        logger.info(f"  选择策略: {recommendation.strategy_name}")
        logger.info(f"  置信度: {recommendation.confidence_score:.3f}")
        logger.info(f"  推理: {recommendation.reasoning}")
        
        # 记录所有候选策略的分数
        for strategy, scores in final_scores.items():
            logger.debug(f"  {strategy}: {scores['final_score']:.3f} "
                        f"(性能:{scores['performance_score']:.2f}, "
                        f"兼容:{scores['compatibility_score']:.2f}, "
                        f"趋势:{scores['trend_score']:.2f})")
    
    # 批量优化相关方法
    def _analyze_batch_features(self, content_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析批次特征"""
        # 实现批次特征分析逻辑
        return {
            "total_items": len(content_batch),
            "content_types": {},
            "complexity_distribution": {},
            "average_length": 0
        }
    
    def _generate_batch_optimization_summary(self, strategy_assignments: Dict[str, Any],
                                           batch_features: Dict[str, Any]) -> Dict[str, Any]:
        """生成批次优化摘要"""
        # 实现批次优化摘要生成逻辑
        return {
            "strategy_distribution": {},
            "expected_performance": {},
            "resource_requirements": {}
        }
    
    def _predict_batch_performance(self, strategy_assignments: Dict[str, Any]) -> Dict[str, Any]:
        """预测批次性能"""
        # 实现批次性能预测逻辑
        return {
            "overall_success_rate": 0.8,
            "average_quality": 0.75,
            "total_processing_time": 0
        }
    
    def _suggest_resource_allocation(self, strategy_assignments: Dict[str, Any]) -> Dict[str, Any]:
        """建议资源分配"""
        # 实现资源分配建议逻辑
        return {
            "high_priority_items": [],
            "parallel_processing_groups": [],
            "sequential_processing_items": []
        }
    
    # 自适应调整相关方法
    def _analyze_recent_performance(self, recent_performance: Dict[str, Any]) -> Dict[str, Any]:
        """分析最近性能"""
        # 实现最近性能分析逻辑
        return {
            "performance_changes": {},
            "trend_analysis": {},
            "optimization_opportunities": []
        }
    
    def _calculate_weight_adjustments(self, performance_analysis: Dict[str, Any]) -> Dict[str, float]:
        """计算权重调整"""
        # 实现权重调整计算逻辑
        return {
            "performance_weight": 0.0,
            "compatibility_weight": 0.0,
            "trend_weight": 0.0
        }
    
    def _apply_weight_adjustments(self, weight_adjustments: Dict[str, float]):
        """应用权重调整"""
        for key, adjustment in weight_adjustments.items():
            if key in self.optimization_config:
                old_value = self.optimization_config[key]
                new_value = max(0.1, min(0.9, old_value + adjustment))
                self.optimization_config[key] = new_value
    
    def _assess_adaptation_impact(self, weight_adjustments: Dict[str, float]) -> Dict[str, Any]:
        """评估自适应调整影响"""
        # 实现调整影响评估逻辑
        return {
            "expected_improvements": [],
            "potential_risks": [],
            "monitoring_recommendations": []
        }