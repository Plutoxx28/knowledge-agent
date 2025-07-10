"""
历史数据分析器 - 分析策略性能趋势和模式
"""
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from .strategy_history import StrategyHistoryDB

logger = logging.getLogger(__name__)


class HistoryAnalyzer:
    """历史数据分析器"""
    
    def __init__(self, history_db: Optional[StrategyHistoryDB] = None):
        self.history_db = history_db or StrategyHistoryDB()
        self.min_samples_for_analysis = 5  # 最少需要5个样本才进行分析
        
    def analyze_strategy_performance_trends(self, time_window: str = "30d") -> Dict[str, Any]:
        """分析策略性能趋势"""
        try:
            # 获取所有策略的排名
            rankings = self.history_db.get_strategy_rankings(time_window=time_window)
            
            if not rankings:
                return {
                    "status": "insufficient_data",
                    "message": "没有足够的历史数据进行分析",
                    "trends": {}
                }
            
            trends = {}
            
            for strategy_data in rankings:
                strategy_name = strategy_data["strategy_name"]
                
                # 获取历史趋势
                trend_data = self._calculate_strategy_trend(strategy_name, time_window)
                
                trends[strategy_name] = {
                    "current_performance": {
                        "success_rate": strategy_data["success_rate"],
                        "avg_quality_score": strategy_data["avg_quality_score"],
                        "avg_processing_time": strategy_data["avg_processing_time"],
                        "composite_score": strategy_data["composite_score"]
                    },
                    "trend": trend_data["trend"],
                    "trend_confidence": trend_data["confidence"],
                    "performance_category": self._categorize_performance(strategy_data),
                    "recommendations": self._generate_strategy_recommendations(strategy_name, strategy_data, trend_data)
                }
            
            # 整体分析
            overall_analysis = self._analyze_overall_performance(rankings)
            
            return {
                "status": "success",
                "analysis_timestamp": time.time(),
                "time_window": time_window,
                "strategy_trends": trends,
                "overall_analysis": overall_analysis,
                "top_performers": self._identify_top_performers(rankings),
                "improvement_opportunities": self._identify_improvement_opportunities(rankings)
            }
            
        except Exception as e:
            logger.error(f"分析策略性能趋势失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def identify_failure_patterns(self, time_window: str = "14d") -> Dict[str, Any]:
        """识别失败模式"""
        try:
            # 获取失败的执行记录
            history_records = self.history_db.get_execution_history(limit=500)
            failed_records = [r for r in history_records if not r.get("success", True)]
            
            if len(failed_records) < 3:
                return {
                    "status": "insufficient_failures",
                    "message": "失败案例不足，无法进行模式分析",
                    "failure_count": len(failed_records)
                }
            
            failure_analysis = {
                "total_failures": len(failed_records),
                "failure_by_strategy": Counter(),
                "failure_by_content_type": Counter(),
                "failure_by_complexity": Counter(),
                "failure_by_stage": Counter(),
                "failure_by_error_type": Counter(),
                "temporal_patterns": {},
                "common_error_sequences": []
            }
            
            # 统计各维度的失败分布
            for record in failed_records:
                failure_analysis["failure_by_strategy"][record.get("strategy_name", "unknown")] += 1
                failure_analysis["failure_by_content_type"][record.get("content_type", "unknown")] += 1
                failure_analysis["failure_by_complexity"][record.get("content_complexity", "unknown")] += 1
                failure_analysis["failure_by_stage"][record.get("error_stage", "unknown")] += 1
                failure_analysis["failure_by_error_type"][record.get("error_type", "unknown")] += 1
            
            # 转换为字典格式
            for key in ["failure_by_strategy", "failure_by_content_type", "failure_by_complexity", 
                       "failure_by_stage", "failure_by_error_type"]:
                failure_analysis[key] = dict(failure_analysis[key])
            
            # 时间模式分析
            failure_analysis["temporal_patterns"] = self._analyze_temporal_failure_patterns(failed_records)
            
            # 识别高风险组合
            failure_analysis["high_risk_combinations"] = self._identify_high_risk_combinations(failed_records)
            
            # 生成改进建议
            failure_analysis["improvement_suggestions"] = self._generate_failure_improvement_suggestions(failure_analysis)
            
            return {
                "status": "success",
                "analysis_timestamp": time.time(),
                "time_window": time_window,
                "failure_analysis": failure_analysis
            }
            
        except Exception as e:
            logger.error(f"识别失败模式失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_strategy_recommendations(self, content_features: Dict[str, Any]) -> Dict[str, Any]:
        """基于内容特征获取策略推荐"""
        try:
            # 查找相似的历史案例
            similar_cases = self.history_db.find_similar_executions(content_features, limit=20)
            
            if len(similar_cases) < 3:
                return {
                    "status": "insufficient_data",
                    "message": "没有足够的相似案例进行推荐",
                    "fallback_strategy": "标准处理流程"
                }
            
            # 分析成功策略
            strategy_performance = defaultdict(lambda: {
                "success_count": 0,
                "total_count": 0,
                "avg_quality": 0.0,
                "avg_time": 0.0,
                "quality_scores": []
            })
            
            for case in similar_cases:
                strategy = case["strategy_name"]
                performance = strategy_performance[strategy]
                
                performance["total_count"] += 1
                if case["success"]:
                    performance["success_count"] += 1
                
                performance["quality_scores"].append(case["quality_score"])
                performance["avg_time"] += case["processing_time"]
            
            # 计算策略评分
            strategy_scores = {}
            for strategy, perf in strategy_performance.items():
                if perf["total_count"] >= 2:  # 至少2个样本
                    success_rate = perf["success_count"] / perf["total_count"]
                    avg_quality = np.mean(perf["quality_scores"])
                    avg_time = perf["avg_time"] / perf["total_count"]
                    
                    # 综合评分
                    composite_score = (
                        success_rate * 0.5 +
                        avg_quality * 0.3 +
                        (1.0 / (avg_time + 1)) * 0.2
                    )
                    
                    strategy_scores[strategy] = {
                        "success_rate": success_rate,
                        "avg_quality": avg_quality,
                        "avg_time": avg_time,
                        "composite_score": composite_score,
                        "sample_count": perf["total_count"]
                    }
            
            # 排序推荐
            sorted_strategies = sorted(strategy_scores.items(), 
                                     key=lambda x: x[1]["composite_score"], 
                                     reverse=True)
            
            if not sorted_strategies:
                return {
                    "status": "no_suitable_strategy",
                    "message": "没有找到合适的策略推荐",
                    "fallback_strategy": "标准处理流程"
                }
            
            # 生成推荐结果
            top_strategy = sorted_strategies[0]
            recommendations = []
            
            for strategy, scores in sorted_strategies[:3]:  # 取前3个
                recommendations.append({
                    "strategy_name": strategy,
                    "confidence": min(scores["composite_score"], 1.0),
                    "expected_success_rate": scores["success_rate"],
                    "expected_quality": scores["avg_quality"],
                    "expected_time": scores["avg_time"],
                    "sample_count": scores["sample_count"],
                    "reasoning": self._generate_strategy_reasoning(strategy, scores, content_features)
                })
            
            return {
                "status": "success",
                "primary_recommendation": top_strategy[0],
                "confidence": top_strategy[1]["composite_score"],
                "all_recommendations": recommendations,
                "similar_cases_count": len(similar_cases)
            }
            
        except Exception as e:
            logger.error(f"获取策略推荐失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def analyze_content_type_performance(self, content_type: str = None) -> Dict[str, Any]:
        """分析特定内容类型的处理性能"""
        try:
            rankings = self.history_db.get_strategy_rankings(content_type=content_type)
            
            if not rankings:
                return {
                    "status": "no_data",
                    "content_type": content_type,
                    "message": "没有该内容类型的历史数据"
                }
            
            # 分析内容类型的整体表现
            total_executions = sum(r["total_executions"] for r in rankings)
            weighted_success_rate = sum(r["success_rate"] * r["total_executions"] for r in rankings) / total_executions
            weighted_quality = sum(r["avg_quality_score"] * r["total_executions"] for r in rankings) / total_executions
            
            # 识别最优策略
            best_strategy = max(rankings, key=lambda x: x["composite_score"])
            
            # 识别问题策略
            problem_strategies = [r for r in rankings if r["success_rate"] < 0.7 or r["avg_quality_score"] < 0.6]
            
            return {
                "status": "success",
                "content_type": content_type or "all_types",
                "analysis_timestamp": time.time(),
                "overall_metrics": {
                    "total_executions": total_executions,
                    "weighted_success_rate": weighted_success_rate,
                    "weighted_avg_quality": weighted_quality,
                    "strategy_count": len(rankings)
                },
                "best_strategy": {
                    "name": best_strategy["strategy_name"],
                    "success_rate": best_strategy["success_rate"],
                    "avg_quality": best_strategy["avg_quality_score"],
                    "sample_count": best_strategy["total_executions"]
                },
                "strategy_rankings": rankings,
                "problem_strategies": problem_strategies,
                "optimization_suggestions": self._generate_content_type_optimizations(rankings)
            }
            
        except Exception as e:
            logger.error(f"分析内容类型性能失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_strategy_trend(self, strategy_name: str, time_window: str) -> Dict[str, Any]:
        """计算策略的性能趋势"""
        try:
            # 获取历史记录
            history = self.history_db.get_execution_history(limit=200, strategy_name=strategy_name)
            
            if len(history) < 5:
                return {"trend": "insufficient_data", "confidence": 0.0}
            
            # 按时间排序
            history.sort(key=lambda x: x["timestamp"])
            
            # 分析最近的趋势
            recent_records = history[-10:]  # 最近10条记录
            older_records = history[-20:-10] if len(history) >= 20 else history[:-10]
            
            if not older_records:
                return {"trend": "insufficient_history", "confidence": 0.0}
            
            # 计算趋势指标
            recent_success_rate = np.mean([r["success"] for r in recent_records])
            older_success_rate = np.mean([r["success"] for r in older_records])
            
            recent_quality = np.mean([r["quality_score"] for r in recent_records])
            older_quality = np.mean([r["quality_score"] for r in older_records])
            
            # 判断趋势
            success_trend = recent_success_rate - older_success_rate
            quality_trend = recent_quality - older_quality
            
            if success_trend > 0.1 and quality_trend > 0.1:
                trend = "improving"
                confidence = min(abs(success_trend) + abs(quality_trend), 1.0)
            elif success_trend < -0.1 or quality_trend < -0.1:
                trend = "declining"
                confidence = min(abs(success_trend) + abs(quality_trend), 1.0)
            else:
                trend = "stable"
                confidence = 0.7
            
            return {
                "trend": trend,
                "confidence": confidence,
                "success_rate_change": success_trend,
                "quality_change": quality_trend,
                "sample_count": len(history)
            }
            
        except Exception as e:
            logger.error(f"计算策略趋势失败: {e}")
            return {"trend": "error", "confidence": 0.0}
    
    def _categorize_performance(self, strategy_data: Dict[str, Any]) -> str:
        """对策略性能进行分类"""
        success_rate = strategy_data["success_rate"]
        quality_score = strategy_data["avg_quality_score"]
        
        if success_rate >= 0.9 and quality_score >= 0.8:
            return "excellent"
        elif success_rate >= 0.8 and quality_score >= 0.7:
            return "good"
        elif success_rate >= 0.7 and quality_score >= 0.6:
            return "acceptable"
        elif success_rate >= 0.5:
            return "poor"
        else:
            return "failing"
    
    def _generate_strategy_recommendations(self, strategy_name: str, 
                                         performance_data: Dict[str, Any], 
                                         trend_data: Dict[str, Any]) -> List[str]:
        """生成策略改进建议"""
        recommendations = []
        
        # 基于性能分类的建议
        category = self._categorize_performance(performance_data)
        
        if category == "failing":
            recommendations.append(f"策略 {strategy_name} 成功率过低，建议暂停使用并调查原因")
        elif category == "poor":
            recommendations.append(f"策略 {strategy_name} 表现不佳，考虑优化工具组合或调整参数")
        
        # 基于趋势的建议
        if trend_data["trend"] == "declining":
            recommendations.append(f"策略 {strategy_name} 性能下降，需要检查最近的变更")
        elif trend_data["trend"] == "improving":
            recommendations.append(f"策略 {strategy_name} 性能提升，可以增加使用频率")
        
        # 基于具体指标的建议
        if performance_data["avg_processing_time"] > 30:
            recommendations.append(f"策略 {strategy_name} 处理时间过长，考虑优化工具选择")
        
        if performance_data["avg_quality_score"] < 0.6:
            recommendations.append(f"策略 {strategy_name} 质量分数偏低，需要改进质量控制")
        
        return recommendations
    
    def _analyze_overall_performance(self, rankings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析整体性能"""
        if not rankings:
            return {}
        
        total_executions = sum(r["total_executions"] for r in rankings)
        avg_success_rate = np.mean([r["success_rate"] for r in rankings])
        avg_quality = np.mean([r["avg_quality_score"] for r in rankings])
        
        return {
            "total_strategies": len(rankings),
            "total_executions": total_executions,
            "avg_success_rate": avg_success_rate,
            "avg_quality_score": avg_quality,
            "performance_variance": np.std([r["composite_score"] for r in rankings]),
            "strategies_above_threshold": len([r for r in rankings if r["success_rate"] > 0.8])
        }
    
    def _identify_top_performers(self, rankings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别表现最好的策略"""
        return sorted(rankings, key=lambda x: x["composite_score"], reverse=True)[:3]
    
    def _identify_improvement_opportunities(self, rankings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别改进机会"""
        opportunities = []
        
        # 识别表现差的策略
        poor_performers = [r for r in rankings if r["success_rate"] < 0.7 or r["avg_quality_score"] < 0.6]
        
        for strategy in poor_performers:
            opportunities.append({
                "type": "underperforming_strategy",
                "strategy_name": strategy["strategy_name"],
                "issue": f"成功率 {strategy['success_rate']:.2f}, 质量 {strategy['avg_quality_score']:.2f}",
                "suggestion": "需要优化或替换"
            })
        
        return opportunities
    
    def _analyze_temporal_failure_patterns(self, failed_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析时间相关的失败模式"""
        # 简化实现，分析失败的时间分布
        failure_hours = [datetime.fromtimestamp(r["timestamp"]).hour for r in failed_records]
        failure_days = [datetime.fromtimestamp(r["timestamp"]).weekday() for r in failed_records]
        
        return {
            "peak_failure_hours": Counter(failure_hours).most_common(3),
            "peak_failure_days": Counter(failure_days).most_common(3)
        }
    
    def _identify_high_risk_combinations(self, failed_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别高风险的组合"""
        combinations = defaultdict(int)
        
        for record in failed_records:
            combo = f"{record.get('content_type', 'unknown')}_{record.get('strategy_name', 'unknown')}"
            combinations[combo] += 1
        
        high_risk = []
        for combo, count in combinations.items():
            if count >= 3:  # 至少3次失败
                content_type, strategy = combo.split('_', 1)
                high_risk.append({
                    "content_type": content_type,
                    "strategy_name": strategy,
                    "failure_count": count
                })
        
        return sorted(high_risk, key=lambda x: x["failure_count"], reverse=True)
    
    def _generate_failure_improvement_suggestions(self, failure_analysis: Dict[str, Any]) -> List[str]:
        """生成失败改进建议"""
        suggestions = []
        
        # 基于失败策略的建议
        if failure_analysis["failure_by_strategy"]:
            worst_strategy = max(failure_analysis["failure_by_strategy"].items(), key=lambda x: x[1])
            suggestions.append(f"策略 '{worst_strategy[0]}' 失败次数最多 ({worst_strategy[1]} 次)，需要重点优化")
        
        # 基于内容类型的建议
        if failure_analysis["failure_by_content_type"]:
            problematic_type = max(failure_analysis["failure_by_content_type"].items(), key=lambda x: x[1])
            suggestions.append(f"内容类型 '{problematic_type[0]}' 处理失败率高，需要专门的处理策略")
        
        return suggestions
    
    def _generate_strategy_reasoning(self, strategy_name: str, 
                                   scores: Dict[str, Any], 
                                   content_features: Dict[str, Any]) -> str:
        """生成策略推荐的理由"""
        reasoning_parts = []
        
        if scores["success_rate"] > 0.8:
            reasoning_parts.append(f"高成功率 ({scores['success_rate']:.1%})")
        
        if scores["avg_quality"] > 0.7:
            reasoning_parts.append(f"高质量输出 ({scores['avg_quality']:.2f})")
        
        if scores["avg_time"] < 10:
            reasoning_parts.append("处理速度快")
        
        if scores["sample_count"] >= 5:
            reasoning_parts.append(f"充足的历史数据支持 ({scores['sample_count']} 样本)")
        
        return "基于 " + "、".join(reasoning_parts) if reasoning_parts else "历史数据支持"
    
    def _generate_content_type_optimizations(self, rankings: List[Dict[str, Any]]) -> List[str]:
        """生成内容类型优化建议"""
        suggestions = []
        
        if not rankings:
            return suggestions
        
        # 找出表现差的策略
        poor_strategies = [r for r in rankings if r["success_rate"] < 0.7]
        
        if poor_strategies:
            suggestions.append(f"有 {len(poor_strategies)} 个策略表现不佳，考虑停用或优化")
        
        # 检查是否有明显的最优策略
        best_score = max(r["composite_score"] for r in rankings)
        good_strategies = [r for r in rankings if r["composite_score"] > best_score * 0.9]
        
        if len(good_strategies) == 1:
            suggestions.append(f"策略 '{good_strategies[0]['strategy_name']}' 表现突出，建议优先使用")
        
        return suggestions