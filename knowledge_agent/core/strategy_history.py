"""
策略历史数据库模块 - 记录和管理策略执行历史
"""
import sqlite3
import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class ExecutionRecord:
    """执行记录数据类"""
    timestamp: float
    session_id: str
    content_type: str
    content_length: int
    content_complexity: str  # simple/medium/complex
    content_language: str
    technical_density: float  # 0.0-1.0
    content_hash: str
    strategy_name: str
    strategy_version: str
    tools_selected: List[str]
    tool_sequence: List[str]
    success: bool
    processing_time: float
    quality_score: float
    concepts_extracted: int
    links_created: int
    ai_calls_count: int
    token_usage: int
    cost_estimate: float
    fallback_triggered: bool
    error_type: Optional[str] = None
    error_stage: Optional[str] = None
    error_details: Optional[str] = None


class StrategyHistoryDB:
    """策略历史数据库管理器"""
    
    def __init__(self, db_path: str = "data/strategy_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup_database()
        logger.info(f"策略历史数据库初始化完成: {self.db_path}")
    
    def _setup_database(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 主执行记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS execution_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    session_id TEXT,
                    
                    -- 输入内容特征
                    content_type TEXT,
                    content_length INTEGER,
                    content_complexity TEXT,
                    content_language TEXT,
                    technical_density REAL,
                    content_hash TEXT,
                    
                    -- 策略信息
                    strategy_name TEXT,
                    strategy_version TEXT,
                    tools_selected TEXT,  -- JSON array
                    tool_sequence TEXT,   -- JSON array
                    
                    -- 执行结果
                    success BOOLEAN,
                    processing_time REAL,
                    quality_score REAL,
                    concepts_extracted INTEGER,
                    links_created INTEGER,
                    
                    -- 性能指标
                    ai_calls_count INTEGER,
                    token_usage INTEGER,
                    cost_estimate REAL,
                    fallback_triggered BOOLEAN,
                    
                    -- 错误信息
                    error_type TEXT,
                    error_stage TEXT,
                    error_details TEXT
                )
            """)
            
            # 策略性能统计表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy_name TEXT,
                    content_type TEXT,
                    time_window TEXT,
                    
                    total_executions INTEGER,
                    success_rate REAL,
                    avg_quality_score REAL,
                    avg_processing_time REAL,
                    avg_cost REAL,
                    
                    performance_trend TEXT,
                    last_updated TIMESTAMP,
                    
                    PRIMARY KEY (strategy_name, content_type, time_window)
                )
            """)
            
            # 内容特征到策略映射表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_strategy_mapping (
                    content_signature TEXT PRIMARY KEY,
                    optimal_strategy TEXT,
                    confidence_score REAL,
                    last_success_time TIMESTAMP,
                    usage_count INTEGER
                )
            """)
            
            # 创建索引优化查询性能
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON execution_records(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_name ON execution_records(strategy_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_type ON execution_records(content_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_success ON execution_records(success)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON execution_records(content_hash)")
            
            conn.commit()
    
    def record_execution(self, record: ExecutionRecord) -> int:
        """记录执行历史"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO execution_records (
                        timestamp, session_id, content_type, content_length,
                        content_complexity, content_language, technical_density,
                        content_hash, strategy_name, strategy_version,
                        tools_selected, tool_sequence, success, processing_time,
                        quality_score, concepts_extracted, links_created,
                        ai_calls_count, token_usage, cost_estimate,
                        fallback_triggered, error_type, error_stage, error_details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.timestamp, record.session_id, record.content_type,
                    record.content_length, record.content_complexity,
                    record.content_language, record.technical_density,
                    record.content_hash, record.strategy_name, record.strategy_version,
                    json.dumps(record.tools_selected), json.dumps(record.tool_sequence),
                    record.success, record.processing_time, record.quality_score,
                    record.concepts_extracted, record.links_created,
                    record.ai_calls_count, record.token_usage, record.cost_estimate,
                    record.fallback_triggered, record.error_type, record.error_stage,
                    record.error_details
                ))
                
                record_id = cursor.lastrowid
                conn.commit()
                
                # 异步更新策略性能统计
                self._update_strategy_performance(record)
                
                return record_id
                
        except Exception as e:
            logger.error(f"记录执行历史失败: {e}")
            return -1
    
    def get_strategy_performance(self, strategy_name: str, 
                               content_type: Optional[str] = None,
                               time_window: str = "30d") -> Dict[str, Any]:
        """获取策略性能数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 计算时间范围
                if time_window.endswith('d'):
                    days = int(time_window[:-1])
                    since_timestamp = time.time() - (days * 24 * 3600)
                elif time_window.endswith('h'):
                    hours = int(time_window[:-1])
                    since_timestamp = time.time() - (hours * 3600)
                else:
                    since_timestamp = 0
                
                # 构建查询条件
                where_conditions = ["strategy_name = ?", "timestamp >= ?"]
                params = [strategy_name, since_timestamp]
                
                if content_type:
                    where_conditions.append("content_type = ?")
                    params.append(content_type)
                
                where_clause = " AND ".join(where_conditions)
                
                # 查询性能数据
                cursor.execute(f"""
                    SELECT 
                        COUNT(*) as total_executions,
                        AVG(CAST(success AS FLOAT)) as success_rate,
                        AVG(quality_score) as avg_quality_score,
                        AVG(processing_time) as avg_processing_time,
                        AVG(cost_estimate) as avg_cost,
                        SUM(CAST(fallback_triggered AS INT)) as fallback_count
                    FROM execution_records 
                    WHERE {where_clause}
                """, params)
                
                result = cursor.fetchone()
                
                if result[0] == 0:  # 没有数据
                    return {
                        "strategy_name": strategy_name,
                        "total_executions": 0,
                        "success_rate": 0.0,
                        "avg_quality_score": 0.0,
                        "avg_processing_time": 0.0,
                        "avg_cost": 0.0,
                        "fallback_rate": 0.0
                    }
                
                return {
                    "strategy_name": strategy_name,
                    "total_executions": result[0],
                    "success_rate": result[1] or 0.0,
                    "avg_quality_score": result[2] or 0.0,
                    "avg_processing_time": result[3] or 0.0,
                    "avg_cost": result[4] or 0.0,
                    "fallback_rate": (result[5] or 0) / result[0]
                }
                
        except Exception as e:
            logger.error(f"获取策略性能数据失败: {e}")
            return {}
    
    def find_similar_executions(self, content_features: Dict[str, Any], 
                              limit: int = 10) -> List[Dict[str, Any]]:
        """查找相似的执行记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 构建相似性查询（简化版本，基于内容特征匹配）
                content_type = content_features.get("content_type", "")
                complexity = content_features.get("complexity", "")
                length_range = content_features.get("length_range", "")
                
                cursor.execute("""
                    SELECT strategy_name, success, quality_score, processing_time,
                           content_complexity, technical_density, fallback_triggered
                    FROM execution_records 
                    WHERE content_type = ? 
                      AND content_complexity = ?
                      AND success = 1
                    ORDER BY quality_score DESC, processing_time ASC
                    LIMIT ?
                """, (content_type, complexity, limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "strategy_name": row[0],
                        "success": bool(row[1]),
                        "quality_score": row[2],
                        "processing_time": row[3],
                        "content_complexity": row[4],
                        "technical_density": row[5],
                        "fallback_triggered": bool(row[6])
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"查找相似执行记录失败: {e}")
            return []
    
    def get_strategy_rankings(self, content_type: Optional[str] = None,
                            time_window: str = "7d") -> List[Dict[str, Any]]:
        """获取策略排名"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 计算时间范围
                if time_window.endswith('d'):
                    days = int(time_window[:-1])
                    since_timestamp = time.time() - (days * 24 * 3600)
                else:
                    since_timestamp = 0
                
                where_conditions = ["timestamp >= ?"]
                params = [since_timestamp]
                
                if content_type:
                    where_conditions.append("content_type = ?")
                    params.append(content_type)
                
                where_clause = " AND ".join(where_conditions)
                
                cursor.execute(f"""
                    SELECT 
                        strategy_name,
                        COUNT(*) as total_executions,
                        AVG(CAST(success AS FLOAT)) as success_rate,
                        AVG(quality_score) as avg_quality_score,
                        AVG(processing_time) as avg_processing_time,
                        AVG(cost_estimate) as avg_cost
                    FROM execution_records 
                    WHERE {where_clause}
                    GROUP BY strategy_name
                    HAVING total_executions >= 3
                    ORDER BY 
                        (success_rate * 0.4 + avg_quality_score * 0.4 + 
                         (1.0 / (avg_processing_time + 1)) * 0.1 + 
                         (1.0 / (avg_cost + 1)) * 0.1) DESC
                """, params)
                
                rankings = []
                for row in cursor.fetchall():
                    # 计算综合评分
                    success_rate = row[2] or 0.0
                    quality_score = row[3] or 0.0
                    processing_time = row[4] or 1.0
                    cost = row[5] or 1.0
                    
                    composite_score = (
                        success_rate * 0.4 +
                        quality_score * 0.4 +
                        (1.0 / (processing_time + 1)) * 0.1 +
                        (1.0 / (cost + 1)) * 0.1
                    )
                    
                    rankings.append({
                        "strategy_name": row[0],
                        "total_executions": row[1],
                        "success_rate": success_rate,
                        "avg_quality_score": quality_score,
                        "avg_processing_time": processing_time,
                        "avg_cost": cost,
                        "composite_score": composite_score
                    })
                
                return rankings
                
        except Exception as e:
            logger.error(f"获取策略排名失败: {e}")
            return []
    
    def _update_strategy_performance(self, record: ExecutionRecord):
        """更新策略性能统计"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 更新每日、每周、每月统计
                for window in ["1d", "7d", "30d"]:
                    cursor.execute("""
                        INSERT OR REPLACE INTO strategy_performance (
                            strategy_name, content_type, time_window,
                            total_executions, success_rate, avg_quality_score,
                            avg_processing_time, avg_cost, performance_trend,
                            last_updated
                        ) VALUES (?, ?, ?, 
                            (SELECT COUNT(*) FROM execution_records 
                             WHERE strategy_name = ? AND content_type = ? 
                               AND timestamp >= ?),
                            (SELECT AVG(CAST(success AS FLOAT)) FROM execution_records 
                             WHERE strategy_name = ? AND content_type = ? 
                               AND timestamp >= ?),
                            (SELECT AVG(quality_score) FROM execution_records 
                             WHERE strategy_name = ? AND content_type = ? 
                               AND timestamp >= ?),
                            (SELECT AVG(processing_time) FROM execution_records 
                             WHERE strategy_name = ? AND content_type = ? 
                               AND timestamp >= ?),
                            (SELECT AVG(cost_estimate) FROM execution_records 
                             WHERE strategy_name = ? AND content_type = ? 
                               AND timestamp >= ?),
                            'stable', ?)
                    """, (
                        record.strategy_name, record.content_type, window,
                        record.strategy_name, record.content_type, self._get_window_timestamp(window),
                        record.strategy_name, record.content_type, self._get_window_timestamp(window),
                        record.strategy_name, record.content_type, self._get_window_timestamp(window),
                        record.strategy_name, record.content_type, self._get_window_timestamp(window),
                        record.strategy_name, record.content_type, self._get_window_timestamp(window),
                        time.time()
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"更新策略性能统计失败: {e}")
    
    def _get_window_timestamp(self, window: str) -> float:
        """计算时间窗口的起始时间戳"""
        if window.endswith('d'):
            days = int(window[:-1])
            return time.time() - (days * 24 * 3600)
        elif window.endswith('h'):
            hours = int(window[:-1])
            return time.time() - (hours * 3600)
        else:
            return 0
    
    def get_execution_history(self, limit: int = 100, 
                            strategy_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取执行历史记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if strategy_name:
                    cursor.execute("""
                        SELECT * FROM execution_records 
                        WHERE strategy_name = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (strategy_name, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM execution_records 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                columns = [description[0] for description in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    record_dict = dict(zip(columns, row))
                    # 解析JSON字段
                    if record_dict.get('tools_selected'):
                        record_dict['tools_selected'] = json.loads(record_dict['tools_selected'])
                    if record_dict.get('tool_sequence'):
                        record_dict['tool_sequence'] = json.loads(record_dict['tool_sequence'])
                    
                    results.append(record_dict)
                
                return results
                
        except Exception as e:
            logger.error(f"获取执行历史失败: {e}")
            return []
    
    def calculate_content_signature(self, content_features: Dict[str, Any]) -> str:
        """计算内容特征签名"""
        # 创建内容特征的哈希签名
        feature_string = f"{content_features.get('type', '')}_" \
                        f"{content_features.get('complexity', '')}_" \
                        f"{content_features.get('length_category', '')}_" \
                        f"{content_features.get('technical_density', 0):.1f}"
        
        return hashlib.md5(feature_string.encode()).hexdigest()[:16]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM execution_records")
                total_records = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT strategy_name) FROM execution_records")
                unique_strategies = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(CAST(success AS FLOAT)) FROM execution_records")
                overall_success_rate = cursor.fetchone()[0] or 0.0
                
                cursor.execute("SELECT AVG(quality_score) FROM execution_records WHERE success = 1")
                avg_quality = cursor.fetchone()[0] or 0.0
                
                return {
                    "total_records": total_records,
                    "unique_strategies": unique_strategies,
                    "overall_success_rate": overall_success_rate,
                    "average_quality_score": avg_quality,
                    "database_size_mb": self.db_path.stat().st_size / (1024 * 1024)
                }
                
        except Exception as e:
            logger.error(f"获取数据库统计失败: {e}")
            return {}