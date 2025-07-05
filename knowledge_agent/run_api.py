#!/usr/bin/env python3
"""
API服务器启动器 - 解决模块导入问题
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['PYTHONPATH'] = str(project_root)

# 现在导入并运行API服务器
if __name__ == "__main__":
    try:
        # 尝试导入完整版API服务器
        import api_server
        print("🚀 启动完整版API服务器...")
        api_server.main()
    except ImportError as e:
        print(f"⚠️ 完整版导入失败: {e}")
        print("🔄 启动简化版API服务器...")
        
        # 使用简化版
        import simple_api_server
        simple_api_server.main()