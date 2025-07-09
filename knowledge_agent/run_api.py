#!/usr/bin/env python3
"""
API服务器启动器 - 优化版
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
        # 导入API服务器
        import api_server
        print("🚀 启动Knowledge Agent API服务器...")
        api_server.main()
    except ImportError as e:
        print(f"❌ API服务器导入失败: {e}")
        print("请检查依赖是否正确安装")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)