"""
Pandas + LLM 多表关联示例
演示如何使用自然语言关联查询多个数据集
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
try:
    from csv_analyst.pandas_llm_helper import PandasMultiTableAgent
except ImportError:
    from .pandas_llm_helper import PandasMultiTableAgent

# 加载环境变量
load_dotenv()


def main():
    """运行多表关联示例"""
    
    # 检查 API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 错误: 请先设置 OPENAI_API_KEY 环境变量")
        print("   在项目根目录的 .env 文件中配置")
        return
    
    print("🚀 初始化 Pandas 多表 LLM Agent...")
    
    # 获取 API Base URL (DashScope 兼容模式)
    api_base = os.getenv("OPENAI_API_BASE")
    
    # 加载数据
    data_dir = Path(__file__).parent / "data"
    
    print("📊 加载员工数据...")
    employees_df = pd.read_csv(data_dir / "employees.csv")
    print(f"   {employees_df.shape[0]} 行 × {employees_df.shape[1]} 列")
    
    print("📊 加载薪资数据...")
    salaries_df = pd.read_csv(data_dir / "salaries.csv")
    print(f"   {salaries_df.shape[0]} 行 × {salaries_df.shape[1]} 列\n")
    
    # 创建多表 Agent
    agent = PandasMultiTableAgent(
        dataframes={
            "employees": employees_df,
            "salaries": salaries_df
        },
        api_key=api_key,
        api_base=api_base,
        model="qwen-plus"
    )
    
    # 关联查询示例
    questions = [
        "Who gets paid the most?",
        "What is the average salary by department?",
        "Which department has the highest total compensation (salary + bonus)?",
        "List employees in IT department with their salaries",
        "What is the total salary cost for employees in USA?",
        "Show me top 3 employees by total compensation (salary + bonus) with their names and departments",
    ]
    
    print("=" * 70)
    print("开始多表关联查询")
    print("=" * 70)
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ 问题 {i}: {question}")
        print("-" * 70)
        try:
            # verbose=True 会显示生成的代码
            response = agent.chat(question, verbose=True)
            print(f"📊 结果:")
            print(response)
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    print("\n" + "=" * 70)
    print("查询完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
