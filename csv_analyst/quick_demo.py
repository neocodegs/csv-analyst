"""
快速演示脚本
展示 Pandas + LLM 的核心功能
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
try:
    from csv_analyst.pandas_llm_helper import PandasLLMAgent, PandasMultiTableAgent
except ImportError:
    from .pandas_llm_helper import PandasLLMAgent, PandasMultiTableAgent

# 加载环境变量
load_dotenv()


def demo_basic_query():
    """演示基础查询"""
    print("\n" + "=" * 70)
    print("🔹 演示 1: 基础数据查询")
    print("=" * 70)
    
    # 加载数据
    data_dir = Path(__file__).parent / "data"
    df = pd.read_csv(data_dir / "companies.csv")
    
    print(f"\n📊 数据预览 ({df.shape[0]} 行):")
    print(df.head(3))
    
    # 创建 Agent
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    
    agent = PandasLLMAgent(df, api_key=api_key, api_base=api_base, model="qwen-plus")
    
    # 查询
    question = "Which region has the highest average revenue?"
    print(f"\n❓ 问题: {question}")
    
    result = agent.chat(question, verbose=True)
    print(f"📊 答案: {result}\n")


def demo_multi_table():
    """演示多表关联"""
    print("\n" + "=" * 70)
    print("🔹 演示 2: 多表关联查询")
    print("=" * 70)
    
    # 加载数据
    data_dir = Path(__file__).parent / "data"
    employees_df = pd.read_csv(data_dir / "employees.csv")
    salaries_df = pd.read_csv(data_dir / "salaries.csv")
    
    print(f"\n📊 员工表: {employees_df.shape[0]} 行")
    print(employees_df.head(3))
    print(f"\n💰 薪资表: {salaries_df.shape[0]} 行")
    print(salaries_df.head(3))
    
    # 创建 Agent
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    
    agent = PandasMultiTableAgent(
        dataframes={"employees": employees_df, "salaries": salaries_df},
        api_key=api_key,
        api_base=api_base,
        model="qwen-plus"
    )
    
    # 查询
    question = "Show me the name and department of the person with the highest salary"
    print(f"\n❓ 问题: {question}")
    
    result = agent.chat(question, verbose=True)
    print(f"📊 答案:\n{result}\n")


def demo_data_analysis():
    """演示数据分析"""
    print("\n" + "=" * 70)
    print("🔹 演示 3: 复杂数据分析")
    print("=" * 70)
    
    # 加载数据
    data_dir = Path(__file__).parent / "data"
    df = pd.read_csv(data_dir / "companies.csv")
    
    # 创建 Agent
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    
    agent = PandasLLMAgent(df, api_key=api_key, api_base=api_base, model="qwen-plus")
    
    # 多个分析任务
    analyses = [
        "Calculate the revenue per employee for each country",
        "Find countries where revenue is above the average",
    ]
    
    for question in analyses:
        print(f"\n❓ 分析: {question}")
        result = agent.chat(question, verbose=True)
        print(f"📊 结果:\n{result}\n")


def main():
    """运行所有演示"""
    
    # 检查 API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误: 请先配置 OPENAI_API_KEY")
        print("   在项目根目录的 .env 文件中设置")
        return
    
    print("🚀 Pandas + LLM 快速演示")
    print("=" * 70)
    print("展示如何使用自然语言进行数据分析\n")
    
    try:
        # 演示 1: 基础查询
        demo_basic_query()
        
        # 演示 2: 多表关联
        demo_multi_table()
        
        # 演示 3: 数据分析
        demo_data_analysis()
        
        print("=" * 70)
        print("✅ 所有演示完成!")
        print("=" * 70)
        
        print("\n💡 提示:")
        print("  • 所有代码都是 LLM 自动生成的")
        print("  • 支持任何 Pandas 能做的分析")
        print("  • 完全透明，可查看生成的代码")
        print("\n📖 详细文档: pandas_ai/README.md\n")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")


if __name__ == "__main__":
    main()

