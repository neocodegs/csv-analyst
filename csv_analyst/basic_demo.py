"""
Pandas + LLM 基础示例
演示如何使用自然语言查询 Pandas 数据集
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
try:
    from csv_analyst.pandas_llm_helper import PandasLLMAgent
except ImportError:
    from .pandas_llm_helper import PandasLLMAgent

# 加载环境变量
load_dotenv()


def main():
    """运行基础示例"""
    
    # 检查 API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 错误: 请先设置 OPENAI_API_KEY 环境变量")
        print("   在项目根目录的 .env 文件中配置")
        return
    
    print("🚀 初始化 Pandas + LLM Agent...")
    
    # 获取 API Base URL (DashScope 兼容模式)
    api_base = os.getenv("OPENAI_API_BASE")
    
    # 加载数据
    data_dir = Path(__file__).parent / "data"
    companies_file = data_dir / "companies.csv"
    
    print(f"📊 加载数据: {companies_file}")
    df = pd.read_csv(companies_file)
    
    print(f"   数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"\n数据预览:\n{df.head()}\n")
    
    # 创建 LLM Agent
    agent = PandasLLMAgent(
        df=df,
        api_key=api_key,
        api_base=api_base,
        model="qwen-plus"
    )
    
    # 示例查询
    questions = [
        "What is the average revenue by region?",
        "Which country has the highest revenue?",
        "How many employees are there in total?",
        "What is the total revenue for Asia region?",
        "Show me the top 3 countries by number of employees",
    ]
    
    print("=" * 70)
    print("开始自然语言数据查询")
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
