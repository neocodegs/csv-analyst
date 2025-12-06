"""
Pandas + LLM 辅助类
结合 Pandas 数据分析和大语言模型，实现自然语言数据查询
"""

import pandas as pd
from openai import OpenAI
from typing import Any, Dict, Optional
import json


class PandasLLMAgent:
    """Pandas 数据分析 LLM 代理"""
    
    def __init__(self, df: pd.DataFrame, api_key: str, api_base: str = None, model: str = "qwen-plus"):
        """
        初始化代理
        
        Args:
            df: Pandas DataFrame
            api_key: API 密钥
            api_base: API Base URL (可选)
            model: 模型名称
        """
        self.df = df
        self.model = model
        
        # 初始化 OpenAI 客户端
        client_args = {"api_key": api_key}
        if api_base:
            client_args["base_url"] = api_base
        
        self.client = OpenAI(**client_args)
        
        # 获取数据框基本信息
        self.df_info = self._get_dataframe_info()
    
    def _get_dataframe_info(self) -> str:
        """获取 DataFrame 的基本信息"""
        info_parts = []
        
        # 列信息
        info_parts.append("列信息:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            non_null = self.df[col].notna().sum()
            sample_values = self.df[col].dropna().head(3).tolist()
            info_parts.append(f"  - {col} ({dtype}): {non_null} non-null, 样例: {sample_values}")
        
        # 数据维度
        info_parts.append(f"\n数据维度: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")
        
        # 前几行数据
        info_parts.append(f"\n前 3 行数据:\n{self.df.head(3).to_string()}")
        
        return "\n".join(info_parts)
    
    def _generate_code(self, question: str) -> str:
        """使用 LLM 生成 Pandas 代码"""
        
        system_prompt = """你是一个 Python 数据分析专家。用户会提供一个数据集的信息和一个问题。
你需要生成 Python 代码来回答这个问题。

规则:
1. 只使用 pandas 库，DataFrame 变量名为 'df'
2. 如果是数据分析/计算，必须将最终结果赋值给变量 'result'
3. 如果需要画图，使用 matplotlib.pyplot (plt) 绘图，不要调用 plt.show()，也不需要设置 'result' 变量
4. 不要使用 print() 语句
5. 代码要简洁高效
6. 只返回纯 Python 代码，不要有任何解释或 markdown 格式

示例:
问题: "平均收入是多少?"
代码: result = df['Revenue'].mean()

问题: "画出收入分布的直方图"
代码: 
import matplotlib.pyplot as plt
plt.figure()
df['Revenue'].hist()
plt.title('Revenue Distribution')
"""
        
        user_prompt = f"""数据集信息:
{self.df_info}

问题: {question}

请生成 Python 代码来回答这个问题。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            code = response.choices[0].message.content.strip()
            
            # 清理代码（移除 markdown 代码块标记）
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            
            return code.strip()
        
        except Exception as e:
            raise Exception(f"生成代码失败: {e}")
    
    def _execute_code(self, code: str) -> Any:
        """安全执行生成的代码"""
        try:
            # 准备执行环境
            local_vars = {"df": self.df.copy(), "pd": pd}
            
            # 执行代码
            exec(code, {"pd": pd}, local_vars)
            
            # 获取结果 (优先检查 result，如果没有则返回 None)
            return local_vars.get("result")
        
        except Exception as e:
            raise Exception(f"执行代码失败: {e}\\n代码:\\n{code}")
    
    def chat(self, question: str, verbose: bool = False) -> Any:
        """
        使用自然语言查询数据
        
        Args:
            question: 自然语言问题
            verbose: 是否显示详细信息（包括生成的代码）
        
        Returns:
            查询结果
        """
        if verbose:
            print(f"\n🤔 分析问题: {question}")
        
        # 生成代码
        code = self._generate_code(question)
        
        if verbose:
            print(f"\n💻 生成的代码:\n{code}\n")
        
        # 执行代码
        result = self._execute_code(code)
        
        if verbose:
            print(f"✅ 执行成功\n")
        
        return result


class PandasMultiTableAgent:
    """多表关联查询代理"""
    
    def __init__(self, dataframes: Dict[str, pd.DataFrame], api_key: str, 
                 api_base: str = None, model: str = "qwen-plus"):
        """
        初始化多表代理
        
        Args:
            dataframes: 数据框字典 {名称: DataFrame}
            api_key: API 密钥
            api_base: API Base URL (可选)
            model: 模型名称
        """
        self.dataframes = dataframes
        self.model = model
        
        # 初始化 OpenAI 客户端
        client_args = {"api_key": api_key}
        if api_base:
            client_args["base_url"] = api_base
        
        self.client = OpenAI(**client_args)
        
        # 获取所有表的信息
        self.tables_info = self._get_tables_info()
    
    def _get_tables_info(self) -> str:
        """获取所有表的信息"""
        info_parts = []
        
        for name, df in self.dataframes.items():
            info_parts.append(f"\n表名: {name}")
            info_parts.append(f"维度: {df.shape[0]} 行 × {df.shape[1]} 列")
            info_parts.append("列:")
            for col in df.columns:
                dtype = df[col].dtype
                sample = df[col].dropna().head(2).tolist()
                info_parts.append(f"  - {col} ({dtype}), 样例: {sample}")
            info_parts.append(f"前 2 行:\n{df.head(2).to_string()}")
        
        return "\n".join(info_parts)
    
    def _generate_code(self, question: str) -> str:
        """使用 LLM 生成多表关联代码"""
        
        table_names = ", ".join([f"'{name}'" for name in self.dataframes.keys()])
        
        system_prompt = f"""你是一个 Python 数据分析专家。用户会提供多个数据表和一个问题。
你需要生成 Python 代码来回答这个问题。

可用的表: {table_names}

规则:
1. 使用 pandas 库，表名就是 DataFrame 变量名
2. 如果是数据分析/计算，必须将最终结果赋值给变量 'result'
3. 如果需要画图，使用 matplotlib.pyplot (plt) 绘图，不要调用 plt.show()，也不需要设置 'result' 变量
4. 不要使用 print() 语句
5. 需要时使用 merge/join 关联表
6. 只返回纯 Python 代码，不要有任何解释

示例:
问题: "谁的工资最高?"
代码:
merged = employees.merge(salaries, on='EmployeeID')
result = merged.loc[merged['Salary'].idxmax(), 'Name']
"""
        
        user_prompt = f"""数据表信息:
{self.tables_info}

问题: {question}

请生成 Python 代码来回答这个问题。"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            code = response.choices[0].message.content.strip()
            
            # 清理代码
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            
            return code.strip()
        
        except Exception as e:
            raise Exception(f"生成代码失败: {e}")
    
    def _execute_code(self, code: str) -> Any:
        """安全执行生成的代码"""
        try:
            # 准备执行环境
            local_vars = {**self.dataframes, "pd": pd}
            
            # 执行代码
            exec(code, {"pd": pd}, local_vars)
            
            # 获取结果 (优先检查 result，如果没有则返回 None)
            return local_vars.get("result")
        
        except Exception as e:
            raise Exception(f"执行代码失败: {e}\\n代码:\\n{code}")
    
    def chat(self, question: str, verbose: bool = False) -> Any:
        """
        使用自然语言查询多表数据
        
        Args:
            question: 自然语言问题
            verbose: 是否显示详细信息
        
        Returns:
            查询结果
        """
        if verbose:
            print(f"\n🤔 分析问题: {question}")
        
        # 生成代码
        code = self._generate_code(question)
        
        if verbose:
            print(f"\n💻 生成的代码:\n{code}\n")
        
        # 执行代码
        result = self._execute_code(code)
        
        if verbose:
            print(f"✅ 执行成功\n")
        
        return result

