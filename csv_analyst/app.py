"""
Pandas + LLM Streamlit Web 应用
提供图形化界面进行自然语言数据分析
"""

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
try:
    from csv_analyst.pandas_llm_helper import PandasLLMAgent, PandasMultiTableAgent
except ImportError:
    from .pandas_llm_helper import PandasLLMAgent, PandasMultiTableAgent

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="Pandas AI 数据助手",
    page_icon="🐼",
    layout="wide"
)

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}

def reset_chat():
    st.session_state.messages = []

def main():
    # --- 侧边栏配置 ---
    with st.sidebar:
        st.title("🐼 Pandas AI 设置")
        
        # API 配置
        st.subheader("🔑 API 配置")
        api_key = st.text_input(
            "OpenAI API Key", 
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="如果不填，默认使用环境变量中的配置"
        )
        
        api_base = st.text_input(
            "API Base URL",
            value=os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            placeholder="https://api.openai.com/v1"
        )
        
        model_name = st.selectbox(
            "选择模型",
            ["qwen3-coder-plus", "qwen3-max", "qwen-plus", "deepseek-v3.2"],
            index=0
        )
        
        st.markdown("---")
        
        # 文件上传
        st.subheader("📂 数据上传")
        uploaded_files = st.file_uploader(
            "上传 CSV/Excel 文件", 
            type=["csv", "xlsx", "xls", "xlsm"], 
            accept_multiple_files=True,
            on_change=reset_chat
        )
        
        if uploaded_files:
            st.session_state.dataframes = {}
            for file in uploaded_files:
                ext = os.path.splitext(file.name)[1].lower()
                base_name = os.path.splitext(file.name)[0]
                try:
                    if ext == ".csv":
                        df = pd.read_csv(file)
                        st.session_state.dataframes[base_name] = df
                        st.success(f"已加载: {base_name} ({df.shape[0]}行 × {df.shape[1]}列)")
                    else:
                        excel_file = pd.ExcelFile(file)
                        for sheet_name in excel_file.sheet_names:
                            df = excel_file.parse(sheet_name)
                            table_name = f"{base_name}_{sheet_name}"
                            st.session_state.dataframes[table_name] = df
                        st.success(f"已加载: {base_name} 共 {len(excel_file.sheet_names)} 个工作表")
                except Exception as e:
                    st.error(f"读取文件 {file.name} 失败: {e}")
        
        st.markdown("---")
        if st.button("🗑️ 清空聊天记录"):
            reset_chat()
            st.rerun()

    # --- 主界面 ---
    st.title("💬 Pandas AI 数据分析助手")
    st.caption("基于 Python Pandas 和 LLM 的自然语言数据分析工具")

    # 检查配置
    if not api_key:
        st.warning("⚠️ 请在左侧侧边栏配置 API Key")
        st.stop()
    
    if not st.session_state.dataframes:
        st.info("👋 请在左侧上传 CSV/Excel 数据文件开始分析")
        st.stop()

    # 初始化 Agent
    try:
        if len(st.session_state.dataframes) == 1:
            # 单表模式
            name = list(st.session_state.dataframes.keys())[0]
            df = st.session_state.dataframes[name]
            agent = PandasLLMAgent(df, api_key=api_key, api_base=api_base, model=model_name)
            mode_msg = f"当前模式: 单表分析 ({name})"
        else:
            # 多表模式
            agent = PandasMultiTableAgent(
                st.session_state.dataframes, 
                api_key=api_key, 
                api_base=api_base, 
                model=model_name
            )
            mode_msg = f"当前模式: 多表关联分析 ({', '.join(st.session_state.dataframes.keys())})"
            
        # 在右上角显示模式
        st.toast(mode_msg)
            
    except Exception as e:
        st.error(f"初始化 Agent 失败: {e}")
        st.stop()

    # --- 聊天逻辑 ---
    
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 如果有代码，显示代码
            if "code" in msg:
                with st.expander("查看生成的代码"):
                    st.code(msg["code"], language="python")
            # 如果有图表，显示图表 (这里简化处理，历史图表难以持久化，暂不显示历史图表或需特殊处理)
            if "result_type" in msg and msg["result_type"] == "plot":
                st.info("🖼️ (图表已生成，历史记录暂不支持重绘)")

    # 处理用户输入
    if prompt := st.chat_input("输入你的问题，例如：'哪个地区的平均收入最高？'"):
        # 1. 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 生成回答
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # 第一步：生成代码
                with st.status("🤔 正在思考...", expanded=True) as status:
                    st.write("正在分析数据结构...")
                    # 调用 agent 内部方法生成代码 (为了更细粒度的 UI 控制)
                    code = agent._generate_code(prompt)
                    
                    st.write("已生成分析代码:")
                    st.code(code, language="python")
                    
                    st.write("正在执行代码...")
                    
                    # 捕获绘图
                    # 创建一个新的 figure，避免之前的图表干扰
                    plt.close('all')
                    fig = plt.figure(figsize=(10, 6))
                    
                    # 执行代码
                    # 注意：我们需要修改 _execute_code 的上下文以支持 plt.show() 或者自动捕获
                    # 这里我们采用一种简单的策略：在执行后检查 plt.gcf() 是否有内容变化
                    
                    # 为了支持绘图，我们需要在 locals 中注入 plt
                    # 但 agent._execute_code 已经封装好了。
                    # 我们可以简单地通过 agent._execute_code 执行
                    # 但为了更好的图表支持，我们可能需要在 code 前面注入一些 setup 代码
                    
                    # 简单的处理：如果是 PandasLLMAgent，df 在 locals；如果是 Multi，多个 df 在 locals
                    # 我们直接使用 agent 的 _execute_code
                    result = agent._execute_code(code)
                    
                    status.update(label="✅ 分析完成", state="complete", expanded=False)

                # 显示结果
                result_type = "text"
                
                # 检查是否有图表生成 (通过检查当前 figure 是否包含 axes)
                if plt.get_fignums() and plt.gcf().axes:
                    st.pyplot(plt)
                    result_type = "plot"
                    response_text = "已生成图表。"
                    # 清理
                    plt.close('all')
                
                # 显示文本/DataFrame 结果
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                        response_text = "已生成数据表。"
                    elif isinstance(result, pd.Series):
                        st.dataframe(result)
                        response_text = "已生成数据序列。"
                    else:
                        st.markdown(f"### 结果\n{result}")
                        response_text = str(result)
                elif result_type == "plot":
                    pass
                else:
                    st.warning("代码执行完成，但没有返回结果 (result 为 None)。")
                    response_text = "代码执行无返回值。"

                # 保存到历史记录
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "code": code,
                    "result_type": result_type
                })

            except Exception as e:
                st.error(f"❌ 发生错误: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ 发生错误: {e}"
                })

import sys
import subprocess
import os

def run_app():
    """启动 Streamlit 应用的入口函数"""
    # 获取当前文件的绝对路径
    app_path = os.path.abspath(__file__)
    
    # 使用 subprocess 调用 streamlit run
    # sys.executable 确保使用当前的 Python 环境
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    
    # 传递剩余的命令行参数
    cmd.extend(sys.argv[1:])
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

if __name__ == "__main__":
    # 检查是否已经在 Streamlit 环境中运行
    # Streamlit 运行时会设置一些特定的环境变量或状态，但最简单的检查是看是否是主模块运行且没有 streamlit 相关的栈
    # 这里我们保留简单的逻辑：如果是直接运行脚本且不是通过 streamlit 调用的（通常 streamlit 调用会设置特定的 name 或者其他）
    # 但最稳妥的是：如果我们在 run_app 中，我们就是启动器。
    # 如果 __name__ == "__main__"，我们可能是被 python app.py 调用的，也可能是被 streamlit run app.py 调用的
    
    # 实际上，当 streamlit run app.py 运行时，它会执行 app.py，此时 __name__ == "__main__"
    # 但我们需要区分是 "启动器模式" 还是 "应用模式"
    
    # 简单的区分方法：检查环境变量或者 sys.argv
    # 但更好的方法是：让 run_app 只在作为入口点脚本（如 console_script）调用时执行
    # 当被 streamlit run 执行时，它只是执行模块体
    
    # 由于我们把 run_app 定义在同一个文件里，并且 pyproject.toml 指向 pandas_ai.app:run_app
    # 当运行 `uv run pandas-app` 时，它会导入 app 模块并执行 run_app 函数
    pass

# 注意：主要逻辑 main() 应该直接在模块层级调用，当作为脚本执行时
if __name__ == "__main__":
    try:
        # 尝试检查是否在 Streamlit Runtime 中
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx():
            main()
        else:
            # 如果没有上下文，说明是直接 python pandas_ai/app.py 运行的
            run_app()
    except ImportError:
        # 兼容旧版本或无法导入的情况，回退到简单判断
        # 如果是 streamlit run，sys.argv[0] 通常包含 streamlit
        import sys
        if "streamlit" in sys.argv[0]:
            main()
        else:
            run_app()

