"""
Pandas + LLM 可视化示例
演示如何结合 Pandas 和 Matplotlib 生成数据可视化
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载环境变量
load_dotenv()


def main():
    """运行可视化示例"""
    
    print("🚀 Pandas 数据可视化示例...")
    
    # 加载数据
    data_dir = Path(__file__).parent / "data"
    companies_file = data_dir / "companies.csv"
    
    print(f"📊 加载数据: {companies_file}")
    df = pd.read_csv(companies_file)
    
    # 创建图表保存目录
    charts_dir = Path(__file__).parent / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    print(f"📈 生成图表，保存到: {charts_dir}\n")
    
    # 设置样式
    sns.set_style("whitegrid")
    
    # 1. 按国家展示收入 - 条形图
    print("1️⃣  生成条形图: 各国收入")
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('Revenue', ascending=True)
    plt.barh(df_sorted['Country'], df_sorted['Revenue'], color='steelblue')
    plt.xlabel('Revenue', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.title('Revenue by Country', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(charts_dir / '1_revenue_by_country.png', dpi=150)
    plt.close()
    print(f"   ✅ 保存: 1_revenue_by_country.png")
    
    # 2. 按地区展示收入占比 - 饼图
    print("2️⃣  生成饼图: 地区收入分布")
    plt.figure(figsize=(10, 8))
    region_revenue = df.groupby('Region')['Revenue'].sum()
    colors = plt.cm.Set3(range(len(region_revenue)))
    plt.pie(region_revenue, labels=region_revenue.index, autopct='%1.1f%%', 
            startangle=90, colors=colors)
    plt.title('Revenue Distribution by Region', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(charts_dir / '2_revenue_pie_chart.png', dpi=150)
    plt.close()
    print(f"   ✅ 保存: 2_revenue_pie_chart.png")
    
    # 3. 员工数量分布 - 直方图
    print("3️⃣  生成直方图: 员工数量分布")
    plt.figure(figsize=(10, 6))
    plt.hist(df['Employees'], bins=8, color='coral', edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Employees', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Employee Counts', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(charts_dir / '3_employees_histogram.png', dpi=150)
    plt.close()
    print(f"   ✅ 保存: 3_employees_histogram.png")
    
    # 4. 收入 vs 员工数 - 散点图
    print("4️⃣  生成散点图: 收入 vs 员工数")
    plt.figure(figsize=(10, 6))
    regions = df['Region'].unique()
    colors_map = dict(zip(regions, plt.cm.tab10(range(len(regions)))))
    
    for region in regions:
        region_data = df[df['Region'] == region]
        plt.scatter(region_data['Employees'], region_data['Revenue'], 
                   label=region, alpha=0.7, s=100, color=colors_map[region])
    
    plt.xlabel('Employees', fontsize=12)
    plt.ylabel('Revenue', fontsize=12)
    plt.title('Revenue vs Employees by Region', fontsize=14, fontweight='bold')
    plt.legend(title='Region')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(charts_dir / '4_revenue_vs_employees.png', dpi=150)
    plt.close()
    print(f"   ✅ 保存: 4_revenue_vs_employees.png")
    
    # 5. 按地区对比收入和员工数 - 分组条形图
    print("5️⃣  生成分组条形图: 地区统计")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 按地区汇总
    region_stats = df.groupby('Region').agg({
        'Revenue': 'sum',
        'Employees': 'sum'
    }).reset_index()
    
    # 收入
    ax1.bar(region_stats['Region'], region_stats['Revenue'], color='steelblue')
    ax1.set_ylabel('Total Revenue', fontsize=12)
    ax1.set_title('Total Revenue by Region', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # 员工数
    ax2.bar(region_stats['Region'], region_stats['Employees'], color='coral')
    ax2.set_ylabel('Total Employees', fontsize=12)
    ax2.set_title('Total Employees by Region', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(charts_dir / '5_region_comparison.png', dpi=150)
    plt.close()
    print(f"   ✅ 保存: 5_region_comparison.png")
    
    # 6. 相关性热力图
    print("6️⃣  生成热力图: 数据相关性")
    plt.figure(figsize=(8, 6))
    numeric_cols = df[['Revenue', 'Employees']]
    correlation = numeric_cols.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(charts_dir / '6_correlation_heatmap.png', dpi=150)
    plt.close()
    print(f"   ✅ 保存: 6_correlation_heatmap.png")
    
    print("\n" + "=" * 70)
    print(f"✅ 所有图表已生成并保存到: {charts_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
