import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# ==========================================
# 1. 自动寻找 CSV 文件
# ==========================================
filename = 'scalability_FINAL_COMPARISON.csv'
possible_paths = [
    filename,
    os.path.join('training_results', filename),
    os.path.join(os.path.dirname(__file__), filename),  # 脚本所在目录
    os.path.join(os.path.dirname(__file__), 'training_results', filename)
]

df = None
loaded_path = ""

for path in possible_paths:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            loaded_path = path
            print(f"✅ 成功加载数据: {path}")
            break
        except Exception as e:
            print(f"⚠️ 尝试加载 {path} 失败: {e}")

if df is None:
    print("\n❌ 错误: 找不到 CSV 文件！")
    print(f"请确保 '{filename}' 位于当前目录或 'training_results' 文件夹中。")
    sys.exit(1)

# ==========================================
# 2. 设置绘图风格 (论文级出版质量)
# ==========================================
sns.set(style="whitegrid", context="paper", font_scale=1.6)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2.5

# ==========================================
# 3. 定义颜色和样式 (突出 Proposed)
# ==========================================
unique_models = sorted(df['model'].unique())

# 确保 Proposed 排在图例的第一个
if 'Proposed (HYBRID)' in unique_models:
    unique_models.remove('Proposed (HYBRID)')
    unique_models.insert(0, 'Proposed (HYBRID)')

# 自定义调色板和标记
# Proposed = 红色 (tab:red)
# Others = 蓝、绿、橙、紫
palette = {}
markers = {}
dashes = {}
sizes = {}

base_colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']
# 仅使用实心标记，避免报错
filled_markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

print(f"发现模型: {unique_models}")

for i, model in enumerate(unique_models):
    if "Proposed" in model:
        # 主角样式
        palette[model] = 'tab:red'  # 红色
        markers[model] = 'o'  # 实心圆
        dashes[model] = (None, None)  # 实线
        sizes[model] = 3.5  # 线条加粗
    else:
        # 配角样式
        color_idx = i % len(base_colors)
        # 如果配角轮到了红色，就跳过，防止混淆
        if base_colors[color_idx] == 'tab:red':
            color_idx = (color_idx + 1) % len(base_colors)

        palette[model] = base_colors[color_idx]
        markers[model] = filled_markers[(i + 1) % len(filled_markers)]  # 错开标记
        dashes[model] = (None, None)  # 实线 (也可以改成 (2, 2) 虚线)
        sizes[model] = 2.5  # 普通粗细

# ==========================================
# 4. 定义要画的指标
# ==========================================
metrics_map = {
    # 你的核心优势指标
    'beam_alignment_ratio': ('Beam Alignment Ratio', 'Alignment Probability'),
    # 你的次优指标
    'v2v_success_rate': ('V2V Success Rate', 'Success Rate'),
    # 你的延迟优势
    'p95_delay_ms': ('P95 Latency', 'Latency (ms)'),
    # 容量指标
    'v2i_sum_capacity_mbps': ('V2I Sum Capacity', 'Capacity (Mbps)')
}

# ==========================================
# 5. 开始绘图
# ==========================================
output_dir = "Paper_Figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"\n🚀 开始绘图... 图片将保存在 '{output_dir}' 文件夹中")

for col, (title, ylabel) in metrics_map.items():
    if col not in df.columns:
        print(f"⚠️ 跳过 {col}: 列名不存在")
        continue

    plt.figure(figsize=(8, 6))

    # 专门处理 Beam Alignment Ratio 的线宽，让它更粗
    line_width = 3.5 if col == 'beam_alignment_ratio' else 2.5

    try:
        # 使用 lineplot 绘制
        ax = sns.lineplot(
            data=df,
            x='vehicle_count',
            y=col,
            hue='model',
            style='model',
            hue_order=unique_models,
            style_order=unique_models,
            palette=palette,
            markers=markers,
            dashes=dashes,
            markersize=10,
            linewidth=line_width
        )

        # 标题和轴标签
        plt.title(title, fontsize=18, fontweight='bold', y=1.03)
        plt.xlabel('Number of Vehicles', fontsize=16)
        plt.ylabel(ylabel, fontsize=16)

        # 优化图例 (放在合适的位置)
        plt.legend(title='', fontsize=13, title_fontsize=13, loc='best', frameon=True, framealpha=0.9)

        # 网格线
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # 保存
        save_path = os.path.join(output_dir, f"Figure_{col}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  --> 已保存: {save_path}")

    except Exception as e:
        print(f"❌ 绘图失败 ({col}): {e}")
    finally:
        plt.close()

print("\n✅ 所有图片绘制完成！快去查看 Paper_Figures 文件夹吧！")