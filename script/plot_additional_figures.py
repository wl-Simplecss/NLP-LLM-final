#!/usr/bin/env python3
"""
补充可视化脚本：生成额外的报告图表
输出路径：/data/250010066/LLM_course/final_program/report/

图5：RNN vs Transformer vs T5 综合对比
图6：解码策略对比（Greedy vs Beam）
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ============================================================
# 配置：统一风格（与 plot_report_figures.py 保持一致）
# ============================================================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#CCCCCC',
    'axes.grid': True,
    'grid.color': '#E5E5E5',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
})

# 浅色系调色板（与主脚本一致）
COLORS = {
    'blue': '#7EB8DA',       # 淡蓝
    'green': '#98D4A4',      # 淡绿
    'orange': '#FFB87A',     # 淡橙
    'purple': '#C4A8D8',     # 淡紫
    'pink': '#F5A3B5',       # 淡粉
    'yellow': '#F7DC94',     # 淡黄
    'teal': '#7EC8C8',       # 淡青
    'coral': '#F5A896',      # 淡珊瑚
    'gray': '#B8B8B8',       # 淡灰
}

# 深色版本用于边框/强调
COLORS_DARK = {
    'blue': '#4A90B8',
    'green': '#5DAA6B',
    'orange': '#E08040',
    'purple': '#8B6AA8',
    'pink': '#D86080',
    'yellow': '#D4B04A',
    'teal': '#4A9898',
    'coral': '#D87060',
    'gray': '#787878',
}

OUTPUT_DIR = '/data/250010066/LLM_course/final_program/report'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 图5：RNN vs Transformer vs T5 综合对比
# ============================================================
def plot_architecture_comparison():
    """绘制三种架构的综合 BLEU 对比图"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 各架构的最佳配置 BLEU 分数
    # RNN: Teacher Forcing + additive attention (beam)
    # Transformer: Relative PE (greedy)
    # T5: Fine-tuned
    
    models = [
        'RNN\n(Best: TF + Additive)',
        'Transformer\n(Best: Relative PE)',
        'T5\n(Fine-tuned)'
    ]
    
    greedy_scores = [3.98, 4.79, 2.80]  # Greedy BLEU-4 ×100
    beam_scores = [4.36, 4.26, 2.47]    # Beam BLEU-4 ×100
    
    x = np.arange(len(models))
    bar_width = 0.35
    
    # 绘制柱状图
    bars1 = ax.bar(x - bar_width/2, greedy_scores, bar_width, 
                   label='Greedy', color=COLORS['blue'], 
                   edgecolor=COLORS_DARK['blue'], linewidth=1.5)
    bars2 = ax.bar(x + bar_width/2, beam_scores, bar_width, 
                   label='Beam (k=5)', color=COLORS['orange'], 
                   edgecolor=COLORS_DARK['orange'], linewidth=1.5)
    
    # 标注最佳值
    best_idx = np.argmax(greedy_scores)
    bars1[best_idx].set_color(COLORS['green'])
    bars1[best_idx].set_edgecolor(COLORS_DARK['green'])
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 添加水平参考线（最佳分数）
    ax.axhline(y=4.79, color=COLORS_DARK['green'], linestyle='--', 
               alpha=0.7, linewidth=1.5, label='Best (4.79)')
    
    ax.set_ylabel('BLEU-4 (×100)', fontsize=12)
    ax.set_title('Architecture Comparison: RNN vs Transformer vs T5', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 6)
    
    # 添加注释框
    textstr = 'Best Configuration:\nTransformer + Relative PE\nGreedy BLEU-4 = 4.79'
    props = dict(boxstyle='round,pad=0.5', facecolor=COLORS['yellow'], alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_architecture_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: fig5_architecture_comparison.png")


# ============================================================
# 图6：解码策略对比（Greedy vs Beam）
# ============================================================
def plot_decoding_comparison():
    """绘制解码策略对比图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Decoding Strategy Comparison: Greedy vs Beam Search', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    bar_width = 0.35
    
    # --- 子图1：RNN 各配置的解码对比 ---
    ax1 = axes[0]
    
    rnn_configs = ['Dot', 'General', 'Additive', 'TF', 'FR']
    rnn_greedy = [3.65, 3.89, 3.42, 3.98, 0.69]
    rnn_beam = [3.57, 3.29, 4.26, 4.36, 0.95]
    
    x = np.arange(len(rnn_configs))
    bars1 = ax1.bar(x - bar_width/2, rnn_greedy, bar_width, 
                    label='Greedy', color=COLORS['blue'], 
                    edgecolor=COLORS_DARK['blue'], linewidth=1.2)
    bars2 = ax1.bar(x + bar_width/2, rnn_beam, bar_width, 
                    label='Beam (k=5)', color=COLORS['orange'], 
                    edgecolor=COLORS_DARK['orange'], linewidth=1.2)
    
    # 标注 Beam 优于 Greedy 的情况
    for i, (g, b) in enumerate(zip(rnn_greedy, rnn_beam)):
        if b > g:
            bars2[i].set_color(COLORS['green'])
            bars2[i].set_edgecolor(COLORS_DARK['green'])
    
    ax1.set_ylabel('BLEU-4 (×100)')
    ax1.set_title('(a) RNN Model', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(rnn_configs)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 5.5)
    
    # 添加数值标签
    for bar in bars1:
        if bar.get_height() > 0.5:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        if bar.get_height() > 0.5:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    
    # --- 子图2：Transformer 各配置的解码对比 ---
    ax2 = axes[1]
    
    tfm_configs = ['Sinusoidal', 'Learned', 'Relative', 'LayerNorm', 'RMSNorm']
    tfm_greedy = [4.09, 3.64, 4.79, 3.52, 3.25]
    tfm_beam = [3.71, 3.39, 4.26, 2.61, 2.51]
    
    x = np.arange(len(tfm_configs))
    bars1 = ax2.bar(x - bar_width/2, tfm_greedy, bar_width, 
                    label='Greedy', color=COLORS['green'], 
                    edgecolor=COLORS_DARK['green'], linewidth=1.2)
    bars2 = ax2.bar(x + bar_width/2, tfm_beam, bar_width, 
                    label='Beam (k=5)', color=COLORS['purple'], 
                    edgecolor=COLORS_DARK['purple'], linewidth=1.2)
    
    # 标注最佳值
    best_idx = np.argmax(tfm_greedy)
    bars1[best_idx].set_color(COLORS['teal'])
    bars1[best_idx].set_edgecolor(COLORS_DARK['teal'])
    
    ax2.set_ylabel('BLEU-4 (×100)')
    ax2.set_title('(b) Transformer Model', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tfm_configs, rotation=15, ha='right')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 6)
    
    # 添加数值标签
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_decoding_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: fig6_decoding_comparison.png")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("正在生成补充报告图表...")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    plot_architecture_comparison()
    plot_decoding_comparison()
    
    print("=" * 60)
    print("✓ 补充图表生成完成！")
    print("生成的文件:")
    print("  - fig5_architecture_comparison.png (架构综合对比)")
    print("  - fig6_decoding_comparison.png     (解码策略对比)")
    print("=" * 60)


if __name__ == '__main__':
    main()

