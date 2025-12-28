#!/usr/bin/env python3
"""
可视化脚本：生成机器翻译项目报告图表
输出路径：/data/250010066/LLM_course/final_program/report/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ============================================================
# 配置：统一风格（浅色系 + 白底 + 浅灰网格）
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

# 浅色系调色板
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
# 图1：消融实验 BLEU 对比柱状图
# ============================================================
def plot_ablation_comparison():
    """绘制消融实验对比柱状图（分组柱状图：Greedy vs Beam）"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Ablation Study: BLEU-4 Comparison (×100)', fontsize=14, fontweight='bold', y=0.98)
    
    bar_width = 0.35
    
    # --- 子图1：RNN 注意力类型 ---
    ax1 = axes[0, 0]
    labels_attn = ['Dot', 'General', 'Additive']
    greedy_attn = [3.65, 3.89, 3.42]
    beam_attn = [3.57, 3.29, 4.26]
    
    x = np.arange(len(labels_attn))
    bars1 = ax1.bar(x - bar_width/2, greedy_attn, bar_width, label='Greedy', 
                    color=COLORS['blue'], edgecolor=COLORS_DARK['blue'], linewidth=1.2)
    bars2 = ax1.bar(x + bar_width/2, beam_attn, bar_width, label='Beam (k=5)', 
                    color=COLORS['orange'], edgecolor=COLORS_DARK['orange'], linewidth=1.2)
    
    ax1.set_ylabel('BLEU-4 (×100)')
    ax1.set_title('(a) RNN: Attention Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_attn)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 5.5)
    
    # 添加数值标签
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    
    # --- 子图2：RNN 训练策略 ---
    ax2 = axes[0, 1]
    labels_tf = ['Teacher\nForcing', 'Free\nRunning']
    greedy_tf = [3.98, 0.69]
    beam_tf = [4.36, 0.95]
    
    x = np.arange(len(labels_tf))
    bars1 = ax2.bar(x - bar_width/2, greedy_tf, bar_width, label='Greedy', 
                    color=COLORS['blue'], edgecolor=COLORS_DARK['blue'], linewidth=1.2)
    bars2 = ax2.bar(x + bar_width/2, beam_tf, bar_width, label='Beam (k=5)', 
                    color=COLORS['orange'], edgecolor=COLORS_DARK['orange'], linewidth=1.2)
    
    ax2.set_ylabel('BLEU-4 (×100)')
    ax2.set_title('(b) RNN: Training Strategy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_tf)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 5.5)
    
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    
    # --- 子图3：Transformer 位置编码 ---
    ax3 = axes[1, 0]
    labels_pe = ['Sinusoidal', 'Learned', 'Relative']
    greedy_pe = [4.09, 3.64, 4.79]
    beam_pe = [3.71, 3.39, 4.26]
    
    x = np.arange(len(labels_pe))
    bars1 = ax3.bar(x - bar_width/2, greedy_pe, bar_width, label='Greedy', 
                    color=COLORS['green'], edgecolor=COLORS_DARK['green'], linewidth=1.2)
    bars2 = ax3.bar(x + bar_width/2, beam_pe, bar_width, label='Beam (k=5)', 
                    color=COLORS['purple'], edgecolor=COLORS_DARK['purple'], linewidth=1.2)
    
    ax3.set_ylabel('BLEU-4 (×100)')
    ax3.set_title('(c) Transformer: Positional Encoding')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_pe)
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 6)
    
    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    
    # --- 子图4：Transformer 归一化 ---
    ax4 = axes[1, 1]
    labels_norm = ['LayerNorm', 'RMSNorm']
    greedy_norm = [3.52, 3.25]
    beam_norm = [2.61, 2.51]
    
    x = np.arange(len(labels_norm))
    bars1 = ax4.bar(x - bar_width/2, greedy_norm, bar_width, label='Greedy', 
                    color=COLORS['green'], edgecolor=COLORS_DARK['green'], linewidth=1.2)
    bars2 = ax4.bar(x + bar_width/2, beam_norm, bar_width, label='Beam (k=5)', 
                    color=COLORS['purple'], edgecolor=COLORS_DARK['purple'], linewidth=1.2)
    
    ax4.set_ylabel('BLEU-4 (×100)')
    ax4.set_title('(d) Transformer: Normalization')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels_norm)
    ax4.legend(loc='upper right')
    ax4.set_ylim(0, 5)
    
    for bar in bars1:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_bleu_ablation_bars.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: fig1_bleu_ablation_bars.png")


# ============================================================
# 图2：超参数敏感性折线图
# ============================================================
def plot_hyperparameter_sensitivity():
    """绘制超参数敏感性折线图（3个子图）"""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('Hyperparameter Sensitivity Analysis (Transformer)', fontsize=14, fontweight='bold', y=1.02)
    
    marker_size = 8
    line_width = 2
    
    # --- 子图1：Batch Size ---
    ax1 = axes[0]
    batch_sizes = [64, 128, 256]
    greedy_bs = [3.60, 3.17, 3.28]
    beam_bs = [3.01, 3.18, 3.39]
    
    ax1.plot(batch_sizes, greedy_bs, 'o-', color=COLORS_DARK['blue'], 
             markersize=marker_size, linewidth=line_width, label='Greedy', 
             markerfacecolor=COLORS['blue'], markeredgecolor=COLORS_DARK['blue'])
    ax1.plot(batch_sizes, beam_bs, 's--', color=COLORS_DARK['orange'], 
             markersize=marker_size, linewidth=line_width, label='Beam (k=5)',
             markerfacecolor=COLORS['orange'], markeredgecolor=COLORS_DARK['orange'])
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('BLEU-4 (×100)')
    ax1.set_title('(a) Batch Size')
    ax1.set_xticks(batch_sizes)
    ax1.legend(loc='best')
    ax1.set_ylim(2.5, 4.5)
    
    # 添加数值标签
    for i, (x, y) in enumerate(zip(batch_sizes, greedy_bs)):
        ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
    for i, (x, y) in enumerate(zip(batch_sizes, beam_bs)):
        ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
    
    # --- 子图2：Learning Rate ---
    ax2 = axes[1]
    lrs = [0.5, 1.0, 2.0]
    greedy_lr = [4.25, 3.73, 2.58]
    beam_lr = [3.52, 3.32, 2.26]
    
    ax2.plot(lrs, greedy_lr, 'o-', color=COLORS_DARK['green'], 
             markersize=marker_size, linewidth=line_width, label='Greedy',
             markerfacecolor=COLORS['green'], markeredgecolor=COLORS_DARK['green'])
    ax2.plot(lrs, beam_lr, 's--', color=COLORS_DARK['purple'], 
             markersize=marker_size, linewidth=line_width, label='Beam (k=5)',
             markerfacecolor=COLORS['purple'], markeredgecolor=COLORS_DARK['purple'])
    
    ax2.set_xlabel('Learning Rate (scale factor)')
    ax2.set_ylabel('BLEU-4 (×100)')
    ax2.set_title('(b) Learning Rate')
    ax2.set_xticks(lrs)
    ax2.legend(loc='best')
    ax2.set_ylim(1.5, 5)
    
    for i, (x, y) in enumerate(zip(lrs, greedy_lr)):
        ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
    for i, (x, y) in enumerate(zip(lrs, beam_lr)):
        ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
    
    # --- 子图3：Model Scale ---
    ax3 = axes[2]
    scales = ['Small', 'Medium', 'Large']
    scale_x = [0, 1, 2]
    greedy_scale = [2.92, 3.95, 3.41]
    beam_scale = [2.86, 3.36, 3.24]
    
    ax3.plot(scale_x, greedy_scale, 'o-', color=COLORS_DARK['teal'], 
             markersize=marker_size, linewidth=line_width, label='Greedy',
             markerfacecolor=COLORS['teal'], markeredgecolor=COLORS_DARK['teal'])
    ax3.plot(scale_x, beam_scale, 's--', color=COLORS_DARK['coral'], 
             markersize=marker_size, linewidth=line_width, label='Beam (k=5)',
             markerfacecolor=COLORS['coral'], markeredgecolor=COLORS_DARK['coral'])
    
    ax3.set_xlabel('Model Scale')
    ax3.set_ylabel('BLEU-4 (×100)')
    ax3.set_title('(c) Model Scale')
    ax3.set_xticks(scale_x)
    ax3.set_xticklabels(scales)
    ax3.legend(loc='best')
    ax3.set_ylim(2, 4.5)
    
    for i, (x, y) in enumerate(zip(scale_x, greedy_scale)):
        ax3.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
    for i, (x, y) in enumerate(zip(scale_x, beam_scale)):
        ax3.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_hparam_lines.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: fig2_hparam_lines.png")


# ============================================================
# 图3：训练曲线对比（Teacher Forcing vs Free Running）
# ============================================================
def plot_training_curves():
    """绘制训练曲线对比图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Training Curves: Teacher Forcing vs Free Running (RNN)', fontsize=14, fontweight='bold', y=1.02)
    
    # 从日志提取的数据
    # Teacher Forcing (20 epochs)
    epochs_tf = list(range(1, 21))
    train_loss_tf = [5.6182, 4.5439, 4.0441, 3.7175, 3.4803, 3.3023, 3.1638, 3.0547, 
                     2.9630, 2.8902, 2.8271, 2.7737, 2.7263, 2.6875, 2.6546, 2.6231, 
                     2.5960, 2.3857, 2.3155, 2.2804]
    valid_loss_tf = [5.5585, 5.0840, 4.8138, 4.6633, 4.5574, 4.5035, 4.4715, 4.4176, 
                     4.4341, 4.4242, 4.3975, 4.3992, 4.3633, 4.4178, 4.4121, 4.3832, 
                     4.4330, 4.3327, 4.3667, 4.3385]
    
    # Free Running (15 epochs from log)
    epochs_fr = list(range(1, 16))
    train_loss_fr = [6.4904, 5.9561, 5.6942, 5.4921, 5.3251, 5.1823, 5.0650, 4.9653, 
                     4.8801, 4.8096, 4.7461, 4.5170, 4.4298, 4.3814, 4.3416]
    valid_loss_fr = [6.7052, 6.4942, 6.3368, 6.2687, 6.2601, 6.2803, 6.2204, 6.2290, 
                     6.2442, 6.2761, 6.2997, 6.2732, 6.2926, 6.3210, 6.3448]
    
    # --- 子图1：Train Loss ---
    ax1 = axes[0]
    ax1.plot(epochs_tf, train_loss_tf, 'o-', color=COLORS_DARK['blue'], 
             linewidth=2, markersize=5, label='Teacher Forcing',
             markerfacecolor=COLORS['blue'], markeredgecolor=COLORS_DARK['blue'])
    ax1.plot(epochs_fr, train_loss_fr, 's--', color=COLORS_DARK['coral'], 
             linewidth=2, markersize=5, label='Free Running',
             markerfacecolor=COLORS['coral'], markeredgecolor=COLORS_DARK['coral'])
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.set_title('(a) Training Loss')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 21)
    ax1.set_ylim(2, 7)
    
    # --- 子图2：Valid Loss ---
    ax2 = axes[1]
    ax2.plot(epochs_tf, valid_loss_tf, 'o-', color=COLORS_DARK['green'], 
             linewidth=2, markersize=5, label='Teacher Forcing',
             markerfacecolor=COLORS['green'], markeredgecolor=COLORS_DARK['green'])
    ax2.plot(epochs_fr, valid_loss_fr, 's--', color=COLORS_DARK['purple'], 
             linewidth=2, markersize=5, label='Free Running',
             markerfacecolor=COLORS['purple'], markeredgecolor=COLORS_DARK['purple'])
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('(b) Validation Loss')
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 21)
    ax2.set_ylim(4, 7)
    
    # 添加阴影区域标注收敛差异
    ax2.axhline(y=4.33, color=COLORS_DARK['green'], linestyle=':', alpha=0.7, linewidth=1.5)
    ax2.axhline(y=6.22, color=COLORS_DARK['purple'], linestyle=':', alpha=0.7, linewidth=1.5)
    ax2.annotate('Best TF: 4.33', xy=(15, 4.33), xytext=(16, 4.6), fontsize=9, 
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax2.annotate('Best FR: 6.22', xy=(7, 6.22), xytext=(10, 5.8), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: fig3_training_curves.png")


# ============================================================
# 图4：H100 训练效率柱状图
# ============================================================
def plot_h100_efficiency():
    """绘制 H100 训练效率对比图"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 从 BLEU compare 日志提取的推理时间（秒）
    models = [
        'RNN (dot)',
        'RNN (general)',
        'RNN (additive)',
        'RNN (TF)',
        'RNN (FR)',
        'TFM (sinusoidal)',
        'TFM (learned)',
        'TFM (relative)',
        'TFM (LayerNorm)',
        'TFM (RMSNorm)',
        'T5 fine-tuned'
    ]
    
    times = [29.9, 26.3, 29.1, 31.2, 25.6, 79.4, 95.5, 91.2, 60.8, 65.9, 95.4]
    
    # 颜色分组
    colors = [COLORS['blue']] * 5 + [COLORS['green']] * 5 + [COLORS['purple']]
    edge_colors = [COLORS_DARK['blue']] * 5 + [COLORS_DARK['green']] * 5 + [COLORS_DARK['purple']]
    
    y_pos = np.arange(len(models))
    
    bars = ax.barh(y_pos, times, color=colors, edgecolor=edge_colors, linewidth=1.2, height=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Inference Time (seconds) for 200 test samples')
    ax.set_title('H100 GPU: Inference Efficiency Comparison', fontsize=13, fontweight='bold')
    ax.invert_yaxis()  # 最快的在上面
    
    # 添加数值标签
    for i, (bar, time) in enumerate(zip(bars, times)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
               f'{time:.1f}s', va='center', ha='left', fontsize=9)
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['blue'], edgecolor=COLORS_DARK['blue'], label='RNN'),
        mpatches.Patch(facecolor=COLORS['green'], edgecolor=COLORS_DARK['green'], label='Transformer'),
        mpatches.Patch(facecolor=COLORS['purple'], edgecolor=COLORS_DARK['purple'], label='T5')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.set_xlim(0, 110)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_h100_speed_bars.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: fig4_h100_speed_bars.png")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("正在生成机器翻译项目报告图表...")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    plot_ablation_comparison()
    plot_hyperparameter_sensitivity()
    plot_training_curves()
    plot_h100_efficiency()
    
    print("=" * 60)
    print("✓ 所有图表生成完成！")
    print("生成的文件:")
    print("  - fig1_bleu_ablation_bars.png    (消融实验对比)")
    print("  - fig2_hparam_lines.png          (超参数敏感性)")
    print("  - fig3_training_curves.png       (训练曲线对比)")
    print("  - fig4_h100_speed_bars.png       (H100 效率对比)")
    print("=" * 60)


if __name__ == '__main__':
    main()

