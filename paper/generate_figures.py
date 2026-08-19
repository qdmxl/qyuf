#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成论文插图（CJK字体修复版）：熵演化 + 概率分布 + 八卦覆盖 + 参数分析

使用 FontProperties(fname=font_path) 方式避免系统字体制问题。
"""
import sys, os, json
sys.path.insert(0, '/home/lijinhan/MXL/科研/ylyw/QYUF/experiment')
from yijing_quantum_experiment import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# CJK 字体路径
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fp = FontProperties(fname=FONT_PATH)
fp_bold = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
fp_medium = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')

FULL_HEX = [
    "乾","坤","屯","蒙","需","讼","师","比","小畜","履","泰","否",
    "同人","大有","谦","豫","随","蛊","临","观","噬嗑","贲","剥","复",
    "无妄","大畜","颐","大过","坎","离",
    "咸","恒","遁","大壮","晋","明夷","家人","睽","蹇","解","损","益",
    "夬","姤","萃","升","困","井","革","鼎","震","艮","渐","归妹",
    "丰","旅","巽","兑","涣","节","中孚","小过","既济","未济"
]

TRIGRAM_NAMES = ["坤","震","坎","兑","艮","离","巽","乾"]
fig_dir = '/home/lijinhan/MXL/科研/ylyw/QYUF/paper/figures'
os.makedirs(fig_dir, exist_ok=True)

# === 三组参数配置 ===
configs = {
    "A-强ZZ":  {'J_adj': 1.5, 'J_ying': 0.3, 'h_dang': 0.20, 'h_zhong': 0.10, 'J_compete': 0.5},
    "B-弱ZZ强应位": {'J_adj': 0.5, 'J_ying': 0.8, 'h_dang': 0.10, 'h_zhong': 0.05, 'J_compete': 1.0},
    "C-平衡": {'J_adj': 1.0, 'J_ying': 0.5, 'h_dang': 0.30, 'h_zhong': 0.15, 'J_compete': 0.5},
}

def run_evolution(params, n_steps=500, record_every=10):
    """运行单次酉演化并返回完整轨迹"""
    H = build_hamiltonian(1.0, params)
    U = unitary_evolution(H, 0.1)
    psi = initialize_state()
    cur = psi.copy()
    n_records = n_steps // record_every + 1
    entropy_series = np.zeros(n_records)
    prob_series = np.zeros((n_records, 64))
    for t in range(n_records):
        if t > 0:
            for _ in range(record_every):
                cur = U @ cur
        p = np.abs(cur) ** 2
        prob_series[t] = p
        entropy_series[t] = compute_entropy(p)
    return entropy_series, prob_series

# === 运行动态参数扫描 ===
def parameter_scan(hd_range, jc_range):
    """扫描 h_dang × J_compete 参数空间"""
    n_hd, n_jc = len(hd_range), len(jc_range)
    entropy_map = np.zeros((n_hd, n_jc))
    kl_map = np.zeros((n_hd, n_jc))
    base_p = {'J_adj': 0.8, 'J_ying': 0.5, 'h_zhong': 0.0}
    for i, hd in enumerate(hd_range):
        for j, jc in enumerate(jc_range):
            p_test = base_p.copy()
            p_test['h_dang'] = hd
            p_test['J_compete'] = jc
            _, prob_s = run_evolution(p_test)
            p_f = prob_s[-1]
            entropy_map[i, j] = compute_entropy(p_f)
            uniform = np.ones(64) / 64
            kl_map[i, j] = np.sum(p_f * np.log(p_f / uniform + 1e-10))
    return entropy_map, kl_map

# 运行 Config-C（作为默认插图配置）
entropy_series, prob_series = run_evolution(configs["C-平衡"])
time_axis = np.arange(len(entropy_series)) * 10
final_p = prob_series[-1]
x = np.arange(64)

# ============================================================
# Figure 1: 熵演化图（含统计分析标注）
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(time_axis, entropy_series, 'b-', linewidth=1.5, label='熵 H(t)')
ax.axhline(y=6.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=np.min(entropy_series), color='r', linestyle=':', alpha=0.5, linewidth=1)

# 分析标注
min_ent = np.min(entropy_series)
final_ent = entropy_series[-1]
initial_ent = entropy_series[0]
drop_pct = (initial_ent - final_ent) / initial_ent * 100

ax.annotate(f'最小熵 H_min={min_ent:.2f}\n(较初始下降 {(initial_ent-min_ent)/initial_ent*100:.1f}%)',
            xy=(np.argmin(entropy_series)*10, min_ent),
            xytext=(np.argmin(entropy_series)*10+80, min_ent-0.25),
            arrowprops=dict(arrowstyle='->', color='red'), fontproperties=fp, fontsize=8, color='red')

ax.set_xlabel('时间步 t', fontproperties=fp, fontsize=12)
ax.set_ylabel('香农熵 H(t)', fontproperties=fp, fontsize=12)
ax.set_title('图1  系统香农熵演化：从"太极"均态到结构涌现', fontproperties=fp, fontsize=13)
ax.legend(fontsize=10, prop=fp)
ax.grid(True, alpha=0.3)

# 统计信息框
stats_text = (f'初始熵: {initial_ent:.2f} bit\n'
              f'最终熵: {final_ent:.2f} bit\n'
              f'最小熵: {min_ent:.2f} bit\n'
              f'熵降幅: {drop_pct:.1f}%')
ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontproperties=fp, fontsize=8,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig1_entropy_evolution.png'), dpi=200)
plt.close()
print("图1 已生成")

# ============================================================
# Figure 2: 最终概率分布（含KL散度、非均匀度标注）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
uniform_line = np.ones(64) / 64

ax.bar(x, final_p, width=0.8, color='steelblue', alpha=0.8, label='涌现概率')
ax.plot(x, uniform_line, 'r--', linewidth=1.5, label='均匀分布(1/64≈0.0156)')

# 标记Top-10
top10_idx = np.argsort(final_p)[::-1][:10]
colors = plt.cm.Reds(np.linspace(0.3, 1.0, 10))
for rank, i in enumerate(top10_idx):
    color = colors[rank]
    ax.bar(i, final_p[i], width=0.8, color=color, alpha=0.9, edgecolor='darkred', linewidth=0.5)
    ax.text(i, final_p[i] + 0.003, f'{FULL_HEX[i]}\n{final_p[i]:.3f}', 
            ha='center', va='bottom', fontproperties=fp, fontsize=6.5, fontweight='bold', color='darkred')

# 分析标注
kl_div = np.sum(final_p * np.log(final_p / uniform_line))
gini = 1 - np.sum(final_p**2)  # 简单非均匀度
top5_sum = sum(final_p[np.argsort(final_p)[::-1][:5]])

ax.annotate(f'Top-5合计: {top5_sum:.1%}', xy=(0.75, 0.92), xycoords='axes fraction',
            fontproperties=fp, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax.set_xlabel('卦索引 (0-63)', fontproperties=fp, fontsize=12)
ax.set_ylabel('概率 P(卦)', fontproperties=fp, fontsize=12)
ax.set_title('图2  最终态概率分布：64卦的非均匀涌现', fontproperties=fp, fontsize=13)
ax.legend(fontsize=10, prop=fp)
ax.set_xlim(-0.5, 63.5)
ax.grid(True, alpha=0.2)

# 统计信息框
stats_text2 = (f'KL散度(均匀参考): {kl_div:.3f}\n'
               f'Top-5概率和: {top5_sum:.3f}\n'
               f'均匀期望: {5/64:.3f}\n'
               f'非均匀增强: {top5_sum/(5/64):.1f}×')
ax.text(0.02, 0.97, stats_text2, transform=ax.transAxes, fontproperties=fp, fontsize=8,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig2_final_distribution.png'), dpi=200)
plt.close()
print("图2 已生成")

# ============================================================
# Figure 3: 八卦覆盖热图（含覆盖率标注）
# ============================================================
joint_probs = np.zeros((8, 8))
for i in range(64):
    lower = ((i >> 0) & 1) | (((i >> 1) & 1) << 1) | (((i >> 2) & 1) << 2)
    upper = ((i >> 3) & 1) | (((i >> 4) & 1) << 1) | (((i >> 5) & 1) << 2)
    joint_probs[lower, upper] += final_p[i]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(joint_probs, cmap='YlOrRd', aspect='equal', vmin=0)
ax.set_xticks(range(8))
ax.set_yticks(range(8))
ax.set_xticklabels(TRIGRAM_NAMES, fontproperties=fp, fontsize=11)
ax.set_yticklabels(TRIGRAM_NAMES, fontproperties=fp, fontsize=11)
ax.set_xlabel('上卦（外卦）', fontproperties=fp, fontsize=12)
ax.set_ylabel('下卦（内卦）', fontproperties=fp, fontsize=12)

for i in range(8):
    for j in range(8):
        val = joint_probs[i, j]
        if val > 0.02:
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8,
                   color='white' if val > 0.08 else 'black')

plt.colorbar(im, ax=ax, label='概率')

# 分析标注
lower_marg = np.sum(joint_probs, axis=1)
upper_marg = np.sum(joint_probs, axis=0)
lower_covered = sum(1 for v in lower_marg if v > 0.001)
upper_covered = sum(1 for v in upper_marg if v > 0.001)
max_trigram_lower = TRIGRAM_NAMES[np.argmax(lower_marg)]
max_trigram_upper = TRIGRAM_NAMES[np.argmax(upper_marg)]

annotation = (f'下卦覆盖: {lower_covered}/8\n'
              f'上卦覆盖: {upper_covered}/8\n'
              f'最大下卦: {max_trigram_lower} ({lower_marg.max():.3f})\n'
              f'最大上卦: {max_trigram_upper} ({upper_marg.max():.3f})\n'
              f'总覆盖: {lower_covered+upper_covered}/16')
ax.text(1.25, 0.5, annotation, transform=ax.transAxes, fontproperties=fp, fontsize=9,
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax.set_title('图3  上下卦联合概率分布（8×8八卦空间）', fontproperties=fp, fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig3_trigram_joint.png'), dpi=200)
plt.close()
print("图3 已生成")

# ============================================================
# Figure 4: 三配置Top-5演化轨迹对比
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

colors_idx = plt.cm.tab10(np.linspace(0, 1, 10))
for row, (label, params) in enumerate(configs.items()):
    ax = axes[row]
    ent_s, prob_s = run_evolution(params)
    time = np.arange(len(ent_s)) * 10
    final = prob_s[-1]
    top5 = np.argsort(final)[::-1][:5]
    
    # 背景：所有卦的演化
    for i in range(64):
        p_t = prob_s[:, i]
        if np.max(p_t) > 0.02:
            ax.plot(time, p_t, linewidth=0.5, alpha=0.15, color='gray')
    
    # 高亮Top-5
    for rank, idx in enumerate(top5):
        p_t = prob_s[:, idx]
        ax.plot(time, p_t, linewidth=2.0, color=colors_idx[rank], alpha=0.8,
                label=f'#{rank+1} {FULL_HEX[idx]}')
    
    ax.set_ylabel('概率', fontproperties=fp, fontsize=9)
    ax.set_title(f'{label}', fontproperties=fp, fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, prop=fp, ncol=5, loc='upper right')

axes[-1].set_xlabel('时间步 t', fontproperties=fp, fontsize=12)
fig.suptitle('图4  三配置下Top-5卦象概率时间演化轨迹', fontproperties=fp_bold, fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig4_top5_evolution.png'), dpi=200)
plt.close()
print("图4 已生成")

# ============================================================
# Figure 5: 三配置概率分布对比（含分析标注）
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
for idx, (label, params) in enumerate(configs.items()):
    _, prob_s = run_evolution(params)
    final_p_c = prob_s[-1]
    top5_c = np.argsort(final_p_c)[::-1][:5]
    
    ax = axes[idx]
    bars = ax.bar(x, final_p_c, width=0.8, color='steelblue', alpha=0.7)
    for i in top5_c:
        bars[i].set_color('crimson')
        bars[i].set_alpha(0.9)
        ax.text(i, final_p_c[i] + 0.008, FULL_HEX[i], 
                ha='center', va='bottom', fontproperties=fp, fontsize=7, fontweight='bold', color='darkred')
    
    # 分析标注
    kl_c = np.sum(final_p_c * np.log(final_p_c / (np.ones(64)/64)))
    top5_sum_c = sum(final_p_c[np.argsort(final_p_c)[::-1][:5]])
    ax.text(0.98, 0.93, f'KL={kl_c:.2f}  Top5合={top5_sum_c:.3f}',
            transform=ax.transAxes, ha='right', fontproperties=fp, fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_ylabel('概率', fontproperties=fp, fontsize=10)
    ax.set_title(f'{label}', fontproperties=fp, fontsize=11)
    ax.set_xlim(-0.5, 63.5)
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('卦索引 (0-63)', fontproperties=fp, fontsize=11)
fig.suptitle('图5  三种参数配置下的涌现概率分布对比', fontproperties=fp_bold, fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig5_configs_comparison.png'), dpi=200)
plt.close()
print("图5 已生成")

# ============================================================
# Figure 6: 错卦/综卦关系网络图（含对位分析）
# ============================================================
def find_cuo_zong(top_idx):
    """找出错卦和综卦关系"""
    cuo, zong = [], []
    for j in range(len(top_idx)):
        yj = int_to_yao(top_idx[j])
        opp = yao_to_int([1-x for x in yj])
        rev = yao_to_int(yj[::-1])
        for k in range(j+1, len(top_idx)):
            if top_idx[k] == opp:
                cuo.append((top_idx[j], top_idx[k]))
            if top_idx[k] == rev:
                zong.append((top_idx[j], top_idx[k]))
    return cuo, zong

# 用Config-C做网络图（错综关系更丰富）
_, prob_b = run_evolution(configs["C-平衡"])
final_p_b = prob_b[-1]
top_idx_b = np.argsort(final_p_b)[::-1][:10]
cuo_pairs, zong_pairs = find_cuo_zong(top_idx_b)

fig, ax = plt.subplots(figsize=(9, 7))

# 圆周布局
n_nodes = len(top_idx_b)
angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
r = 0.35
xs = 0.5 + r * np.cos(angles)
ys = 0.5 + r * np.sin(angles)
node_pos = {top_idx_b[i]: (xs[i], ys[i]) for i in range(n_nodes)}

# 画边
all_edges = []
for a, b in cuo_pairs:
    all_edges.append((a, b, '错卦', '#E74C3C'))
for a, b in zong_pairs:
    all_edges.append((a, b, '综卦', '#2980B9'))

for a, b, label, color in all_edges:
    x1, y1 = node_pos[a]
    x2, y2 = node_pos[b]
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.0, alpha=0.7, zorder=1)
    mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
    ax.text(mid_x, mid_y, label, fontproperties=fp, fontsize=8, color=color,
           ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor=color))

# 画节点
sizes = [final_p_b[i] * 3000 + 300 for i in top_idx_b]
for i, idx in enumerate(top_idx_b):
    x, y = node_pos[idx]
    sz = sizes[i]
    ax.scatter(x, y, s=sz, c='#3498DB', alpha=0.8, edgecolors='#2C3E50', linewidth=1.5, zorder=5)
    ax.text(x, y-0.06, FULL_HEX[idx], ha='center', va='top', fontproperties=fp, fontsize=10, fontweight='bold')
    ax.text(x, y+0.06, f'{final_p_b[idx]:.3f}', ha='center', va='bottom', fontproperties=fp, fontsize=8)

# 分析标注
n_cuo = len(cuo_pairs)
n_zong = len(zong_pairs)
total_pairs = len(set(all_edges))
max_comb_prob = 0
for a, b, _, _ in all_edges:
    cp = final_p_b[a] + final_p_b[b]
    max_comb_prob = max(max_comb_prob, cp)

ax.text(0.02, 0.98, f'Config-C分析:\n错卦{len(cuo_pairs)}对\n综卦{len(zong_pairs)}对\n最大联合概率: {max_comb_prob:.3f}',
        transform=ax.transAxes, fontproperties=fp, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('图6  涌现卦象错综关系网络（Config-C，8组关系）', fontproperties=fp_bold, fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig6_network.png'), dpi=200)
plt.close()
print("图6 已生成")

# ============================================================
# Figure 7: 参数空间扫描热图（含最优区标注）
# ============================================================
hd_vals = np.array([0.0, 0.1, 0.2, 0.3, 0.5])
jc_vals = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
entropy_map, kl_map = parameter_scan(hd_vals, jc_vals)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：熵
ax = axes[0]
im1 = ax.imshow(entropy_map, cmap='viridis_r', aspect='auto',
               extent=[min(jc_vals), max(jc_vals), max(hd_vals), min(hd_vals)])
ax.set_xlabel('竞争强度 J_compete', fontproperties=fp, fontsize=12)
ax.set_ylabel('当位场强 h_dang', fontproperties=fp, fontsize=12)
ax.set_title('最终熵（越低=结构越强）', fontproperties=fp, fontsize=12)
# 标注最优区
circle = plt.Circle((0.5, 0.2), 0.15, color='red', fill=False, linewidth=2.5, linestyle='--')
ax.add_patch(circle)
# 标注三个配置位置
ax.plot(0.5, 0.2, 'r*', markersize=15, label='Config-C (C-平衡)')
ax.plot(1.0, 0.1, 'b*', markersize=12, label='Config-B')
ax.plot(0.3, 0.5, 'g*', markersize=10, label='Config-A')
ax.legend(fontsize=8, prop=fp)
cbar1 = plt.colorbar(im1, ax=ax, label='熵 H (bit)')

# 右图：KL散度
ax = axes[1]
im2 = ax.imshow(kl_map, cmap='inferno', aspect='auto',
               extent=[min(jc_vals), max(jc_vals), max(hd_vals), min(hd_vals)])
ax.set_xlabel('竞争强度 J_compete', fontproperties=fp, fontsize=12)
ax.set_ylabel('当位场强 h_dang', fontproperties=fp, fontsize=12)
ax.set_title('KL散度（越大=非均匀越强）', fontproperties=fp, fontsize=12)
circle2 = plt.Circle((0.5, 0.2), 0.15, color='white', fill=False, linewidth=2.5, linestyle='--')
ax.add_patch(circle2)
ax.plot(0.5, 0.2, 'w*', markersize=15)
ax.plot(1.0, 0.1, 'c*', markersize=12)
ax.plot(0.3, 0.5, marker='*', markersize=10, color='lime')
cbar2 = plt.colorbar(im2, ax=ax, label='KL散度')

fig.suptitle('图7  参数空间扫描分析', fontproperties=fp_bold, fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, 'fig7_parameter_scan.png'), dpi=200)
plt.close()
print("图7 已生成")

# ============================================================
# 统计
# ============================================================
print(f"\n插图文件大小:")
for f in sorted(os.listdir(fig_dir)):
    path = os.path.join(fig_dir, f)
    if f.endswith('.png'):
        sz = os.path.getsize(path) / 1024
        print(f"  {f}: {sz:.1f} KB")

print(f"\n所有插图已生成完毕（CJK字体正常渲染）！")
