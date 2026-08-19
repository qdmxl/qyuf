#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《卦者，爻际干涉之瞬时显象也》— 实验验证仿真

核心假说：
  如果"刚柔相摩，八卦相荡"对应于量子干涉，那么设计一个编码了
  "乘承比应"等爻际规则的酉变换U，让6个量子位在其中不断演化，
  系统应该会自发地呈现出64个相对稳定的"吸引子"状态，
  且这些状态与《易经》六十四卦在结构上存在高度对应。

作者：马老师课题组
日期：2026-07-29
"""

import numpy as np
from scipy.linalg import expm
import itertools
import json
import os

# ============================================================
# 第一部分：基本常量与易经卦象系统
# ============================================================

N_QUBITS = 6           # 六爻
DIM = 2 ** N_QUBITS    # 64维希尔伯特空间

# 六十四卦的二进制编码
# 按传统顺序：上经30卦 + 下经34卦
# 二进制编码规则：从初爻（LSB）到上爻（MSB），阴=0，阳=1
# 卦名列表（按通行本序）
HEXAGRAM_NAMES = [
    "乾", "坤", "屯", "蒙", "需", "讼", "师", "比",
    "小畜", "履", "泰", "否", "同人", "大有", "谦", "豫",
    "随", "蛊", "临", "观", "噬嗑", "贲", "剥", "复",
    "无妄", "大畜", "颐", "大过", "坎", "离",
    "咸", "恒", "遁", "大壮", "晋", "明夷", "家人", "睽",
    "蹇", "解", "损", "益", "夬", "姤", "萃", "升",
    "困", "井", "革", "鼎", "震", "艮", "渐", "归妹",
    "丰", "旅", "巽", "兑", "涣", "节", "中孚", "小过",
    "既济", "未济"
]

def binary_to_hexagram(binary_int):
    """将6位整数映射为卦名（基于通行本次序）"""
    return HEXAGRAM_NAMES[binary_int]

def int_to_yao(n):
    """将0-63整数转换为六爻数组 [初爻, 二爻, ..., 上爻]，阴=0，阳=1"""
    return np.array([(n >> i) & 1 for i in range(N_QUBITS)])

def yao_to_int(yao):
    """六爻数组转整数"""
    return sum(int(yao[i]) << i for i in range(N_QUBITS))

def hexagram_symbol(n):
    """返回卦的阴阳符号表示，从上爻到初爻"""
    yao = int_to_yao(n)
    symbols = []
    for i in range(N_QUBITS - 1, -1, -1):
        symbols.append("━" if yao[i] == 1 else "┅")
    return "".join(symbols)

# ============================================================
# 第二部分：酉变换 U(θ) 的设计
# ============================================================

def pauli_z(i, n=N_QUBITS):
    """返回作用在第i个量子位上的Z算符的（展开到n位的）矩阵"""
    ops = []
    for j in range(n):
        if j == i:
            ops.append(np.array([[1, 0], [0, -1]], dtype=complex))
        else:
            ops.append(np.eye(2, dtype=complex))
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def pauli_x(i, n=N_QUBITS):
    """返回作用在第i个量子位上的X算符"""
    ops = []
    for j in range(n):
        if j == i:
            ops.append(np.array([[0, 1], [1, 0]], dtype=complex))
        else:
            ops.append(np.eye(2, dtype=complex))
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def zz_interaction(i, j, n=N_QUBITS):
    """返回量子位i和j之间的ZZ相互作用算符 Z_i @ Z_j"""
    ops = []
    for k in range(n):
        if k == i or k == j:
            ops.append(np.array([[1, 0], [0, -1]], dtype=complex))
        else:
            ops.append(np.eye(2, dtype=complex))
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def build_hamiltonian(theta, params=None):
    """
    构建编码了"刚柔相摩"规则的哈密顿量 H(θ)
    
    四条规则编码为哈密顿量中的不同项：
    
    规则1 - 阴阳相引（相邻爻位间）：
        H_adj = Σ_{adjacent i,j} (Z_i ⊗ Z_j)
        相邻爻位一阴一阳 → 正相位（合作）
        同阴同阳 → 负相位（竞争）
    
    规则2 - 远距感应（应位爻之间）：
        H_ying = Z_0⊗Z_3 + Z_1⊗Z_4 + Z_2⊗Z_5
        初应四、二应五、三应上
    
    规则3 - 当位倾向：
        H_dang = Σ_i (-1)^{i} Z_i
        奇位（初、三、五）倾向阳 → Z负能量低（倾向 |1⟩）
        偶位（二、四、上）倾向阴 → Z正能量低（倾向 |0⟩）
    
    规则4 - 得中强化：
        H_zhong = 作用于二爻和五爻，当它们"得中"时给予额外稳定度
        二爻（index 1）当位为阴，五爻（index 4）当位为阳
    """
    if params is None:
        params = {
            'J_adj': 1.0,        # 相邻耦合强度
            'J_ying': 0.8,       # 应位耦合强度
            'h_dang': 0.6,       # 当位场强度
            'h_zhong': 0.4,      # 得中强化
            'J_compete': 0.2,    # 竞争项
        }
    
    H = np.zeros((DIM, DIM), dtype=complex)
    
    # 规则1：阴阳相引 - 相邻爻位 ZZ 耦合
    adjacent_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    for i, j in adjacent_pairs:
        H += params['J_adj'] * zz_interaction(i, j)
    
    # 规则2：远距感应 - 应位爻 ZZ 耦合
    ying_pairs = [(0, 3), (1, 4), (2, 5)]
    for i, j in ying_pairs:
        H += params['J_ying'] * zz_interaction(i, j)
    
    # 规则3：当位倾向
    # 奇位（初0、三2、五4）倾向阳 → -Z
    # 偶位（二1、四3、上5）倾向阴 → +Z
    for i in range(N_QUBITS):
        sign = -1.0 if i % 2 == 0 else 1.0  # 奇偶修正
        H += params['h_dang'] * sign * pauli_z(i)
    
    # 规则4：得中强化
    # 二爻（index 1）当位为阴: +Z 方向
    H += params['h_zhong'] * pauli_z(1)
    # 五爻（index 4）当位为阳: -Z 方向
    H += params['h_zhong'] * (-1) * pauli_z(4)
    
    # 额外：竞争/非谐项 — 引入非平凡动力学
    # 使用X算符在相邻位间产生跃迁
    for i in range(N_QUBITS - 1):
        H += params['J_compete'] * (
            pauli_x(i) @ pauli_x(i+1, n=N_QUBITS) +
            pauli_y(i) @ pauli_y(i+1, n=N_QUBITS)
        )
    
    return H

def pauli_y(i, n=N_QUBITS):
    """返回作用在第i个量子位上的Y算符"""
    ops = []
    for j in range(n):
        if j == i:
            ops.append(np.array([[0, -1j], [1j, 0]], dtype=complex))
        else:
            ops.append(np.eye(2, dtype=complex))
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def unitary_evolution(H, dt=0.1):
    """从哈密顿量生成酉算符 U = exp(-i * H * dt)"""
    return expm(-1j * H * dt)


# ============================================================
# 第三部分：演化引擎
# ============================================================

def initialize_state():
    """初始化为均等叠加态（太极）：所有计算基的等幅叠加"""
    psi = np.ones(DIM, dtype=complex) / np.sqrt(DIM)
    return psi

def evolve(psi, U, steps):
    """
    迭代施加酉变换 U
    
    参数：
        psi: 初始态矢量
        U: 酉算符
        steps: 迭代步数
    
    返回：
        states: 每个时间步的态矢量列表
        probs: 每个时间步的概率分布 [steps+1, 64]
    """
    states = [psi.copy()]
    probs = [np.abs(psi) ** 2]
    
    current = psi.copy()
    for _ in range(steps):
        current = U @ current
        states.append(current.copy())
        probs.append(np.abs(current) ** 2)
    
    return states, np.array(probs)


# ============================================================
# 第四部分：稳定模式分析
# ============================================================

def find_attractors(prob_series, threshold=0.01, min_stability=5):
    """
    从演化轨迹中识别稳定模式（吸引子）
    
    参数：
        prob_series: 概率分布序列 [time_steps, 64]
        threshold: 被视为显著模式的最小概率阈值
        min_stability: 连续稳定所需的最少时间步数
    
    返回：
        attractors: 稳定模式列表 [(卦索引, 概率, 首次出现时间, 持续时间), ...]
    """
    n_steps, n_basis = prob_series.shape
    dominant_states = np.argmax(prob_series, axis=1)
    
    attractors = []
    current_state = dominant_states[0]
    current_prob = prob_series[0, current_state]
    start_step = 0
    count = 1
    
    for t in range(1, n_steps):
        if dominant_states[t] == current_state:
            count += 1
        else:
            if count >= min_stability and current_prob >= threshold:
                attractors.append((int(current_state), float(current_prob), start_step, count))
            current_state = dominant_states[t]
            current_prob = prob_series[t, current_state]
            start_step = t
            count = 1
    
    # 检查最后一个
    if count >= min_stability and current_prob >= threshold:
        attractors.append((int(current_state), float(current_prob), start_step, count))
    
    return attractors


def compute_entropy(probs):
    """计算概率分布的香农熵"""
    p = np.array(probs)
    p = p[p > 1e-12]
    return -np.sum(p * np.log2(p))


def compute_purity(rho):
    """计算密度矩阵的纯度 Tr(ρ²)"""
    return np.real(np.trace(rho @ rho))


def density_matrix(psi):
    """从态矢量构建密度矩阵"""
    return np.outer(psi, np.conj(psi))


def analyze_structure(prob_series):
    """
    分析涌现结构的统计特征
    
    返回结构化分析结果
    """
    final_probs = prob_series[-1]
    sorted_idx = np.argsort(final_probs)[::-1]
    top_k = 10
    
    result = {
        'top_states': [],
        'n_significant': int(np.sum(final_probs > 0.01)),
        'entropy': float(compute_entropy(final_probs)),
    }
    
    for idx in sorted_idx[:top_k]:
        if final_probs[idx] < 0.001:
            break
        yao = int_to_yao(idx)
        result['top_states'].append({
            'index': int(idx),
            'name': binary_to_hexagram(idx),
            'symbol': hexagram_symbol(idx),
            'probability': float(final_probs[idx]),
            'yao': [int(y) for y in yao],
            'yang_count': int(np.sum(yao)),
        })
    
    return result


# ============================================================
# 第五部分：卦象结构相似度分析
# ============================================================

def yao_similarity(idx1, idx2):
    """计算两卦之间的爻位相似度（0-1），1为完全相同"""
    y1 = int_to_yao(idx1)
    y2 = int_to_yao(idx2)
    return np.mean(y1 == y2)


def build_hexagram_similarity_matrix():
    """构建64卦之间的相似度矩阵"""
    n = 64
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = yao_similarity(i, j)
    return sim


def cluster_attractors(attractor_indices, similarity_matrix, threshold=0.67):
    """将相近的吸引子聚类"""
    if len(attractor_indices) <= 1:
        return attractor_indices
    
    clusters = []
    used = set()
    
    for i, idx in enumerate(attractor_indices):
        if idx in used:
            continue
        cluster = [idx]
        used.add(idx)
        for j, idx2 in enumerate(attractor_indices):
            if idx2 in used:
                continue
            if np.all(similarity_matrix[idx, idx2] >= threshold) or \
               np.any([similarity_matrix[idx, c] >= threshold for c in cluster]):
                cluster.append(idx2)
                used.add(idx2)
        clusters.append(cluster)
    
    return clusters


# ============================================================
# 第六部分：主实验流程
# ============================================================

def run_experiment(params=None, dt=0.1, steps=200, theta=1.0):
    """
    运行完整的实验流程
    
    返回实验结果的详细字典
    """
    print("=" * 60)
    print("《卦者，爻际干涉之瞬时显象也》— 实验验证仿真")
    print("=" * 60)
    
    # Step 1: 构建哈密顿量
    print(f"\n[Step 1] 构建编码'刚柔相摩'规则的哈密顿量 H(θ) ...")
    H = build_hamiltonian(theta, params)
    print(f"         哈密顿量维度: {H.shape}")
    
    # Step 2: 生成酉算符
    print(f"\n[Step 2] 生成酉算符 U = exp(-iH·dt), dt = {dt} ...")
    U = unitary_evolution(H, dt)
    
    # Step 3: 初始化太极态
    print(f"\n[Step 3] 初始化均等叠加态（太极）...")
    psi = initialize_state()
    initial_probs = np.abs(psi) ** 2
    print(f"         初始熵: {compute_entropy(initial_probs):.4f}")
    
    # Step 4: 演化
    print(f"\n[Step 4] 迭代酉演化 {steps} 步 ...")
    states, probs = evolve(psi, U, steps)
    print(f"         最终熵: {compute_entropy(probs[-1]):.4f}")
    
    # Step 5: 识别稳定模式
    print(f"\n[Step 5] 识别稳定模式（吸引子）...")
    attractors = find_attractors(probs)
    print(f"         发现 {len(attractors)} 个稳定模式")
    
    # Step 6: 结构分析
    print(f"\n[Step 6] 涌现结构分析 ...")
    analysis = analyze_structure(probs)
    
    # 构建结果
    result = {
        'params': params if params else {
            'J_adj': 1.0, 'J_ying': 0.8, 'h_dang': 0.6,
            'h_zhong': 0.4, 'J_compete': 0.2
        },
        'dt': dt,
        'steps': steps,
        'theta': theta,
        'final_entropy': float(compute_entropy(probs[-1])),
        'initial_entropy': float(compute_entropy(initial_probs)),
        'attractors': attractors,
        'analysis': analysis,
        'prob_series_compressed': probs[::max(1, steps//50)],  # 降采样以存储
    }
    
    return result


def print_results(result):
    """打印实验结果"""
    print("\n" + "=" * 60)
    print("实验结果摘要")
    print("=" * 60)
    
    print(f"\n初始熵: {result['initial_entropy']:.4f}")
    print(f"最终熵: {result['final_entropy']:.4f}")
    
    params = result['params']
    print(f"\n哈密顿量参数:")
    for key, val in params.items():
        print(f"  {key}: {val:.2f}")
    
    print(f"\n稳定模式（吸引子）:")
    for idx, prob, start, duration in result['attractors']:
        name = binary_to_hexagram(idx)
        symbol = hexagram_symbol(idx)
        yao = int_to_yao(idx)
        yang = int(np.sum(yao))
        print(f"  {name:4s} {symbol:6s} | 索引={idx:2d} | 阳数={yang} | "
              f"概率={prob:.3f} | 时间步={start}-{start+duration-1} | 持续={duration}")
    
    print(f"\nTop-10 涌现模式:")
    print(f"  {'排名':<4s} {'卦名':<4s} {'卦象':<8s} {'阳数':>4s} {'概率':>8s}")
    print(f"  {'-'*32}")
    for i, st in enumerate(result['analysis']['top_states']):
        print(f"  {i+1:<4d} {st['name']:<4s} {st['symbol']:<8s} "
              f"{st['yang_count']:>4d} {st['probability']:>8.4f}")


def save_result(result, filename=None):
    """保存实验结果"""
    if filename is None:
        filename = 'experiment_result.json'
    
    # 转为可序列化
    serializable = {
        'params': result['params'],
        'dt': result['dt'],
        'steps': result['steps'],
        'final_entropy': result['final_entropy'],
        'initial_entropy': result['initial_entropy'],
        'attractors': [(int(i), float(p), int(s), int(d)) 
                       for i, p, s, d in result['attractors']],
        'analysis': result['analysis'],
    }
    
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {filepath}")
    return filepath


# ============================================================
# 第七部分：参数扫描与敏感性分析
# ============================================================

def parameter_scan(param_name, values, base_params=None, dt=0.1, steps=200, theta=1.0):
    """对单个参数进行扫描，观察系统行为变化"""
    if base_params is None:
        base_params = {
            'J_adj': 1.0, 'J_ying': 0.8, 'h_dang': 0.6,
            'h_zhong': 0.4, 'J_compete': 0.2
        }
    
    results = []
    for val in values:
        p = base_params.copy()
        p[param_name] = val
        res = run_experiment(params=p, dt=dt, steps=steps, theta=theta)
        
        # 提取关键指标
        n_attractors = len(res['attractors'])
        
        results.append({
            'param_value': val,
            'final_entropy': res['final_entropy'],
            'n_attractors': n_attractors,
            'top_state': res['analysis']['top_states'][0] if res['analysis']['top_states'] else None,
        })
    
    return results


# ============================================================
# 第八部分：可视化
# ============================================================

def plot_evolution(result, save_path=None):
    """绘制概率演化图"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("matplotlib不可用，跳过绘图")
        return
    
    probs = result['prob_series_compressed']
    steps = probs.shape[0]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 图1: 前20个基态的概率演化热图
    ax1 = axes[0]
    top_n = 20
    top_idx = np.argsort(probs[-1])[::-1][:top_n]
    
    extent = [0, steps-1, 0, top_n-1]
    im = ax1.imshow(probs[:, top_idx].T, aspect='auto', cmap='hot', 
                    extent=extent, interpolation='nearest')
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([f"{binary_to_hexagram(i)}({i})" for i in top_idx], fontsize=7)
    ax1.set_xlabel('时间步')
    ax1.set_ylabel('卦象')
    ax1.set_title('Top-20 基态概率演化')
    plt.colorbar(im, ax=ax1)
    
    # 图2: 熵演化
    ax2 = axes[1]
    entropies = [compute_entropy(p) for p in probs]
    ax2.plot(entropies, 'b-', linewidth=1)
    ax2.set_xlabel('时间步')
    ax2.set_ylabel('香农熵')
    ax2.set_title('系统熵演化（从太极到结构涌现）')
    ax2.grid(True, alpha=0.3)
    
    # 图3: 最终概率分布
    ax3 = axes[2]
    final = probs[-1]
    x = np.arange(64)
    bars = ax3.bar(x, final, width=0.8, color='steelblue', alpha=0.8)
    # 标记Top-5
    top5 = np.argsort(final)[::-1][:5]
    for i in top5:
        bars[i].set_color('crimson')
        ax3.text(i, final[i] + 0.01, binary_to_hexagram(i), 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax3.set_xlabel('卦索引 (0-63)')
    ax3.set_ylabel('概率')
    ax3.set_title('最终态概率分布')
    ax3.set_xlim(-0.5, 63.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图形已保存至: {save_path}")
    
    plt.show()


# ============================================================
# 第九部分：主函数
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='《易经》量子干涉仿真实验')
    parser.add_argument('--dt', type=float, default=0.1, help='时间步长')
    parser.add_argument('--steps', type=int, default=200, help='演化步数')
    parser.add_argument('--theta', type=float, default=1.0, help='整体缩放因子')
    parser.add_argument('--scan', type=str, default=None, 
                        help='参数扫描: 参数名 (如 J_adj)')
    parser.add_argument('--save', action='store_true', help='保存结果')
    parser.add_argument('--plot', action='store_true', help='绘制图形')
    
    args = parser.parse_args()
    
    if args.scan:
        values = np.linspace(0.0, 2.0, 9)
        results = parameter_scan(args.scan, values, dt=args.dt, 
                                 steps=args.steps, theta=args.theta)
        print(f"\n参数扫描: {args.scan}")
        print(f"{'值':>8s} {'最终熵':>8s} {'吸引子数':>8s} {'Top卦':>6s}")
        print("-" * 36)
        for r in results:
            top = r['top_state']['name'] if r['top_state'] else 'N/A'
            print(f"{r['param_value']:>8.2f} {r['final_entropy']:>8.4f} "
                  f"{r['n_attractors']:>8d} {top:>6s}")
    else:
        result = run_experiment(dt=args.dt, steps=args.steps, theta=args.theta)
        print_results(result)
        
        if args.save:
            save_result(result)
        
        if args.plot:
            plot_evolution(result, save_path='evolution_plot.png')
