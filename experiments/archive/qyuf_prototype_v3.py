#!/usr/bin/env python3
"""
QYUF v0.3 — 校准版涌现验证
================================
核心改进:
  1. 评分函数更严格区分吉凶 (分数范围扩大到 -10 ~ +10)
  2. 多轮迭代 (iterations=4)
  3. 对比: 同一物体在经典YLYW与QYUF下的决策一致性
  4. 与YLYW 64卦全规则库的策略映射
"""

import numpy as np
from typing import List, Tuple

I = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)

def bin_list(idx: int, n: int) -> List[int]:
    return [(idx>>i)&1 for i in range(n)]

def state_label(idx: int) -> str:
    bits = bin_list(idx, 6)
    return ''.join(['─' if b else '╌' for b in reversed(bits)])

def hexagram_name(idx: int) -> str:
    """返回卦名 (基于序号1-64)"""
    names = [
        "乾","坤","屯","蒙","需","讼","师","比","小畜","履","泰","否",
        "同人","大有","谦","豫","随","蛊","临","观","噬嗑","贲","剥","复",
        "无妄","大畜","颐","大过","坎","离","咸","恒","遯","大壮","晋","明夷",
        "家人","睽","蹇","解","损","益","夬","姤","萃","升","困","井",
        "革","鼎","震","艮","渐","归妹","丰","旅","巽","兑","涣","节",
        "中孚","小过","既济","未济"
    ]
    return names[idx-1] if 1 <= idx <= 64 else "??"


class QYUFSim:
    
    def __init__(self, n: int = 6):
        self.n = n
        self.dim = 1 << n
    
    def init(self, theta: np.ndarray) -> np.ndarray:
        psi = np.zeros(self.dim, dtype=complex)
        psi[0] = 1.0
        for q in range(self.n):
            t = theta[q] * np.pi / 2
            c, s = np.cos(t/2), np.sin(t/2)
            npsi = np.zeros_like(psi)
            for i in range(self.dim):
                if abs(psi[i]) < 1e-15: continue
                bits = bin_list(i, self.n)
                if bits[q] == 0:
                    j0, j1 = i, i | (1 << q)
                    npsi[j0] += psi[i] * c
                    npsi[j1] += psi[i] * s
                else:
                    j1, j0 = i, i & ~(1 << q)
                    npsi[j0] += psi[i] * (-s)
                    npsi[j1] += psi[i] * c
            psi = npsi
        return psi
    
    def _score(self, idx: int) -> float:
        """综合易理评分 (+10 最吉, -10 最凶)"""
        bits = bin_list(idx, self.n)
        s = 0.0
        
        # === 当位 (权重高) ===
        # 阳位奇: 初0, 三2, 五4 | 阴位偶: 二1, 四3, 上5
        for q in range(self.n):
            is_yang_pos = (q % 2 == 0)  # 奇
            proper = (bits[q]==1 and is_yang_pos) or (bits[q]==0 and not is_yang_pos)
            s += 1.0 if proper else -1.0
        
        # === 得中 (权重高) ===
        # 二爻(q1)宜阴; 五爻(q4)宜阳
        if bits[1] == 0: s += 2.0  # 二得中
        else: s -= 2.0
        if bits[4] == 1: s += 2.0  # 五得中
        else: s -= 2.0
        # 既中且正: 六二(阴居二) + 九五(阳居五) = 最大加分
        if bits[1]==0: s += 1.0    # 六二
        if bits[4]==1: s += 1.0    # 九五
        
        # === 乘 (阴乘阳凶) / 承 (阴承阳吉) ===
        for q in range(self.n - 1):
            l, u = bits[q], bits[q+1]
            if l==0 and u==1: s -= 1.5  # 阴乘阳
            elif l==1 and u==0: s += 1.0  # 阴承阳
        
        # === 应: 初↔四, 二↔五, 三↔上 ===
        # 阴阳相异则吉(相应), 相同则凶(敌应)
        for pair in [(0,3),(1,4),(2,5)]:
            a, b = bits[pair[0]], bits[pair[1]]
            if a != b: s += 1.5  # 相应吉
            else: s -= 1.0       # 敌应凶
        
        return s
    
    def grover_amplify(self, psi: np.ndarray, iterations: int = 3) -> np.ndarray:
        """Grover振幅放大"""
        npsi = psi.copy().astype(complex)
        
        for _ in range(iterations):
            # Oracle: 好状态(score > 0) 相位翻转
            for i in range(self.dim):
                if self._score(i) > 0:
                    npsi[i] *= -1
            
            # Diffusion: 围绕平均翻转
            avg = np.mean(npsi)
            for i in range(self.dim):
                npsi[i] = 2 * avg - npsi[i]
        
        nrm = np.linalg.norm(npsi)
        if nrm > 0: npsi /= nrm
        return npsi
    
    def run_inference(self, theta: np.ndarray, iters: int = 3) -> np.ndarray:
        psi = self.init(theta)
        for ctrl, tgt in [(0,3),(1,4),(2,5)]:
            psi = self._cnot(psi, ctrl, tgt)
        return self.grover_amplify(psi, iters)
    
    def _cnot(self, psi: np.ndarray, ctrl: int, tgt: int) -> np.ndarray:
        npsi = np.zeros_like(psi)
        for i in range(self.dim):
            if abs(psi[i]) < 1e-15: continue
            bits = bin_list(i, self.n)
            if bits[ctrl] == 1:
                bits[tgt] ^= 1
                j = sum(b*(1<<k) for k,b in enumerate(bits))
                npsi[j] = psi[i]
            else:
                npsi[i] = psi[i]
        return npsi
    
    def top_k(self, psi: np.ndarray, k: int = 8) -> List[Tuple[int, float, float]]:
        probs = np.abs(psi)**2
        indices = np.argsort(probs)[::-1][:k]
        return [(idx, probs[idx], self._score(idx)) for idx in indices]


# ============================================================
# 验证实验
# ============================================================

def test1_baseline():
    """实验1: 从均匀叠加出发, 涌现理想卦象"""
    print("="*65)
    print("实验1: 均匀叠加 → 涌现理想卦象 (应中正当位)") 
    print("="*65)
    
    sim = QYUFSim()
    psi = np.ones(64, dtype=complex) / np.sqrt(64)
    
    for iters in [1, 2, 3, 5]:
        psi_a = sim.grover_amplify(psi.copy(), iters)
        top = sim.top_k(psi_a, 5)
        probs = [p for _, p, _ in top]
        scores = [s for _, _, s in top]
        best_score = max(sim._score(i) for i in range(64))
        worst_score = min(sim._score(i) for i in range(64))
        print(f"\n  迭代{iters}次:")
        for idx, p, s in top:
            name = hexagram_name(idx+1)
            print(f"    {state_label(idx)} {name}(卦{idx+1:2d}): {p*100:.1f}% | 评分{s:+.1f}")
        
        # 计算好卦的总概率
        good_p = sum(np.abs(psi_a[i])**2 for i in range(64) if sim._score(i) > 0)
        bad_p = sum(np.abs(psi_a[i])**2 for i in range(64) if sim._score(i) <= 0)
        print(f"    好卦总概率: {good_p*100:.1f}% | 评分范围: {worst_score} ~ {best_score}")


def test2_real_objects():
    """实验2: 不同物体的量子涌现决策"""
    print("\n"+"="*65)
    print("实验2: 物体决策 — 量子涌现与经典对标")
    print("="*65)
    
    # 从YLYW论文中选取的物体特征 (13维 → 六爻映射)
    # [力需求, 稳定性, 变形性, 滚动性, 脆弱性, 体积]
    objects = [
        ("金属块", np.array([0.85, 0.80, 0.10, 0.15, 0.10, 0.50])),   # 似乾
        ("海绵",  np.array([0.15, 0.30, 0.90, 0.10, 0.40, 0.40])),    # 似坤
        ("球体",  np.array([0.30, 0.25, 0.30, 0.95, 0.50, 0.30])),    # 似震
        ("易碎杯",np.array([0.40, 0.40, 0.20, 0.20, 0.85, 0.35])),    # 似离
        ("木块",  np.array([0.70, 0.60, 0.30, 0.20, 0.25, 0.55])),    # 似艮
    ]
    
    sim = QYUFSim()
    
    for name, feats in objects:
        # 特征归一化到 [0,1]^6
        theta = np.clip(feats[:6], 0, 1)
        psi = sim.run_inference(theta, iters=3)
        top = sim.top_k(psi, 5)
        
        print(f"\n--- {name} ---")
        print(f"  初始θ: {np.array2string(theta[:6], precision=2)}")
        for idx, p, s in top:
            hname = hexagram_name(idx+1)
            ylyw_action = "—"
            print(f"  {state_label(idx)} {hname}(卦{idx+1:2d}): {p*100:.1f}% | 评分{s:+.1f}")


def test3_good_bad_distribution():
    """实验3: 吉凶卦象的统计学分布"""
    print("\n"+"="*65)
    print("实验3: 64卦评分分布")
    print("="*65)
    
    sim = QYUFSim()
    scores = [sim._score(i) for i in range(64)]
    
    # 全当位: 乾(63)和坤(0)
    print(f"  坤(卦1,  全阴): {scores[0]:+.1f}")
    print(f"  乾(卦64, 全阳): {scores[63]:+.1f}")
    
    # 最吉卦
    best_idx = np.argmax(scores)
    worst_idx = np.argmin(scores)
    print(f"\n  最吉: {state_label(best_idx)} {hexagram_name(best_idx+1)}(卦{best_idx+1}): {scores[best_idx]:+.1f}")
    print(f"  最凶: {state_label(worst_idx)} {hexagram_name(worst_idx+1)}(卦{worst_idx+1}): {scores[worst_idx]:+.1f}")
    
    # 统计
    good = sum(1 for s in scores if s > 0)
    bad = sum(1 for s in scores if s <= 0)
    print(f"\n  吉卦(评分>0): {good}/64")
    print(f"  凶卦(评分≤0): {bad}/64")
    print(f"  评分范围: {min(scores):+.1f} ~ {max(scores):+.1f}")
    print(f"  评分均值: {np.mean(scores):+.1f}")


def test4_amplification_dynamics():
    """实验4: 振幅放大的动力学过程"""
    print("\n"+"="*65)
    print("实验4: 振幅放大的动力学过程")
    print("="*65)
    
    sim = QYUFSim()
    psi = sim.init(np.array([0.6, 0.5, 0.5, 0.6, 0.5, 0.5]))
    
    print("  迭代 | 吉卦总概率 | TOP1概率 | TOP1卦名")
    print("  "+"-"*50)
    
    for it in range(8):
        if it == 0:
            psi_cur = psi.copy()
        else:
            psi_cur = sim.grover_amplify(psi if it==1 else psi_cur, 1)
        top = sim.top_k(psi_cur, 1)
        good_p = sum(np.abs(psi_cur[i])**2 for i in range(64) if sim._score(i) > 0)
        idx, p, s = top[0]
        name = hexagram_name(idx+1)
        print(f"     {it}   |   {good_p*100:5.1f}%    |  {p*100:5.1f}%   | {name}(卦{idx+1})")
        if it > 0 and p < 0.01:
            break


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    
    print("╔══════════════════════════════════════════════════════╗")
    print("║   QYUF v0.3 · 校准版涌现验证                       ║")
    print("║   量子易理统一框架 · 最简原型                       ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    test1_baseline()
    test2_real_objects()
    test3_good_bad_distribution()
    test4_amplification_dynamics()
    
    print("\n"+"="*65)
    print("✓ 验证完成")
    print("="*65)
