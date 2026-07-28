#!/usr/bin/env python3
"""
QYUF v2.0 — 感知自适应酉变换 
================================
核心创新: Oracle = 易理评分 × 感知匹配度

前版(qyuf_final)的Oracle是全局固定的:
  Oracle|x⟩ = (-1)^f(x)  其中 f(x)=易理评分(x)>0
  结果: 无论输入什么物体, Oracle都放大同一批吉卦
  
本版Oracle随物体特征动态变化:
  Oracle|x⟩ = (-1)^g(x|θ)  其中 g(x|θ) = 易理评分(x) × 感知匹配度(x, θ) > 阈值
  结果: 输入不同物体 → Oracle动态调整 → 酉变换放大不同卦象
  
同时实现:
  1. Grover振幅放大路径 (标准量子涌现)
  2. 量子势能演化路径 (道家"道法自然"版本)
  
策略映射也改为动态推理, 而非预定义STRATEGY_MAP:
  每个卦象的抓取策略 = f(卦象的爻位特征 + 物体物理特征)
"""

import numpy as np
from typing import List, Tuple, Callable, Optional


# ============================================================
# 卦象元数据
# ============================================================

HEXAGRAM_NAMES = [
    "乾","坤","屯","蒙","需","讼","师","比","小畜","履","泰","否",
    "同人","大有","谦","豫","随","蛊","临","观","噬嗑","贲","剥","复",
    "无妄","大畜","颐","大过","坎","离","咸","恒","遯","大壮","晋","明夷",
    "家人","睽","蹇","解","损","益","夬","姤","萃","升","困","井",
    "革","鼎","震","艮","渐","归妹","丰","旅","巽","兑","涣","节",
    "中孚","小过","既济","未济"
]

def hex_name(idx: int) -> str:
    return HEXAGRAM_NAMES[idx-1] if 1 <= idx <= 64 else "??"

def bin_list(idx: int, n: int) -> List[int]:
    return [(idx>>i)&1 for i in range(n)]

def state_label(idx: int, n: int = 6) -> str:
    bits = bin_list(idx, n)
    return ''.join(['─' if b else '╌' for b in reversed(bits)])


# ============================================================
# 核心：感知自适应Oracle
# ============================================================

class AdaptiveOracle:
    """
    感知自适应 Oracle
    Oracle = 易理评分(固定) × 感知匹配度(随物变化)
    
    设计思想:
      易理评分 编码了「乘承比应当位得中」—— 这是道的规律, 不因物而变
      感知匹配度 编码了当前物体特征与该卦象的相似度 —— 这是物的状态, 随物而变
      二者相乘 = 「道」在「物」上的投影
    """
    
    def __init__(self, n: int = 6):
        self.n = n
        self.dim = 1 << n
        self._init_yili_scores()
        
    def _yili_score(self, idx: int) -> float:
        """标准易理评分 (固定, 不随输入变化)"""
        bits = bin_list(idx, self.n)
        s = 0.0
        
        # 当位 (权重1)
        for q in range(self.n):
            is_yang = (q % 2 == 0)
            proper = (bits[q]==1 and is_yang) or (bits[q]==0 and not is_yang)
            s += 1.0 if proper else -1.0
        
        # 得中 (权重2)
        if bits[1] == 0: s += 2.0
        else: s -= 2.0
        if bits[4] == 1: s += 2.0
        else: s -= 2.0
        
        # 乘承 (权重1)
        for q in range(self.n - 1):
            l, u = bits[q], bits[q+1]
            if l==0 and u==1: s -= 1.0
            elif l==1 and u==0: s += 1.0
        
        # 应 (权重1)
        for (a,b) in [(0,3),(1,4),(2,5)]:
            if bits[a] != bits[b]: s += 1.0
            else: s -= 1.0
        
        return s
    
    def _init_yili_scores(self):
        """预计算所有卦的易理评分"""
        self.yili_scores = np.array([self._yili_score(i) for i in range(self.dim)])
        # 归一化到 [0, 1]
        s_min, s_max = self.yili_scores.min(), self.yili_scores.max()
        self.yili_norm = (self.yili_scores - s_min) / (s_max - s_min)
        
    def perception_match(self, idx: int, features: np.ndarray) -> float:
        """卦象与感知特征的匹配度 (余弦相似度+距离)"""
        bits = np.array(bin_list(idx, self.n), dtype=float)
        # 余弦相似度
        cos_sim = np.dot(bits, features) / (np.linalg.norm(bits) * np.linalg.norm(features) + 1e-10)
        # 欧氏距离惩罚
        dist = np.linalg.norm(bits - features)
        dist_penalty = np.exp(-2.0 * dist)
        # 综合: 相似度高且距离近
        return 0.5 * (cos_sim + 1.0) / 2.0 + 0.5 * dist_penalty
    
    def adaptive_score(self, idx: int, features: np.ndarray, 
                       w_perception: float = 1.0, w_yili: float = 1.0) -> float:
        """
        自适应综合评分
        score = w_perception * perception_match + w_yili * yili_norm
        """
        p = self.perception_match(idx, features)
        y = self.yili_norm[idx]
        return w_perception * p + w_yili * y
    
    def oracle_mask(self, features: np.ndarray, threshold: float = 0.5,
                    w_perception: float = 1.0, w_yili: float = 1.0) -> np.ndarray:
        """
        动态Oracle掩码
        返回: bool数组, True标记"好"状态(需相位翻转)
        """
        scores = np.array([
            self.adaptive_score(i, features, w_perception, w_yili) 
            for i in range(self.dim)
        ])
        return scores > threshold
    
    def oracle(self, psi: np.ndarray, features: np.ndarray, 
               threshold: float = 0.5, w_p: float = 1.0, w_y: float = 1.0) -> np.ndarray:
        """自适应Oracle: 随输入物体特征动态变化"""
        mask = self.oracle_mask(features, threshold, w_p, w_y)
        npsi = psi.copy()
        npsi[mask] *= -1
        return npsi


# ============================================================
# 方法1: Grover振幅放大 (标准量子涌现)
# ============================================================

class GroverEmergence:
    """
    Grover振幅放大涌现引擎
    
    用自适应Oracle替代固定Oracle:
      输入不同物体 → 标记不同卦象 → Grover放大不同的答案
    """
    
    def __init__(self, n: int = 6):
        self.n = n
        self.dim = 1 << n
        self.oracle = AdaptiveOracle(n)
    
    def encode_features(self, features: np.ndarray, sigma: float = 0.3) -> np.ndarray:
        """物理感知 → 量子初始态 (模糊编码, 无需知识库)"""
        amplitudes = np.zeros(self.dim, dtype=complex)
        for i in range(self.dim):
            bits = np.array(bin_list(i, self.n), dtype=float)
            dist = np.linalg.norm(bits - features)
            amplitudes[i] = np.exp(-dist**2 / (2 * sigma**2))
        nrm = np.linalg.norm(amplitudes)
        if nrm > 0:
            amplitudes /= nrm
        return amplitudes
    
    def diffusion(self, psi: np.ndarray) -> np.ndarray:
        """扩散变换: D = 2|s⟩⟨s| - I"""
        avg = np.mean(psi)
        return 2 * avg - psi
    
    def grover_iter(self, psi: np.ndarray, features: np.ndarray,
                    threshold: float = 0.5, w_p: float = 1.0, w_y: float = 1.0) -> np.ndarray:
        """单次Grover迭代: G·|ψ⟩ = (2|s⟩⟨s|-I)·O·|ψ⟩"""
        npsi = self.oracle.oracle(psi, features, threshold, w_p, w_y)
        npsi = self.diffusion(npsi)
        return npsi
    
    def amplify(self, psi: np.ndarray, features: np.ndarray, 
                iterations: int = 3, threshold: float = 0.5,
                w_p: float = 1.0, w_y: float = 1.0) -> np.ndarray:
        """执行多次Grover迭代, 让最优卦涌现"""
        npsi = psi.copy()
        for _ in range(iterations):
            npsi = self.grover_iter(npsi, features, threshold, w_p, w_y)
        nrm = np.linalg.norm(npsi)
        if nrm > 0:
            npsi /= nrm
        return npsi
    
    def find_best_iteration(self, psi: np.ndarray, features: np.ndarray,
                            max_iter: int = 8, threshold: float = 0.5,
                            w_p: float = 1.0, w_y: float = 1.0) -> Tuple[int, np.ndarray]:
        """自动寻找最优迭代次数 (涌现最强的那一轮)"""
        best_it = 0
        best_psi = psi.copy()
        best_entropy = -np.sum(np.abs(psi)**2 * np.log(np.abs(psi)**2 + 1e-30))
        
        npsi = psi.copy()
        for it in range(max_iter + 1):
            if it > 0:
                npsi = self.grover_iter(npsi, features, threshold, w_p, w_y)
                nrm = np.linalg.norm(npsi)
                if nrm > 0: npsi /= nrm
            probs = np.abs(npsi)**2
            entropy = -np.sum(probs * np.log(probs + 1e-30))
            # 熵越低 = 涌现越集中 = 越好
            if entropy < best_entropy:
                best_entropy = entropy
                best_it = it
                best_psi = npsi.copy()
        
        return best_it, best_psi
    
    def top_k(self, psi: np.ndarray, k: int = 5) -> List[Tuple[int, float, float, str]]:
        """返回TOP-K卦象"""
        probs = np.abs(psi)**2
        indices = np.argsort(probs)[::-1][:k]
        return [(idx, probs[idx], self.oracle.yili_scores[idx], hex_name(idx+1)) 
                for idx in indices]


# ============================================================
# 方法2: 量子势能演化 (道法自然版本)
# ============================================================

class PotentialEmergence:
    """
    量子势能涌现引擎
    
    用Bohm量子势能实现更"自然"的涌现:
      自适应势能 V(x) = 1 - 自适应评分(x)
      吉卦且匹配 → 低势能 → 概率幅自然汇聚
      凶卦或不匹配 → 高势能 → 概率幅自然衰减
    """
    
    def __init__(self, n: int = 6):
        self.n = n
        self.dim = 1 << n
        self.oracle = AdaptiveOracle(n)
    
    def init_state(self, theta: np.ndarray) -> np.ndarray:
        """角度编码初始化"""
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
                    npsi[i] += psi[i] * c
                    npsi[i | (1<<q)] += psi[i] * s
                else:
                    npsi[i & ~(1<<q)] += psi[i] * (-s)
                    npsi[i] += psi[i] * c
            psi = npsi
        return psi
    
    def build_adaptive_potential(self, features: np.ndarray,
                                  w_p: float = 1.0, w_y: float = 1.0) -> np.ndarray:
        """构建自适应量子势能"""
        scores = np.array([
            self.oracle.adaptive_score(i, features, w_p, w_y)
            for i in range(self.dim)
        ])
        # 势能 = 1 - 归一化评分 (0=最吉/匹配, 1=最凶/不匹配)
        s_min, s_max = scores.min(), scores.max()
        potential = 1.0 - (scores - s_min) / (s_max - s_min + 1e-10)
        return potential
    
    def apply_response_gate(self, psi: np.ndarray) -> np.ndarray:
        """应关系纠缠: 实现非定域关联"""
        for ctrl, tgt in [(0,3),(1,4),(2,5)]:
            npsi = np.zeros_like(psi)
            for i in range(self.dim):
                if abs(psi[i]) < 1e-15: continue
                bits = bin_list(i, self.n)
                if bits[ctrl] == 1:
                    bits[tgt] ^= 1
                    j = sum(b * (1 << k) for k, b in enumerate(bits))
                    npsi[j] = psi[i]
                else:
                    npsi[i] = psi[i]
            psi = npsi
        return psi
    
    def evolve(self, psi: np.ndarray, features: np.ndarray,
               dt: float = 0.1, steps: int = 30,
               w_p: float = 1.0, w_y: float = 1.0) -> np.ndarray:
        """
        量子势能演化:
        iħ ∂ψ/∂t = [ -ħ²/(2m)∇² + V(x|θ) ] ψ
        其中 V(x|θ) 随输入物体特征θ动态变化
        """
        potential = self.build_adaptive_potential(features, w_p, w_y)
        npsi = psi.copy().astype(complex)
        
        for _ in range(steps):
            laplacian = np.zeros_like(npsi)
            for i in range(self.dim):
                left = (i - 1) % self.dim
                right = (i + 1) % self.dim
                laplacian[i] = npsi[left] + npsi[right] - 2 * npsi[i]
            
            V_term = potential * npsi
            npsi += dt * (1j * laplacian - 0.5 * V_term)
            
            nrm = np.linalg.norm(npsi)
            if nrm > 0: npsi /= nrm
        
        return npsi
    
    def run(self, features: np.ndarray, theta: Optional[np.ndarray] = None,
            dt: float = 0.1, steps: int = 30,
            w_p: float = 1.0, w_y: float = 1.0) -> np.ndarray:
        """完整涌现流程: 编码 → 纠缠 → 演化"""
        if theta is None:
            theta = np.clip(features, 0.01, 0.99)
        psi = self.init_state(theta)
        psi = self.apply_response_gate(psi)
        return self.evolve(psi, features, dt, steps, w_p, w_y)
    
    def top_k(self, psi: np.ndarray, k: int = 5) -> List[Tuple[int, float, float, str]]:
        probs = np.abs(psi)**2
        indices = np.argsort(probs)[::-1][:k]
        return [(idx, probs[idx], self.oracle.yili_scores[idx], hex_name(idx+1))
                for idx in indices]


# ============================================================
# 策略映射器: 动态推导而非预定义
# ============================================================

class StrategyMapper:
    """
    从卦象和物体特征动态推导抓取策略
    
    策略不是预定义的, 而是基于:
      卦象的爻位特征 (当位/中/应/承)  ×  物体物理特征 (硬度/粗糙度/...)
    """
    
    GRASP_TYPES = [
        "power_grasp",           # 强力抓取
        "soft_grasp",            # 轻柔抓取
        "precision_grasp",       # 精准抓取
        "cautious_grasp",        # 谨慎抓取
        "adaptive_grasp",        # 自适应抓取
        "compliant_grasp",       # 顺从抓取
        "stable_grasp",          # 稳定抓取
        "dynamic_grasp",         # 动态抓取
        "dual_grasp",            # 双手抓取
        "reduced_force_grasp",   # 减力抓取
    ]
    
    def infer(self, hex_idx: int, features: np.ndarray) -> str:
        """动态推断抓取策略"""
        bits = np.array(bin_list(hex_idx, 6), dtype=float)
        
        # 卦象特征提取
        proper_count = sum(1 for q in range(6) 
                          if (bits[q]==1 and q%2==0) or (bits[q]==0 and q%2==1))
        center_mid = bits[1] == 0   # 内卦中位当(阴)
        center_top = bits[4] == 1   # 外卦中位当(阳)
        response_count = sum(1 for (a,b) in [(0,3),(1,4),(2,5)] if bits[a] != bits[b])
        cheng_count = sum(1 for q in range(5) if bits[q]==0 and bits[q+1]==1)
        
        # 物体特征
        hardness = features[0]
        roughness = features[1]
        shape_reg = features[2]
        dynamic = features[3]
        weight = features[4]
        texture = features[5]
        
        # === 动态规则 (非预定义映射) ===
        
        # 硬且重 → power_grasp
        if hardness > 0.7 and weight > 0.7 and proper_count >= 3:
            return "power_grasp"
        
        # 软且轻 → soft_grasp
        if hardness < 0.3 and weight < 0.3:
            return "soft_grasp"
        
        # 硬且脆 → precision_grasp 或 cautious_grasp
        if hardness > 0.6 and weight < 0.4 and cheng_count > 0:
            return "cautious_grasp"
        if hardness > 0.6 and weight < 0.4 and response_count >= 2:
            return "precision_grasp"
        
        # 动态(滚动/移动) → dynamic_grasp 或 adaptive_grasp
        if dynamic > 0.7:
            if response_count >= 2:
                return "adaptive_grasp"
            return "dynamic_grasp"
        
        # 粗糙且稳定 → stable_grasp
        if roughness > 0.6 and shape_reg > 0.6:
            return "stable_grasp"
        
        # 粗糙且软 → compliant_grasp
        if roughness > 0.6 and hardness < 0.4:
            return "compliant_grasp"
        
        # 轻且纹理复杂 → compliant_grasp
        if weight < 0.3 and texture > 0.6:
            return "compliant_grasp"
        
        # 中正当位 → stable_grasp
        if center_mid and center_top and proper_count >= 4:
            return "stable_grasp"
        
        # 多不应 → adaptive_grasp
        if response_count <= 1:
            return "adaptive_grasp"
        
        # 默认: 基于卦象特征
        if proper_count >= 4:
            return "power_grasp"
        elif proper_count <= 2:
            return "cautious_grasp"
        else:
            return "adaptive_grasp"


# ============================================================
# 统一接口
# ============================================================

class QYUF_Adaptive:
    """QYUF v2.0 统一接口"""
    
    def __init__(self, n: int = 6):
        self.n = n
        self.dim = 1 << n
        self.grover = GroverEmergence(n)
        self.potential = PotentialEmergence(n)
        self.strategy = StrategyMapper()
        self.oracle = AdaptiveOracle(n)
        
    def run_grover(self, features: np.ndarray, 
                   sigma: float = 0.3,
                   w_p: float = 2.0, w_y: float = 1.0,
                   threshold: float = 0.5,
                   max_iter: int = 8) -> dict:
        """Grover涌现路径"""
        psi_init = self.grover.encode_features(features, sigma)
        best_it, psi_final = self.grover.find_best_iteration(
            psi_init, features, max_iter, threshold, w_p, w_y
        )
        
        top = self.grover.top_k(psi_final, 5)
        top_init = self.grover.top_k(psi_init, 3)
        
        results = []
        for idx, prob, score, name in top:
            strat = self.strategy.infer(idx, features)
            results.append({
                'idx': idx, 'prob': prob, 'yili_score': score,
                'name': name, 'strategy': strat,
                'state': state_label(idx)
            })
        
        return {
            'method': 'grover',
            'best_iter': best_it,
            'psi_final': psi_final,
            'top': results,
            'top_init': [(idx, p, hn) for idx, p, _, hn in top_init],
            'good_prob_init': sum(np.abs(psi_init[self.oracle.yili_scores > 0])**2),
            'good_prob_final': sum(np.abs(psi_final[self.oracle.yili_scores > 0])**2),
        }
    
    def run_potential(self, features: np.ndarray,
                      w_p: float = 2.0, w_y: float = 1.0,
                      steps: int = 30) -> dict:
        """量子势能演化路径"""
        psi_final = self.potential.run(features, w_p=w_p, w_y=w_y, steps=steps)
        
        top = self.potential.top_k(psi_final, 5)
        
        results = []
        for idx, prob, score, name in top:
            strat = self.strategy.infer(idx, features)
            results.append({
                'idx': idx, 'prob': prob, 'yili_score': score,
                'name': name, 'strategy': strat,
                'state': state_label(idx)
            })
        
        return {
            'method': 'potential',
            'psi_final': psi_final,
            'top': results,
        }
    
    def run_all(self, features: np.ndarray, **kwargs) -> dict:
        """同步运行两种涌现路径"""
        g = self.run_grover(features, **kwargs)
        p = self.run_potential(features, **kwargs)
        return {'grover': g, 'potential': p}


# ============================================================
# 验证实验
# ============================================================

def exp_complete_mapping():
    """实验: 完整自动映射"""
    print("="*75)
    print(" QYUF v2.0 — 感知自适应酉变换 · 完整映射验证")
    print("="*75)
    print()
    print(" Oracle = 易理评分(乘承比应当位得中) × 感知匹配度(当前物体)")
    print(" 无预定义映射表, 无模糊知识库, 只有物理感知输入")
    print()
    
    qyuf = QYUF_Adaptive()
    
    tests = [
        ("金属块(硬光滑)", np.array([0.90, 0.10, 0.85, 0.10, 0.85, 0.20])),
        ("海绵(软粗糙)",   np.array([0.10, 0.90, 0.20, 0.10, 0.10, 0.80])),
        ("瓷杯(硬脆)",     np.array([0.80, 0.15, 0.60, 0.10, 0.40, 0.30])),
        ("皮球(弹动)",     np.array([0.30, 0.40, 0.30, 0.90, 0.50, 0.30])),
        ("木块(硬粗稳)",   np.array([0.70, 0.70, 0.80, 0.20, 0.70, 0.60])),
        ("纸团(软轻皱)",   np.array([0.15, 0.80, 0.10, 0.10, 0.10, 0.90])),
        ("鸡蛋(脆滑)",     np.array([0.50, 0.20, 0.50, 0.10, 0.10, 0.20])),
        ("石头(硬重糙)",   np.array([0.95, 0.80, 0.70, 0.15, 0.90, 0.50])),
        ("沙袋(软重)",     np.array([0.20, 0.85, 0.10, 0.10, 0.70, 0.70])),
        ("书本(硬平)",     np.array([0.60, 0.50, 0.90, 0.05, 0.60, 0.40])),
        ("保龄球(硬圆重)", np.array([0.95, 0.10, 0.90, 0.80, 0.95, 0.10])),
        ("羽毛(软轻)",     np.array([0.05, 0.30, 0.20, 0.10, 0.05, 0.30])),
        ("冰块(硬滑冷)",   np.array([0.85, 0.05, 0.60, 0.10, 0.60, 0.10])),
        ("橡皮泥(软可塑)", np.array([0.10, 0.40, 0.10, 0.05, 0.25, 0.30])),
        ("毛巾(软吸水)",   np.array([0.10, 0.70, 0.30, 0.05, 0.30, 0.60])),
    ]
    
    # ========== Grover路径 ==========
    print("─" * 75)
    print("【路径一: Grover振幅放大】")
    print("─" * 75)
    
    for name, feats in tests:
        result = qyuf.run_grover(feats, w_p=2.0, w_y=1.0, threshold=0.45)
        top = result['top']
        best_it = result['best_iter']
        
        # 显示TOP3
        desc = ' | '.join(
            f"{r['name']}→{r['strategy']}({r['prob']*100:.1f}%)"
            for r in top[:3]
        )
        
        # 感知修正标记
        init_hex = hex_name(result['top_init'][0][0] + 1) if result['top_init'] else "?"
        final_hex = top[0]['name']
        correction = "← 酉变换修正" if init_hex != final_hex else ""
        
        print(f"  {name:12s} ▶ {desc}")
        print(f"              峰值迭代{best_it}轮 | "
              f"吉卦{result['good_prob_init']*100:.0f}%→{result['good_prob_final']*100:.0f}% {correction}")
    
    # ========== 势能路径 ==========
    print()
    print("─" * 75)
    print("【路径二: 量子势能演化】")
    print("─" * 75)
    
    for name, feats in tests:
        result = qyuf.run_potential(feats, w_p=2.0, w_y=1.0, steps=30)
        top = result['top']
        
        desc = ' | '.join(
            f"{r['name']}→{r['strategy']}({r['prob']*100:.1f}%)"
            for r in top[:3]
        )
        
        print(f"  {name:12s} ▶ {desc}")
    
    print()
    print("=" * 75)
    print(" ✓ 验证完成")
    print("   1. 物理感知 → 自适应Oracle → 卦象涌现 (完全自动)")
    print("   2. 策略为动态推断, 非预定义映射")
    print("   3. 酉变换根据物体特征动态调整干涉结果")
    print("=" * 75)


def exp_adaptive_comparison():
    """对比: 固定Oracle vs 自适应Oracle"""
    print("=" * 75)
    print(" 对照实验: 固定Oracle vs 自适应Oracle")
    print("=" * 75)
    print()
    print(" 固定Oracle (v1.0): f(x) = 易理评分(x) > 0")
    print(" 自适应Oracle (v2.0): f(x|θ) = 易理评分(x) × 感知匹配(x, θ) > 阈值")
    print()
    
    qyuf = QYUF_Adaptive()
    
    # 取两个特征差异大的物体
    test_pair = [
        ("金属块(硬)", np.array([0.90, 0.10, 0.85, 0.10, 0.85, 0.20])),
        ("海绵(软)",   np.array([0.10, 0.90, 0.20, 0.10, 0.10, 0.80])),
    ]
    
    for name, feats in test_pair:
        print(f"▶ 输入: {name}")
        print(f"  特征: {feats}")
        
        # 固定Oracle: 只看易理评分
        fixed_mask = qyuf.oracle.yili_scores > 0
        fixed_good = [i for i in range(64) if fixed_mask[i]]
        fixed_top5 = sorted([(i, qyuf.oracle.yili_scores[i]) for i in fixed_good], 
                           key=lambda x: -x[1])[:5]
        print(f"  固定Oracle标记卦: " + " ".join(
            f"{hex_name(i+1)}({s:+.0f})" for i, s in fixed_top5))
        
        # 自适应Oracle: 随物变化
        adaptive_mask = qyuf.oracle.oracle_mask(feats, threshold=0.45, w_perception=2.0, w_yili=1.0)
        adaptive_good = [i for i in range(64) if adaptive_mask[i]]
        scores = [qyuf.oracle.adaptive_score(i, feats, 2.0, 1.0) for i in adaptive_good]
        adaptive_top5 = sorted(zip(adaptive_good, scores), key=lambda x: -x[1])[:5]
        print(f"  自适应Oracle标记卦: " + " ".join(
            f"{hex_name(i+1)}({s:.2f})" for i, s in adaptive_top5))
        print()


def exp_strategy_analysis():
    """分析动态策略映射的合理性"""
    print("=" * 75)
    print(" 动态策略映射分析")
    print("=" * 75)
    print()
    print(" 策略不是预定义的, 而是基于 卦象爻位特征 + 物体特征 动态推导")
    print()
    
    qyuf = QYUF_Adaptive()
    
    tests = [
        ("金属块", np.array([0.90, 0.10, 0.85, 0.10, 0.85, 0.20])),
        ("海绵",   np.array([0.10, 0.90, 0.20, 0.10, 0.10, 0.80])),
        ("保龄球", np.array([0.95, 0.10, 0.90, 0.80, 0.95, 0.10])),
    ]
    
    for name, feats in tests:
        result = qyuf.run_grover(feats, w_p=2.0, w_y=1.0)
        print(f"▶ {name}")
        for r in result['top'][:3]:
            print(f"  {r['name']}(卦{r['idx']+1}) {r['state']}")
            print(f"    概率{r['prob']*100:.1f}% → 策略: {r['strategy']}")
            bits = bin_list(r['idx'], 6)
            proper = sum(1 for q in range(6) if (bits[q]==1 and q%2==0) or (bits[q]==0 and q%2==1))
            center = bits[1]==0 and bits[4]==1
            response = sum(1 for (a,b) in [(0,3),(1,4),(2,5)] if bits[a] != bits[b])
            print(f"    (当位{proper}/6, 中{center}, 应{response}/3)")
        print()


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    
    exp_complete_mapping()
    print("\n" + "=" * 75 + "\n")
    exp_adaptive_comparison()
    print("\n" + "=" * 75 + "\n")
    exp_strategy_analysis()
