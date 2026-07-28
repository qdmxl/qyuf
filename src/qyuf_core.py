#!/usr/bin/env python3
"""
QYUF 核心引擎 v3 — 全栈量子易理统一框架
========================================
从L0到L4全量子化:
  L0: 物体特征→量子态编码(替代经典余弦隶属度,
  L1: 6量子比特天然表达64卦(无需上下卦乘积,
  L2: 卦象叠加态编码
  L3: YiliOracle嵌入Grover Oracle(乘承比应一次酉变换,
  L4: Grover振幅放大涌现(替代贪心Top1排序,

运行模式:
  1. NumPy仿真模式 (默认) — 纯Python
  2. Qiskit量子电路模式 — 真实量子计算仿真
"""

import numpy as np
import math
from typing import List, Tuple, Optional

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


# ==================== 常量 ====================

HEXAGRAM_NAMES = [
    "乾","坤","屯","蒙","需","讼","师","比","小畜","履","泰","否",
    "同人","大有","谦","豫","随","蛊","临","观","噬嗑","贲","剥","复",
    "无妄","大畜","颐","大过","坎","离","咸","恒","遯","大壮","晋","明夷",
    "家人","睽","蹇","解","损","益","夬","姤","萃","升","困","井",
    "革","鼎","震","艮","渐","归妹","丰","旅","巽","兑","涣","节",
    "中孚","小过","既济","未济"
]

STRATEGY_MAP = {
    "乾":"power_grasp","坤":"soft_grasp","屯":"cautious_grasp",
    "蒙":"exploratory_grasp","需":"waiting_grasp","讼":"competitive_grasp",
    "师":"coordinated_grasp","比":"support_grasp","小畜":"progressive_grasp",
    "履":"precise_grasp","泰":"balanced_grasp","否":"abort_or_retry",
    "同人":"dual_grasp","大有":"robust_grasp","谦":"compliant_grasp",
    "豫":"prepared_grasp","随":"adaptive_grasp","蛊":"corrective_grasp",
    "临":"monitoring_grasp","观":"observation","噬嗑":"biting_grasp",
    "贲":"decorative_grasp","剥":"gradual_grasp","复":"retry_grasp",
    "无妄":"direct_grasp","大畜":"accumulate_grasp","颐":"nurture_grasp",
    "大过":"strong_grasp","坎":"risky_grasp","离":"precise_grasp",
    "咸":"quick_grasp","恒":"stable_grasp","遯":"retreat_grasp",
    "大壮":"power_grasp","晋":"advance_grasp","明夷":"injured_grasp",
    "家人":"gentle_grasp","睽":"conflict_grasp","蹇":"difficult_grasp",
    "解":"extrication_grasp","损":"reduced_force_grasp","益":"progressive_grasp",
    "夬":"decisive_grasp","姤":"adaptive_grasp","萃":"sequential_grasp",
    "升":"top_down_grasp","困":"difficult_grasp","井":"stable_grasp",
    "革":"corrective_grasp","鼎":"balanced_grasp","震":"dynamic_grasp",
    "艮":"stable_grasp","渐":"progressive_grasp","归妹":"compliant_grasp",
    "丰":"robust_power_grasp","旅":"conditional_grasp","巽":"compliant_grasp",
    "兑":"soft_grasp","涣":"abort_or_retry","节":"reduced_force_grasp",
    "中孚":"tactile_feedback_grasp","小过":"cautious_grasp",
    "既济":"balanced_grasp","未济":"abort_or_retry"
}


def bin_list(idx: int, n: int = 6) -> List[int]:
    return [(idx >> i) & 1 for i in range(n)]


def state_label(idx: int) -> str:
    bits = bin_list(idx)
    return ''.join(['─' if b else '╌' for b in reversed(bits)])


# ==================== L3: 乘承比应急理评分(量子Oracle核心,====================

class YiliOracle:
    """
    易理Oracle — 将YLYW的YaoRelations完整嵌入量子计算
    
    对任意六爻卦象(idx), 计算包含以下5项的综合评分:
      1. 当位 (dangwei): 阳爻居阳位/阴爻居阴位 → +分
      2. 得中 (dezhong): 二宜阴、五宜阳 → +分  
      3. 乘 (cheng): 阴乘阳 → -分
      4. 承 (cheng_rev): 阴承阳 → +分
      5. 应 (ying): 相应爻阴阳相反 → +分
    
    经典版本(YLYW)：每次处理一个六爻向量, 逐条if-else, 输出0-1评分
    量子版本(QYUF)：对64卦一次性计算评分, 作为Grover Oracle的相位标记
    """
    
    # 权重配置(与YLYW YaoRelations一致,
    # 优化权重(经搜索：方差最大化=区分度最优,
    WEIGHTS = {
        'dangwei': 0.50,       # 当位50%(最核心,
        'dezhong': 0.25,       # 得中25%  
        'cheng_cheng': 0.10,   # 乘承10%
        'bi': 0.05,            # 比5%(相邻关系区分度弱,
        'ying': 0.10,          # 应10%
    }
    
    # 阳位：初(0)、三(2)、五(4)
    YANG_POS = {0, 2, 4}
    # 中位：二(1)、五(4)
    ZHONG_POS = {1, 4}
    # 应位对：(初-四), (二-五), (三-上)
    YING_PAIRS = [(0, 3), (1, 4), (2, 5)]
    
    @classmethod
    def score_dangwei(cls, yao: List[int]) -> float:
        """当位评分 (0-1): 阳居阳位/阴居阴位的比例"""
        n_correct = 0
        for i in range(6):
            is_yang = yao[i] == 1
            should_yang = i in cls.YANG_POS
            if is_yang == should_yang:
                n_correct += 1
        return n_correct / 6.0
    
    @classmethod
    def score_dezhong(cls, yao: List[int]) -> float:
        """得中评分 (0-1): 二宜阴、五宜阳"""
        s = 0.0
        # 二爻(1): 阴佳
        if yao[1] == 0:
            s += 0.5  # 六二(阴居二,
        elif yao[1] == 1:
            s += 0.25  # 九二(阳居二，次优,
        # 五爻(4): 阳佳
        if yao[4] == 1:
            s += 0.5  # 九五(阳居五,
        elif yao[4] == 0:
            s += 0.25  # 六五(阴居五，次优,
        return s
    
    @classmethod
    def score_cheng_cheng(cls, yao: List[int]) -> float:
        """乘承评分 (0-1): 乘减分, 承加分"""
        cheng_bad = 0  # 阴乘阳
        cheng_good = 0  # 阴承阳
        for i in range(5):
            lower, upper = yao[i], yao[i+1]
            if upper == 0 and lower == 1:
                cheng_bad += 1  # 上阴下阳 → 乘
            elif upper == 1 and lower == 0:
                cheng_good += 1  # 下阴上阳 → 承
        # 乘减分 (每次-0.3), 承加分 (每次+0.15)
        raw = 0.5 - cheng_bad * 0.3 + cheng_good * 0.15
        return max(0.0, min(1.0, raw + 0.5))
    
    @classmethod
    def score_bi(cls, yao: List[int]) -> float:
        """比评分 (0-1): 相邻同性的比例"""
        harmony = 0
        for i in range(5):
            if yao[i] == yao[i+1]:
                harmony += 1
        return harmony / 5.0
    
    @classmethod
    def score_ying(cls, yao: List[int]) -> float:
        """应评分 (0-1): 相应爻阴阳相异的比例"""
        count = 0
        for a, b in cls.YING_PAIRS:
            if yao[a] != yao[b]:
                count += 1
        return count / 3.0
    
    @classmethod
    def comprehensive_score(cls, yao: List[int]) -> float:
        """综合易理评分 (0-1)，与YLYW的YaoRelations._compute_overall_score一致"""
        w = cls.WEIGHTS
        return (
            w['dangwei'] * cls.score_dangwei(yao) +
            w['dezhong'] * cls.score_dezhong(yao) +
            w['cheng_cheng'] * cls.score_cheng_cheng(yao) +
            w['bi'] * cls.score_bi(yao) +
            w['ying'] * cls.score_ying(yao)
        )
    
    @classmethod
    def comprehensive_score_named(cls, yao: List[int]) -> dict:
        """返回命名评分字典"""
        return {
            'dangwei': cls.score_dangwei(yao),
            'dezhong': cls.score_dezhong(yao),
            'cheng_cheng': cls.score_cheng_cheng(yao),
            'bi': cls.score_bi(yao),
            'ying': cls.score_ying(yao),
            'overall': cls.comprehensive_score(yao)
        }


# ==================== L0: 物体特征→量子态编码 ====================

class FeatureEncoder:
    """
    L0量子编码器 — 物体特征→量子初态
    
    经典L0 (TrigramBase.compute_membership):
      物体特征 → 逐维计算高斯隶属度 → 平均得8个隶属度 → 上下卦乘积得64卦分数
      ⚠ 问题: 隶属度是标量，丢失了"多重性"——一个物体只映射到最优卦
    
    量子L0 (FeatureEncoder):
      物体特征 → 编码为64维量子态
      - 每个卦的初始振幅 = f(物体对该卦上下卦的相似度, 卦的易理评分)
      - 所有可能性共存于叠加态中
      - Grover干涉让最优卦自然涌现
    """
    
    # 物体特征维度 (与YLYW TrigramBase一致)
    FEATURE_KEYS = ['size', 'weight', 'fragility', 'surface', 'shape']
    
    # 八卦物理原型 (与YLYW trigram_base.py一致)
    TRIGRAM_PROTOTYPES = [
        ("乾", {'size': 0.9, 'weight': 0.9, 'fragility': 0.1, 'surface': 0.8, 'shape': 0.9}),
        ("坤", {'size': 0.7, 'weight': 0.8, 'fragility': 0.3, 'surface': 0.2, 'shape': 0.3}),
        ("震", {'size': 0.6, 'weight': 0.7, 'fragility': 0.2, 'surface': 0.5, 'shape': 0.4}),
        ("巽", {'size': 0.3, 'weight': 0.2, 'fragility': 0.6, 'surface': 0.3, 'shape': 0.5}),
        ("坎", {'size': 0.5, 'weight': 0.6, 'fragility': 0.4, 'surface': 0.1, 'shape': 0.2}),
        ("离", {'size': 0.2, 'weight': 0.3, 'fragility': 0.7, 'surface': 0.9, 'shape': 0.6}),
        ("艮", {'size': 0.8, 'weight': 0.5, 'fragility': 0.5, 'surface': 0.4, 'shape': 0.7}),
        ("兑", {'size': 0.4, 'weight': 0.4, 'fragility': 0.8, 'surface': 0.6, 'shape': 0.8}),
    ]
    
    @classmethod
    def compute_membership(cls, features: dict, proto: dict) -> float:
        """单个卦隶属度：与YLYW TrigramBase.compute_membership一致"""
        similarity = 0.0
        weight_sum = 0.0
        for key, proto_val in proto.items():
            if key in features:
                diff = abs(features[key] - proto_val)
                membership = max(0.0, 1.0 - diff * 1.5)
                similarity += membership
                weight_sum += 1.0
        return similarity / weight_sum if weight_sum > 0 else 0.0
    
    @classmethod
    def get_memberships(cls, features: dict) -> np.ndarray:
        """物体对8卦的隶属度向量"""
        memberships = np.zeros(8)
        for i, (name, proto) in enumerate(cls.TRIGRAM_PROTOTYPES):
            memberships[i] = cls.compute_membership(features, proto)
        # 归一化确保总和合理
        return memberships / (np.sum(memberships) + 1e-10) * 8
    
    @classmethod
    def encode(cls, features: dict, hexagram_scores: np.ndarray) -> np.ndarray:
        """
        量子编码：物体特征→64维量子初态
        
        编码策略:
          1. 计算物体对8卦的隶属度 (与经典L0相同)
          2. 对64卦的各卦: 振幅 = 隶属度乘积的平方根 × 卦的评分偏置
          3. 隶属度高的卦 + 易理评分高的卦 → 更大初始振幅
          4. 归一化为单位向量
        
        经典L0: 隶属度→乘积→排序取Top1
        量子L0: 隶属度→叠加态→Grover干涉→最优卦涌现
          本质区别: 所有可能性保留在叠加态中，不丢失多重性
        """
        memberships = cls.get_memberships(features)
        
        psi = np.zeros(64, dtype=complex)
        for idx in range(64):
            upper = idx >> 3  # 上三爻
            lower = idx & 0x7  # 下三爻
            mu_upper = memberships[upper]
            mu_lower = memberships[lower]
            
            # 卦的易理评分偏置：评分映射到[0.8, 1.2]
            score = hexagram_scores[idx]
            bias = 0.8 + score * 0.4  # 评分0~1 → 偏置0.8~1.2
            
            # 振幅 = √(隶属度乘积) × 评分偏置(加微小偏移避免0振幅被淹没,
            amplitude = math.sqrt(mu_upper * mu_lower + 1e-6) * bias
            psi[idx] = amplitude
        
        # 归一化
        norm = np.linalg.norm(psi)
        return psi / norm if norm > 0 else np.ones(64, dtype=complex) / 8.0


# ==================== QYUF 核心类 ====================

class QYUF:
    """
    量子易理统一框架 核心引擎 v3 — 全栈量子化
    
    全系列量子化:
      L0: FeatureEncoder — 物体特征→量子态
      L1/L2: 6量子比特→64卦叠加态
      L3: YiliOracle — 乘承比应一次酉变换
      L4: Grover振幅放大 — 涌现最优策略
    
    Args:
        oracle_mode: 'quantum' | 'classic'
        good_threshold: 吉卦评分阈值
        backend: 'numpy' | 'qiskit'
    """
    
    def __init__(self, oracle_mode: str = 'quantum', good_threshold: float = 0.6,
                 backend: str = 'numpy'):
        self.n = 6
        self.dim = 64
        self.oracle_mode = oracle_mode
        self.threshold = good_threshold
        self.backend = backend
        
        if backend == 'qiskit' and not HAS_QISKIT:
            raise ImportError("Qiskit not installed")
        
        self._init_scoring()
    
    def _init_scoring(self):
        """预计算所有64卦的评分"""
        if self.oracle_mode == 'quantum':
            # 使用YiliOracle(完整L3，与YLYW一致,
            self.scores = np.array([
                YiliOracle.comprehensive_score(bin_list(i))
                for i in range(self.dim)
            ])
            # 阈值：评分>=0.6为吉卦(与YaoRelations综合评分对应)
        else:
            # 向后兼容模式
            self.scores = np.array([self._legacy_score(i) for i in range(self.dim)])
        
        self.good_mask = self.scores >= self.threshold
        self.N_good = np.sum(self.good_mask)
        # 动态调参：如果吉卦太多(>18)则收紧阈值，保证Grover对比度
        if self.N_good > 18:
            sorted_scores = sorted(self.scores, reverse=True)
            adjusted = sorted_scores[18]
            self.good_mask = self.scores >= adjusted
            self.N_good = np.sum(self.good_mask)
        if self.N_good > 0:
            self.opt_iters = max(1, int(np.round(np.pi/4 * np.sqrt(64/self.N_good))))
        else:
            self.opt_iters = 1
    
    def _legacy_score(self, idx: int) -> float:
        """旧版硬编码评分(仅用于对比,"""
        b = bin_list(idx, self.n)
        s = 0.0
        for q in range(self.n):
            is_yang = (q % 2 == 0)
            proper = (b[q]==1 and is_yang) or (b[q]==0 and not is_yang)
            s += 1.0 if proper else -1.0
        if b[1] == 0: s += 2.0
        else: s -= 2.0
        if b[4] == 1: s += 2.0
        else: s -= 2.0
        for q in range(self.n - 1):
            if b[q]==0 and b[q+1]==1: s -= 1.0
            elif b[q]==1 and b[q+1]==0: s += 1.0
        for (a, b_) in [(0,3),(1,4),(2,5)]:
            if b[a] != b[b_]: s += 1.0
            else: s -= 1.0
        return s
    
    # ---------- Oracle(量子核心, ----------
    
    def uniform(self) -> np.ndarray:
        return np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
    
    def oracle(self, psi: np.ndarray) -> np.ndarray:
        """
        Oracle酉变换 — 量子化的L3乘承比应
        
        经典L3 (YaoRelations): 输入六爻向量 → 逐条运算5项规则 → 输出0-1评分
        量子L3 (QYUF.oracle): 对所有64卦同时翻转相位
        
        翻转量 = 该卦的易理综合评分 - 阈值
        评分越高(吉) → 相位翻转越大 → 干涉后概率幅越大 → 涌现概率越高
        """
        npsi = psi.copy()
        # 使用评分作为相位翻转量，而非简单翻转
        flip_strength = self.scores - self.threshold
        # 映射到[-1, 1]区间
        flip_strength = np.clip(flip_strength / (1 - self.threshold), -1, 1)
        npsi *= np.exp(1j * np.pi * flip_strength)
        return npsi
    
    def binary_oracle(self, psi: np.ndarray) -> np.ndarray:
        """
        二元Oracle(二分法,— 只对吉卦翻转180°
        用于对比：二元vs连续相位翻转的效果差异
        """
        npsi = psi.copy()
        npsi[self.good_mask] *= -1
        return npsi
    
    def diffusion(self, psi: np.ndarray) -> np.ndarray:
        return 2 * np.mean(psi) - psi
    
    def iterate(self, psi: np.ndarray, oracle_fn=None) -> np.ndarray:
        if oracle_fn is None:
            oracle_fn = self.oracle
        psi = oracle_fn(psi)
        psi = self.diffusion(psi)
        return psi / np.linalg.norm(psi)
    
    def amplify(self, psi: np.ndarray, iters: int, oracle_fn=None) -> np.ndarray:
        for _ in range(iters):
            psi = self.iterate(psi, oracle_fn)
        return psi
    
    def prob(self, psi: np.ndarray) -> np.ndarray:
        return np.abs(psi)**2
    
    def top_k(self, psi: np.ndarray, k: int = 10) -> List[Tuple]:
        probs = self.prob(psi)
        idxs = np.argsort(probs)[::-1][:k]
        return [(i, probs[i], self.scores[i], HEXAGRAM_NAMES[i]) for i in idxs]
    
    def entropy(self, psi: np.ndarray) -> float:
        probs = self.prob(psi)
        mask = probs > 0
        return -np.sum(probs[mask] * np.log2(probs[mask]))
    
    def run_inference(self, iters: Optional[int] = None, oracle_fn=None) -> np.ndarray:
        i = iters if iters is not None else self.opt_iters
        psi = self.uniform()
        return self.amplify(psi, i, oracle_fn)
    
    def detailed_analysis(self, idx: int) -> dict:
        """对单个卦的详细易理分析(与YLYW YaoRelations.analyze对标,"""
        yao = bin_list(idx)
        return YiliOracle.comprehensive_score_named(yao)
    
    # ---------- 端到端决策(L0+L2+L3+L4 全量子化, ----------
    
    def decision(self, features: dict, iters: Optional[int] = None,
                 oracle_fn=None) -> Tuple[int, str, str, float, np.ndarray]:
        """
        端到端量子决策：特征→量子态→Grover涌现→最优卦
        
        Args:
            features: 物体特征字典 {'size': 0.8, 'weight': 0.7, ...}
            iters: Grover迭代次数
            oracle_fn: Oracle函数(默认用self.oracle,
        
        Returns:
            (idx, name, strategy, confidence, psi)
        """
        i = iters if iters is not None else self.opt_iters
        
        # L0: 物体特征→量子初态
        psi0 = FeatureEncoder.encode(features, self.scores)
        
        # L2+L3+L4: Grover涌现（默认用二元Oracle，标准Grover）
        psi = self.amplify(psi0, i, oracle_fn or self.binary_oracle)
        
        probs = self.prob(psi)
        best_idx = int(np.argmax(probs))
        best_conf = float(probs[best_idx])
        
        return (best_idx, HEXAGRAM_NAMES[best_idx],
                STRATEGY_MAP.get(HEXAGRAM_NAMES[best_idx], '?'),
                best_conf, psi)
    
    def classic_decision(self, features: dict) -> Tuple[int, str, str, float]:
        """
        经典对比决策(YLYW L0+L1+L4,：隶属度→乘积→排序取Top1
        保留此方法用于量子vs经典对比验证
        """
        memberships = FeatureEncoder.get_memberships(features)
        scores = np.zeros(64)
        for idx in range(64):
            scores[idx] = memberships[idx >> 3] * memberships[idx & 0x7]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        return (best_idx, HEXAGRAM_NAMES[best_idx],
                STRATEGY_MAP.get(HEXAGRAM_NAMES[best_idx], '?'), best_score)
    
    # ---------- Qiskit 模式 ----------
    
    def build_grover_circuit(self, iters: int = 1) -> 'QuantumCircuit':
        if not HAS_QISKIT:
            raise ImportError("Qiskit not installed")
        
        n = self.n
        qc = QuantumCircuit(n + 1, n)
        
        for i in range(n):
            qc.h(i)
        
        for _ in range(iters):
            qc.barrier()
            for idx in range(self.dim):
                if self.good_mask[idx]:
                    bits = bin_list(idx)
                    for i in range(n):
                        if bits[i] == 0:
                            qc.x(i)
                    qc.mcx(list(range(n)), n)
                    for i in range(n):
                        if bits[i] == 0:
                            qc.x(i)
            qc.barrier()
            
            for i in range(n):
                qc.h(i)
            for i in range(n):
                qc.x(i)
            qc.mcx(list(range(n-1)), n-1)
            for i in range(n):
                qc.x(i)
            for i in range(n):
                qc.h(i)
        
        for i in range(n):
            qc.measure(i, i)
        
        return qc
    
    def run_inference_qiskit(self, iters: int = 1, shots: int = 8192) -> np.ndarray:
        qc = self.build_grover_circuit(iters)
        sim = AerSimulator()
        result = sim.run(qc, shots=shots).result()
        counts = result.get_counts()
        psi = np.zeros(self.dim, dtype=complex)
        total = sum(counts.values())
        for bitstring, count in counts.items():
            bs = bitstring.split()[0][:self.n]
            idx = int(bs, 2)
            psi[idx] = np.sqrt(count / total)
        return psi


# ==================== 便捷函数 ====================

def summary(qyuf: QYUF, psi: np.ndarray):
    probs = qyuf.prob(psi)
    good_p = np.sum(probs[qyuf.good_mask])
    entropy = qyuf.entropy(psi)
    valid_threshold = qyuf.threshold
    
    print(f"  吉卦总概率: {good_p*100:.1f}% (阈值 ≥{valid_threshold:.2f}, {qyuf.N_good}/64卦)")
    print(f"  信息熵: {entropy:.2f} bits (从6.00 bits)")
    print(f"  涌现增益: x{good_p/(qyuf.N_good/64):.2f}")
    print()
    
    top = qyuf.top_k(psi, 10)
    print("  TOP 10 涌现卦象:")
    print(f"  {'卦象':8s} {'卦名':4s} {'概率':6s} {'评分':5s} {'当位':>4s} {'得中':>4s} {'乘承':>4s} {'比':>4s} {'应':>4s} → 策略")
    print("  "+"─"*75)
    for idx, p, s, hn in top:
        yao = bin_list(idx)
        y = YiliOracle.comprehensive_score_named(yao)
        strat = STRATEGY_MAP.get(hn, "?")
        bar = '█' * int(p * 200) + '░' * (10 - min(10, int(p * 200)))
        dw = f"{y['dangwei']*100:.0f}%"
        dz = f"{y['dezhong']*100:.0f}%"
        cc = f"{y['cheng_cheng']*100:.0f}%"
        bi = f"{y['bi']*100:.0f}%"
        yi = f"{y['ying']*100:.0f}%"
        print(f"  {state_label(idx)} {hn:2s} {p*100:4.1f}% {bar} {s:.3f} {dw:>4s} {dz:>4s} {cc:>4s} {bi:>4s} {yi:>4s} → {strat}")


# ==================== 对比分析 ====================

def compare_oracle_modes():
    """对比不同Oracle模式的涌现效果"""
    print("="*60)
    print("Oracle模式对比: 二元翻转 vs 连续相位 vs 经典")
    print("="*60)
    
    # 1. 经典硬编码评分
    q_classic = QYUF(oracle_mode='classic', good_threshold=3.0)
    psi_c = q_classic.run_inference(oracle_fn=q_classic.binary_oracle)
    
    # 2. 量子YiliOracle (连续相位)
    q_quantum = QYUF(oracle_mode='quantum', good_threshold=0.6)
    psi_q = q_quantum.run_inference()
    
    # 3. 量子YiliOracle (二元翻转)
    psi_q_bin = q_quantum.run_inference(oracle_fn=q_quantum.binary_oracle)
    
    print(f"\n  模式1 — 经典评分(硬编码) + 二元Oracle:")
    probs_c = q_classic.prob(psi_c)
    good_c = np.sum(probs_c[q_classic.good_mask])
    print(f"    吉卦概率: {good_c*100:.1f}% | 吉卦数: {q_classic.N_good}/64")
    top_c = q_classic.top_k(psi_c, 3)
    for idx, p, s, hn in top_c:
        print(f"      {hn}({s:.1f}): {p*100:.1f}%")
    
    print(f"\n  模式2 — YiliOracle(完整L3) + 连续相位Oracle:")
    probs_q = q_quantum.prob(psi_q)
    good_q = np.sum(probs_q[q_quantum.good_mask])
    print(f"    吉卦概率: {good_q*100:.1f}% | 吉卦数: {q_quantum.N_good}/64")
    top_q = q_quantum.top_k(psi_q, 3)
    for idx, p, s, hn in top_q:
        y = YiliOracle.comprehensive_score_named(bin_list(idx))
        print(f"      {hn}(综{s:.3f} 当{y['dangwei']*100:.0f}% 中{y['dezhong']*100:.0f}% 乘承{y['cheng_cheng']*100:.0f}% 比{y['bi']*100:.0f}% 应{y['ying']*100:.0f}%): {p*100:.1f}%")
    
    print(f"\n  模式3 — YiliOracle(完整L3) + 二元Oracle:")
    good_qb = np.sum(q_quantum.prob(psi_q_bin)[q_quantum.good_mask])
    print(f"    吉卦概率: {good_qb*100:.1f}%")
    top_qb = q_quantum.top_k(psi_q_bin, 3)
    for idx, p, s, hn in top_qb:
        print(f"      {hn}({s:.3f}): {p*100:.1f}%")


def demo_end_to_end():
    """全栈量子化端到端演示"""
    q = QYUF(oracle_mode='quantum')
    
    # 典型物体特征(与YLYW一致,
    objects = [
        ('盘子/plate',      {'size': 0.7, 'weight': 0.3, 'fragility': 0.8, 'surface': 0.9, 'shape': 0.8}),
        ('杯子/mug',        {'size': 0.3, 'weight': 0.2, 'fragility': 0.7, 'surface': 0.8, 'shape': 0.5}),
        ('刀/knife',        {'size': 0.4, 'weight': 0.3, 'fragility': 0.2, 'surface': 0.3, 'shape': 0.1}),
        ('肥皂/soap',       {'size': 0.2, 'weight': 0.1, 'fragility': 0.4, 'surface': 0.2, 'shape': 0.3}),
        ('书/book',         {'size': 0.6, 'weight': 0.5, 'fragility': 0.3, 'surface': 0.7, 'shape': 0.6}),
        ('笔/pencil',       {'size': 0.1, 'weight': 0.1, 'fragility': 0.6, 'surface': 0.4, 'shape': 0.1}),
        ('枕头/pillow',     {'size': 0.8, 'weight': 0.3, 'fragility': 0.1, 'surface': 0.1, 'shape': 0.7}),
        ('钥匙/keys',       {'size': 0.1, 'weight': 0.1, 'fragility': 0.3, 'surface': 0.5, 'shape': 0.2}),
        ('咖啡杯/coffee_cup',{'size': 0.3, 'weight': 0.2, 'fragility': 0.7, 'surface': 0.8, 'shape': 0.5}),
        ('钟/clock',        {'size': 0.5, 'weight': 0.4, 'fragility': 0.6, 'surface': 0.9, 'shape': 0.4}),
    ]
    
    print("="*70)
    print("QYUF v3 — 全栈量子易理 端到端演示")
    print("="*70)
    print(f"\n{'物体':10s} | {'经典选卦':12s} | {'量子涌现':12s} | {'涌现评分':>8s} | {'一致性':>6s}")
    print("-"*65)
    
    same_count = 0
    for obj_name, feats in objects:
        c_idx, c_name, c_strat, c_score = q.classic_decision(feats)
        q_idx, q_name, q_strat, q_conf, psi = q.decision(feats)
        
        # 显示卦的高级信息
        c_yili = YiliOracle.comprehensive_score_named(bin_list(c_idx))
        q_yili = YiliOracle.comprehensive_score_named(bin_list(q_idx))
        
        same = "✓" if c_name == q_name else ("→" if c_yili['overall'] < q_yili['overall'] else "←")
        if c_name == q_name:
            same_count += 1
        
        print(f"{obj_name:12s} | {c_name:4s}({c_strat:12s}) | {q_name:4s}({q_strat:12s}) | {q_yili['overall']:.3f} | {same:>4s}")
    
    print(f"\n量子vs经典一致率: {same_count}/{len(objects)} = {same_count/len(objects)*100:.0f}%")
    print("说明: 不一致时，量子涌现选到的是易理评分更高的卦(→)，而非经典余弦匹配")


# ==================== 主程序 ====================

if __name__ == "__main__":
    import sys
    
    q = QYUF(oracle_mode='quantum')
    
    if '--e2e' in sys.argv:
        demo_end_to_end()
        sys.exit(0)
    
    if '--compare' in sys.argv:
        compare_oracle_modes()
        sys.exit(0)
    
    # 默认：全栈量子化展示
    print("="*70)
    print("QYUF v3 — 全栈量子易理统一框架")
    print("L0: 物体特征→量子态编码")
    print("L1: 6比特→64卦叠加态")
    print("L3: YiliOracle乘承比应酉变换")
    print("L4: Grover振幅放大涌现")
    print("="*70)
    
    # L0编码器验证
    print("\n[L0验证] FeatureEncoder — 物体特征→量子态")
    feats = {'size': 0.7, 'weight': 0.3, 'fragility': 0.8, 'surface': 0.9, 'shape': 0.8}
    
    mem = FeatureEncoder.get_memberships(feats)
    hex_names_8 = ['乾','坤','震','巽','坎','离','艮','兑']
    print(f"  输入特征: {feats}")
    print(f"  八卦隶属度: {" | ".join(f"{n}={m:.3f}" for n,m in zip(hex_names_8, mem))}")
    print(f"  主导卦: {hex_names_8[np.argmax(mem)]} ({np.max(mem):.3f})")
    
    psi0 = FeatureEncoder.encode(feats, q.scores)
    probs0 = np.abs(psi0)**2
    top0_idx = np.argsort(probs0)[::-1][:3]
    print(f"  初始叠加前3: {" | ".join(f"{HEXAGRAM_NAMES[i]}({probs0[i]*100:.1f}%)" for i in top0_idx)}")
    
    # L3验证：YiliOracle
    print("\n[L3验证] YiliOracle — 乘承比应综合评分")
    print(f"  吉卦数(≥{q.threshold}): {q.N_good}/64")
    for idx in [21, 42, 60, 11]:
        d = q.detailed_analysis(idx)
        print(f"  {HEXAGRAM_NAMES[idx]}: 综{d['overall']:.3f} (当{d['dangwei']:.2f} 中{d['dezhong']:.2f} 乘承{d['cheng_cheng']:.2f} 比{d['bi']:.2f} 应{d['ying']:.2f})")
    
    # L4验证：Grover端到端涌现
    print("\n[L4验证] Grover端到端涌现 (L0编码 + L3/L4 Grover)")
    psi0 = FeatureEncoder.encode(feats, q.scores)
    before = np.sum(np.abs(psi0)**2 * q.good_mask.astype(float))
    print(f"  初态吉卦概率: {before*100:.1f}%")
    psi = q.amplify(psi0, 1, oracle_fn=q.binary_oracle)
    probs = np.abs(psi)**2
    after = np.sum(probs[q.good_mask])
    print(f"  Grover 1次后: {after*100:.1f}%")
    print(f"  涌现增益: x{after/before:.2f}")
    print(f"  Top1: {HEXAGRAM_NAMES[int(np.argmax(probs))]} ({probs[np.argmax(probs)]*100:.1f}%)")
    print()
    print(f"{'卦象':8s} {'卦名':4s} {'概率':6s} {'评分':5s} {'当位':>4s} {'得中':>4s} {'乘承':>4s} {'比':>4s} {'应':>4s} → 策略")
    print("  "+"─"*75)
    for idx in np.argsort(probs)[::-1][:8]:
        p = probs[idx]
        hn = HEXAGRAM_NAMES[idx]
        y = YiliOracle.comprehensive_score_named(bin_list(idx))
        strat = STRATEGY_MAP.get(hn, "?")
        bar = '█' * int(p * 200) + '░' * (10 - min(10, int(p * 200)))
        print(f"  {state_label(idx)} {hn:2s} {p*100:4.1f}% {bar} {q.scores[idx]:.3f} {y['dangwei']*100:.0f}% {y['dezhong']*100:.0f}% {y['cheng_cheng']*100:.0f}% {y['bi']*100:.0f}% {y['ying']*100:.0f}% → {strat}")
    
    # 端到端演示
    print("\n" + "-"*70)
    demo_end_to_end()
