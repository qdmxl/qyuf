#!/usr/bin/env python3
"""
QYUF v4.0 — 严格"八卦相荡"酉变换模型
============================================
严格按论文《八卦相荡而生六十四卦》实现:

  1. H₃空间(8维)上的酉变换  U_摩  = "刚柔相摩"
     真正的8×8酉矩阵, 编码乘承比应当位得中规则

  2. H₃上的概率幅干涉          = "八卦相荡"
     βⱼ = Σᵢ Uⱼᵢ αᵢ           ← 每卦新幅 = 所有八卦干涉结果

  3. 张量积扩展到H₆(64维)      = "六十四卦涌现"
     U_荡 = U_摩 ⊗ U_摩
     
  4. 特征分形注入感知           = "物感"
     物体特征 → 八卦叠加态（各特征独立匹配不同卦）

论文核心公式:
  |ψ₀⟩ = Σ αᵢ |卦ᵢ⟩          (八卦叠加态)
  |ψ′⟩ = U_摩 |ψ₀⟩            (八卦相荡)
  P(k) = |⟨k|ψ′⟩|²            (测量涌现)

六十四卦版:
  |Ψ₀⟩ = |ψ_上⟩ ⊗ |ψ_下⟩     (上下卦各8维)
  |Ψ′⟩ = (U_摩 ⊗ U_摩) |Ψ₀⟩  (上下各自相荡)
  上卦的"荡"偏向显性特征, 下卦的"荡"偏向隐性特征
"""

import numpy as np
from typing import List, Tuple
from scipy.linalg import expm


# ============================================================
# 八卦元数据
# ============================================================

TRIGRAM_NAMES = ["坤","艮","坎","巽","震","离","兑","乾"]
# 二进制: 坤000, 艮001, 坎010, 巽011, 震100, 离101, 兑110, 乾111

def tname(idx: int) -> str:
    return TRIGRAM_NAMES[idx] if 0 <= idx < 8 else "?"

def tbits(idx: int) -> List[int]:
    return [(idx>>i)&1 for i in range(3)]

def tlabel(idx: int) -> str:
    bits = tbits(idx)
    return ''.join('─' if b else '╌' for b in reversed(bits))

HEX_NAMES = [
    "乾","坤","屯","蒙","需","讼","师","比","小畜","履","泰","否",
    "同人","大有","谦","豫","随","蛊","临","观","噬嗑","贲","剥","复",
    "无妄","大畜","颐","大过","坎","离","咸","恒","遁","大壮","晋","明夷",
    "家人","睽","蹇","解","损","益","夬","姤","萃","升","困","井",
    "革","鼎","震","艮","渐","归妹","丰","旅","巽","兑","涣","节",
    "中孚","小过","既济","未济"
]

def hname(idx: int) -> str:
    return HEX_NAMES[idx] if 0 <= idx < 64 else "?"

def hlabel(idx: int) -> str:
    bits = [(idx>>(5-i))&1 for i in range(6)]
    return ''.join('─' if b else '╌' for b in bits)


# ============================================================
# 特征-八卦映射 (每维独立匹配一对对立八卦)
# ============================================================

FEATURE_DIM = [
    {"name": "刚柔",  "yang": 7, "yin": 0},   # 乾为刚(111), 坤为柔(000)
    {"name": "动静",  "yang": 4, "yin": 3},   # 震为动(100), 巽为入(011)
    {"name": "险丽",  "yang": 2, "yin": 5},   # 坎为险(010), 离为丽(101)
    {"name": "止悦",  "yang": 6, "yin": 1},   # 兑为悦(110), 艮为止(001)
    {"name": "轻重",  "yang": 7, "yin": 0},   # 乾重坤轻(复用)
    {"name": "纹理",  "yang": 4, "yin": 3},   # 震粗巽细(复用)
]


# ============================================================
# 核心: 严格U_摩酉矩阵
# ============================================================

class YiliUnitary:
    """
    易理酉矩阵 U_摩 : H₃ → H₃
    
    论文要求: 这是一个8×8的酉矩阵, 编码乘承比应当位得中.
    
    构造方法:
      U_摩 = exp(-i H_易理 τ)
      
      其中H_易理是8×8的厄密矩阵,
      对角元 = 当位评分(内禀吉凶)
      非对角元 = 卦与卦之间的"乘承比应"关系(干涉强度)
    
    这样U_摩作用在八卦叠加态上,
    好卦的概率幅因干涉而增强, 坏卦减弱。
    """
    
    def __init__(self, tau: float = 0.5):
        self.dim = 8
        self.tau = tau
        self._build_hamiltonian()
        self._build_unitary()
    
    def _trigram_yili_score(self, idx: int) -> float:
        """三爻卦的易理评分 (针对3位)"""
        bits = tbits(idx)
        s = 0.0
        
        # 当位: 初位(bit0)阳, 中位(bit1)阴, 上位(bit2)阳
        for q in range(3):
            is_yang = (q % 2 == 0)
            proper = (bits[q]==1 and is_yang) or (bits[q]==0 and not is_yang)
            s += 1.0 if proper else -1.0
        
        # 乘承
        if bits[0]==0 and bits[1]==1: s -= 0.5
        elif bits[0]==1 and bits[1]==0: s += 0.5
        if bits[1]==0 and bits[2]==1: s -= 0.5
        elif bits[1]==1 and bits[2]==0: s += 0.5
        
        # 中位验证
        if bits[1] == 0: s += 1.0  # 阴居中
        else: s -= 1.0
        
        return s
    
    def _bet_cheng_coupling(self, i: int, j: int) -> float:
        """
        卦i与卦j之间的"乘承比应"耦合强度
        这是论文说的"刚柔相摩"的量子纠缠本质:
          两卦在相同爻位上的阴阳关系决定了它们的干涉强度
        
        如果i和j在每一爻上阴阳相反 → 强耦合(相摩)
        如果i和j完全相同 → 自耦合(对角)
        """
        bi = tbits(i)
        bj = tbits(j)
        
        # 逐爻比较
        coupling = 0.0
        for q in range(3):
            if bi[q] != bj[q]:
                coupling += 0.3  # 异爻相摩 → 正耦合
            else:
                coupling += 0.1  # 同爻相安 → 弱耦合
        
        # 应: 初与上相应
        if bi[0] != bj[2]:
            coupling += 0.2
        
        return coupling
    
    def _build_hamiltonian(self):
        """构建易理哈密顿量 H_易理 (8×8厄密矩阵)"""
        H = np.zeros((self.dim, self.dim))
        
        # 对角元: 各卦内禀吉凶
        for i in range(self.dim):
            H[i,i] = self._trigram_yili_score(i)
        
        # 非对角元: 卦间耦合(相摩)
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                coupling = self._bet_cheng_coupling(i, j)
                H[i,j] = coupling
                H[j,i] = coupling  # 厄密性
        
        self.H = H
    
    def _build_unitary(self):
        """U_摩 = exp(-i H τ)"""
        self.U = expm(-1j * self.H * self.tau)
    
    def apply(self, psi: np.ndarray) -> np.ndarray:
        """应用酉变换: |ψ′⟩ = U_摩 |ψ⟩"""
        return self.U @ psi


# ============================================================
# 特征分形与八卦混合态
# ============================================================

class FeatureFractal:
    """
    特征分形: 物体的6个特征维度各自匹配一对对立八卦
    """
    
    def to_trigram_state(self, features: np.ndarray) -> np.ndarray:
        """6维特征 → 8维八卦叠加态"""
        probs = np.zeros(8)
        for fi, fval in enumerate(features):
            yg = FEATURE_DIM[fi]["yang"]
            ng = FEATURE_DIM[fi]["yin"]
            probs[yg] += fval
            probs[ng] += 1.0 - fval
        
        total = probs.sum()
        if total > 0:
            probs /= total
        
        # 概率幅
        psi = np.sqrt(probs).astype(complex)
        return psi
    
    def to_hex_state(self, features: np.ndarray, 
                     upper_skew: float = 0.6) -> np.ndarray:
        """
        特征 → 六十四卦初始态
        
        上卦(外)偏"显": 硬度(0)、动态(3)、重量(4)
        下卦(内)偏"隐": 粗糙(1)、形状(2)、纹理(5)
        """
        # 上卦: 偏显性特征
        upper_feats = np.array([
            features[0],   # 硬度
            features[3],   # 动态
            features[4],   # 重量
        ])
        
        # 下卦: 偏隐性特征
        lower_feats = np.array([
            features[1],   # 粗糙
            features[2],   # 形状
            features[5],   # 纹理
        ])
        
        # 用3维映射到八卦 (借用FEATURE_DIM的前3个和后3个)
        upper_psi = np.zeros(8, dtype=complex)
        lower_psi = np.zeros(8, dtype=complex)
        
        # 上卦: 取FEATURE_DIM[0,3,4]
        for fi_orig, fi in [(0,0), (3,1), (4,2)]:
            fval = features[fi_orig]
            yg = FEATURE_DIM[fi_orig]["yang"]
            ng = FEATURE_DIM[fi_orig]["yin"]
            upper_psi[yg] += fval * upper_skew
            upper_psi[ng] += (1-fval) * (1-upper_skew)
        
        # 下卦: 取FEATURE_DIM[1,2,5]
        for fi_orig, fi in [(1,0), (2,1), (5,2)]:
            fval = features[fi_orig]
            yg = FEATURE_DIM[fi_orig]["yang"]
            ng = FEATURE_DIM[fi_orig]["yin"]
            lower_psi[yg] += fval * (1-upper_skew)
            lower_psi[ng] += (1-fval) * upper_skew
        
        # 归一化
        for psi in [upper_psi, lower_psi]:
            nrm = np.linalg.norm(psi)
            if nrm > 0: psi /= nrm
        
        # 张量积: |Ψ₀⟩ = |ψ_上⟩ ⊗ |ψ_下⟩
        psi_hex = np.zeros(64, dtype=complex)
        for u in range(8):
            for l in range(8):
                idx = (u << 3) | l
                psi_hex[idx] = upper_psi[u] * lower_psi[l]
        
        nrm = np.linalg.norm(psi_hex)
        if nrm > 0: psi_hex /= nrm
        
        return psi_hex, upper_psi, lower_psi


# ============================================================
# 严格"八卦相荡"酉变换
# ============================================================

class StrictYiliInterference:
    """
    严格按论文实现的"八卦相荡"酉变换
    
    步骤:
      1. 特征分形 → 上卦|ψ_上⟩, 下卦|ψ_下⟩ (各8维)
      2. 八卦相荡: |ψ_上′⟩ = U_摩 |ψ_上⟩
                    |ψ_下′⟩ = U_摩 |ψ_下⟩
      3. 张量积:   |Ψ′⟩ = |ψ_上′⟩ ⊗ |ψ_下′⟩
      4. 测量涌现: TOP-K 六十四卦
    
    与v3.5的区别:
      - v3.5: 在H₆上用Grover → 搜索式涌现
      - v4.0: 在H₃上构造U_摩 → 真正的八卦相荡 → 张量积到H₆
      
    U_摩 = exp(-i H_易理 τ)，其中H_易理编码乘承比应耦合
    """
    
    def __init__(self, tau: float = 0.5):
        self.tau = tau
        self.unitary_8 = YiliUnitary(tau)
        self.fractal = FeatureFractal()
        
        # 预计算六十四卦评分 (用于分析)
        self._init_hex_scores()
    
    def _trigram_yili_3bit(self, idx: int) -> float:
        """3位卦评分"""
        bits = [(idx>>i)&1 for i in range(3)]
        s = 0.0
        for q in range(3):
            is_yang = (q%2==0)
            proper = (bits[q]==1 and is_yang) or (bits[q]==0 and not is_yang)
            s += 1.0 if proper else -1.0
        if bits[1]==0: s += 1.0
        else: s -= 1.0
        return s
    
    def _init_hex_scores(self):
        """64卦完整评分"""
        self.hex_scores = np.zeros(64)
        for idx in range(64):
            upper = idx >> 3
            lower = idx & 7
            ub = [(upper>>i)&1 for i in range(3)]
            lb = [(lower>>i)&1 for i in range(3)]
            bits = ub + lb
            s = 0.0
            for q in range(6):
                is_yang = (q%2==0)
                proper = (bits[q]==1 and is_yang) or (bits[q]==0 and not is_yang)
                s += 1.0 if proper else -1.0
            if bits[1]==0: s += 2.0
            else: s -= 2.0
            if bits[4]==1: s += 2.0
            else: s -= 2.0
            for (a,b) in [(0,3),(1,4),(2,5)]:
                if bits[a] != bits[b]: s += 1.0
                else: s -= 1.0
            self.hex_scores[idx] = s
    
    def run_trigram_interference(self, features: np.ndarray) -> dict:
        """八卦相荡 (8维H₃空间)"""
        _, upper_psi, lower_psi = self.fractal.to_hex_state(features)
        
        # 八卦相荡!
        upper_evolved = self.unitary_8.apply(upper_psi)
        lower_evolved = self.unitary_8.apply(lower_psi)
        
        top_u = sorted([(i, np.abs(upper_evolved[i])**2) for i in range(8)], key=lambda x:-x[1])
        top_l = sorted([(i, np.abs(lower_evolved[i])**2) for i in range(8)], key=lambda x:-x[1])
        
        return {
            'upper_init': upper_psi,
            'lower_init': lower_psi,
            'upper_evolved': upper_evolved,
            'lower_evolved': lower_evolved,
            'top_upper': top_u,
            'top_lower': top_l,
        }
    
    def run(self, features: np.ndarray) -> dict:
        """
        完整推理:
        特征分形 → 上/下八卦 → 各自相荡 → 张量积 → 六十四卦涌现
        """
        # 1. 特征分形
        psi_hex_init, upper_psi, lower_psi = self.fractal.to_hex_state(features)
        
        # 2. 八卦相荡 (核心!)
        upper_evolved = self.unitary_8.apply(upper_psi)
        lower_evolved = self.unitary_8.apply(lower_psi)
        
        # 3. 张量积: |Ψ′⟩ = |ψ_上′⟩ ⊗ |ψ_下′⟩
        psi_hex = np.zeros(64, dtype=complex)
        for u in range(8):
            for l in range(8):
                idx = (u << 3) | l
                psi_hex[idx] = upper_evolved[u] * lower_evolved[l]
        
        nrm = np.linalg.norm(psi_hex)
        if nrm > 0: psi_hex /= nrm
        
        # 4. 结果
        probs_init = np.abs(psi_hex_init)**2
        probs_final = np.abs(psi_hex)**2
        
        top_init = sorted([(i, probs_init[i]) for i in range(64)], key=lambda x:-x[1])[:5]
        top_final = sorted([(i, probs_final[i]) for i in range(64)], key=lambda x:-x[1])[:5]
        
        good_init = np.sum(probs_init[self.hex_scores > 0])
        good_final = np.sum(probs_final[self.hex_scores > 0])
        
        # 上/下卦变化
        upper_diff = np.linalg.norm(upper_evolved - upper_psi)
        lower_diff = np.linalg.norm(lower_evolved - lower_psi)
        return {
            'upper_init': upper_psi,
            'lower_init': lower_psi,
            'upper_evolved': upper_evolved,
            'lower_evolved': lower_evolved,
            'upper_diff': upper_diff,
            'lower_diff': lower_diff,
            'top_init': top_init,
            'top_final': top_final,
            'good_prob_init': good_init,
            'good_prob_final': good_final,
        }
    
    def run_trigram_sequence(self, features: np.ndarray, 
                              tau_values: List[float]) -> List[dict]:
        """不同τ值下的八卦相荡序列"""
        results = []
        for tau in tau_values:
            self.unitary_8 = YiliUnitary(tau)
            r = self.run_trigram_interference(features)
            results.append(r)
        return results


# ============================================================
# 策略映射
# ============================================================

def infer_strategy(hex_idx: int, features: np.ndarray) -> str:
    bits = [(hex_idx>>(5-i))&1 for i in range(6)]
    proper = sum(1 for q in range(6) if (bits[q]==1 and q%2==0) or (bits[q]==0 and q%2==1))
    center = (bits[4]==0 and bits[1]==1)
    response = sum(1 for (a,b) in [(5,2),(4,1),(3,0)] if bits[a] != bits[b])
    h, r, shape, dyn, w, tex = features
    
    if h > 0.7 and w > 0.7 and proper >= 3: return "power_grasp"
    if h > 0.6 and w < 0.4: return "cautious_grasp" if response < 2 else "precision_grasp"
    if h < 0.3 and w < 0.3: return "soft_grasp"
    if dyn > 0.7: return "adaptive_grasp" if response >= 2 else "dynamic_grasp"
    if r > 0.6 and shape > 0.6: return "stable_grasp"
    if center and proper >= 4: return "tactile_grasp" if tex > 0.5 else "stable_grasp"
    if r > 0.6 and h < 0.4: return "compliant_grasp"
    if w < 0.3 and tex > 0.6: return "compliant_grasp"
    if response <= 1: return "adaptive_grasp"
    if proper >= 4: return "power_grasp"
    elif proper <= 2: return "cautious_grasp"
    else: return "adaptive_grasp"


# ============================================================
# 验证实验
# ============================================================

def exp1_U_mo_structure():
    """展示U_摩的结构"""
    print("="*70)
    print(" 实验1: U_摩 酉矩阵结构分析")
    print("="*70)
    
    u8 = YiliUnitary(tau=0.5)
    
    print(f"\n  H_易理 (8×8 哈密顿量):")
    print(f"    H[{i},{i}] = {u8.H[i,i]:+.1f}  ({tname(i)})" for i in range(8))
    for i in range(8):
        print(f"    {tname(i)}: 评分{u8.H[i,i]:+.1f}", end="")
    print()
    
    print(f"\n  τ = {u8.tau}")
    print(f"  det(U) = {np.linalg.det(u8.U):.4f}  (应为|det|=1)")
    print(f"  U^†U 是否为单位阵: {np.allclose(u8.U.conj().T @ u8.U, np.eye(8))}")
    
    # 展示U_摩作用在均匀叠加态上的效果
    psi_uniform = np.ones(8, dtype=complex) / np.sqrt(8)
    psi_evolved = u8.apply(psi_uniform)
    
    print(f"\n  均匀叠加态 → 经U_摩干涉后:")
    probs = np.abs(psi_evolved)**2
    for i in sorted(range(8), key=lambda x:-probs[x]):
        print(f"    {tlabel(i)} {tname(i)}: {probs[i]*100:.2f}%")
    
    good_before = sum(np.abs(psi_uniform[u8.H.diagonal()>0])**2)
    good_after = sum(np.abs(psi_evolved[u8.H.diagonal()>0])**2)
    print(f"\n  吉卦总概率: {good_before*100:.1f}% → {good_after*100:.1f}%")
    print(f"  相长增益: {good_after/good_before:.2f}x" if good_before>0 else "")


def exp2_trigram_interference():
    """展示八卦相荡的过程"""
    print("\n"+"="*70)
    print(" 实验2: 八卦相荡 — 上卦与下卦各自干涉")
    print("="*70)
    
    si = StrictYiliInterference(tau=0.5)
    
    cases = [
        ("金属块", np.array([0.90, 0.10, 0.85, 0.10, 0.85, 0.20])),
        ("海绵",   np.array([0.10, 0.90, 0.20, 0.10, 0.10, 0.80])),
        ("皮球",   np.array([0.30, 0.40, 0.30, 0.90, 0.50, 0.30])),
    ]
    
    for name, feats in cases:
        r = si.run_trigram_interference(feats)
        
        print(f"\n▶ {name}")
        
        # 上卦变化
        print(f"  上卦(外) {FEATURE_DIM[0]['name']}/{FEATURE_DIM[3]['name']}/{FEATURE_DIM[4]['name']}:")
        print(f"    干涉前: " + " ".join(
            f"{tname(i)}({np.abs(r['upper_init'][i])**2*100:.0f}%)" 
            for i,p in r['top_upper'][:4]))
        print(f"    干涉后: " + " ".join(
            f"{tname(i)}({np.abs(r['upper_evolved'][i])**2*100:.0f}%)" 
            for i,p in r['top_upper'][:4]))
        
        # 下卦变化
        print(f"  下卦(内) {FEATURE_DIM[1]['name']}/{FEATURE_DIM[2]['name']}/{FEATURE_DIM[5]['name']}:")
        print(f"    干涉前: " + " ".join(
            f"{tname(i)}({np.abs(r['lower_init'][i])**2*100:.0f}%)" 
            for i,p in r['top_lower'][:4]))
        print(f"    干涉后: " + " ".join(
            f"{tname(i)}({np.abs(r['lower_evolved'][i])**2*100:.0f}%)" 
            for i,p in r['top_lower'][:4]))
        
        # 干涉强度 (用run方法的diff)
        r_full = si.run(feats)
        print(f"    上卦变化量: {r_full['upper_diff']:.4f}, 下卦变化量: {r_full['lower_diff']:.4f}")


def exp3_hex_emergence():
    """六十四卦涌现"""
    print("\n"+"="*70)
    print(" 实验3: 六十四卦涌现 — 八卦相荡 → 张量积 → 六十四卦")
    print("="*70)
    
    si = StrictYiliInterference(tau=0.6)
    
    cases = [
        ("金属块", np.array([0.90, 0.10, 0.85, 0.10, 0.85, 0.20])),
        ("海绵",   np.array([0.10, 0.90, 0.20, 0.10, 0.10, 0.80])),
        ("瓷杯",   np.array([0.80, 0.15, 0.60, 0.10, 0.40, 0.30])),
        ("皮球",   np.array([0.30, 0.40, 0.30, 0.90, 0.50, 0.30])),
        ("木块",   np.array([0.70, 0.70, 0.80, 0.20, 0.70, 0.60])),
        ("纸团",   np.array([0.15, 0.80, 0.10, 0.10, 0.10, 0.90])),
        ("鸡蛋",   np.array([0.50, 0.20, 0.50, 0.10, 0.10, 0.20])),
        ("保龄球", np.array([0.95, 0.10, 0.90, 0.80, 0.95, 0.10])),
        ("沙袋",   np.array([0.20, 0.85, 0.10, 0.10, 0.70, 0.70])),
        ("石头",   np.array([0.95, 0.80, 0.70, 0.15, 0.90, 0.50])),
        ("羽毛",   np.array([0.05, 0.30, 0.20, 0.10, 0.05, 0.30])),
        ("书本",   np.array([0.60, 0.50, 0.90, 0.05, 0.60, 0.40])),
    ]
    
    for name, feats in cases:
        r = si.run(feats)
        
        init_desc = ' '.join(f"{hname(i)}({p*100:.1f}%)" for i,p in r['top_init'][:3])
        final_desc = ' '.join(
            f"{hname(i)}→{infer_strategy(i,feats)}({p*100:.1f}%)" 
            for i,p in r['top_final'][:3]
        )
        
        gb = r['good_prob_init']*100
        gf = r['good_prob_final']*100
        gain = gf/gb if gb > 0 else 0
        
        print(f"\n▶ {name}")
        print(f"  干涉前: {init_desc}")
        print(f"  干涉后: {final_desc}")
        print(f"  吉卦: {gb:.1f}% → {gf:.1f}% (增益{gain:.2f}x)")
        print(f"  上卦变化{r['upper_diff']:.4f} 下卦变化{r['lower_diff']:.4f}")


def exp4_tau_sweep():
    """τ参数扫描: 不同强度八卦相荡的效果"""
    print("\n"+"="*70)
    print(" 实验4: τ扫描 — 不同相荡强度的涌现效果")
    print("="*70)
    
    si = StrictYiliInterference()
    feats = np.array([0.90, 0.10, 0.85, 0.10, 0.85, 0.20])
    
    print(f"\n  物体: 金属块 (硬光滑重)")
    print(f"  τ  |  吉卦概率  | 上卦Δ   | 下卦Δ   | TOP1卦")
    print("-"*55)
    
    for tau in [0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5]:
        si.tau = tau
        si.unitary_8 = YiliUnitary(tau)
        r = si.run(feats)
        top1 = hname(r['top_final'][0][0])
        gf = r['good_prob_final']*100
        print(f"  {tau:.1f} | {gf:8.1f}%   | {r['upper_diff']:.4f} | {r['lower_diff']:.4f} | {top1}")


def exp5_uniform_input():
    """全不确定输入: 纯U_摩的干涉效果"""
    print("\n"+"="*70)
    print(" 实验5: 完全不确定输入(全0.5) — 纯八卦相荡")
    print("="*70)
    
    si = StrictYiliInterference(tau=0.6)
    feats = np.array([0.5]*6)
    r = si.run(feats)
    
    print(f"\n  上卦干涉结果:")
    for i in sorted(range(8), key=lambda x:-np.abs(r['upper_evolved'][x])**2):
        print(f"    {tlabel(i)} {tname(i)}: {np.abs(r['upper_evolved'][i])**2*100:.2f}%")
    
    print(f"\n  下卦干涉结果:")
    for i in sorted(range(8), key=lambda x:-np.abs(r['lower_evolved'][x])**2):
        print(f"    {tlabel(i)} {tname(i)}: {np.abs(r['lower_evolved'][i])**2*100:.2f}%")
    
    print(f"\n  六十四卦涌现TOP6:")
    for i, p in r['top_final'][:6]:
        print(f"    {hlabel(i)} {hname(i)}(卦{i+1}): {p*100:.2f}% 评分{si.hex_scores[i]:+.0f}")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  QYUF v4.0 — 严格八卦相荡酉变换                       ║")
    print("║                                                       ║")
    print("║  U_摩 = exp(-i H_易理 τ) : H₃ → H₃                   ║")
    print("║  |Ψ′⟩ = (U_摩 ⊗ U_摩) |Ψ₀⟩                            ║")
    print("║  八卦相荡 → 张量积 → 六十四卦涌现                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    exp1_U_mo_structure()
    exp2_trigram_interference()
    exp3_hex_emergence()
    exp4_tau_sweep()
    exp5_uniform_input()
    
    print("\n"+"="*70)
    print(" 严格对照论文实现总结:")
    print("  ✓ U_摩: 8×8酉矩阵, exp(-iHτ)构造")
    print("  ✓ 刚柔相摩: H的非对角元编码卦间耦合")
    print("  ✓ 八卦相荡: U_摩作用在八卦叠加态上")
    print("  ✓ 乘承比应: 编码在H的耦合强度中")
    print("  ✓ 六十四卦: U_摩⊗U_摩的张量积")
    print("  ✓ 测量涌现: 概率幅平方取TOP-K")
    print("="*70)
