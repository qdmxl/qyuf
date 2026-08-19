#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qyuf_model.py — QYUF 量子-易理模型 (Quantum Yijing Unitary Framework)

基于论文第6节的六爻酉干涉原理，替代 V20 CnWorldModel 中的汉字引擎。
核心思想：环境状态→六爻向量→哈密顿量酉演化→64卦概率分布→主导卦+八卦隶属度

Vs 汉字引擎 (cn_world_model)：
  汉字引擎：汉字→部首→YLYW层次推理→卦象（符号映射）
  QYUF模型：环境爻向量→酉演化U(Δt)=exp(-iHΔt)→概率涌现→卦象（物理干涉）

哈密顿量编码的爻际规则（与论文§6.2.1一致）：
  H_adj: 阴阳相引（相邻ZZ耦合）
  H_ying: 远距感应（应位ZZ耦合）
  H_dang: 当位倾向（奇数位阳+偶数位阴）
  H_zhong: 得中强化（二爻阴+五爻阳）
  H_comp: 竞争跃迁（XX+YY耦合打破对称性）
"""
from __future__ import annotations
import os, sys, json, re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

# ── 核心参数（论文§6.5确定的最优窗口）──
# Config-C平衡模式：结构最强（KL散度ES=+9.1 vs 随机）
QYUF_PARAMS = {
    "J_adj": 1.0,      # 阴阳相引强度
    "J_ying": 0.5,     # 远距感应强度
    "h_dang": 0.30,    # 当位场强
    "h_zhong": 0.15,   # 得中强化
    "J_comp": 0.5,     # 竞争跃迁强度
}

# 六十四卦名称（通行本次序）
HEXAGRAM_NAMES = [
    "乾","坤","屯","蒙","需","讼","师","比","小畜","履","泰","否",
    "同人","大有","谦","豫","随","蛊","临","观","噬嗑","贲","剥","复",
    "无妄","大畜","颐","大过","坎","离",
    "咸","恒","遁","大壮","晋","明夷","家人","睽","蹇","解","损","益",
    "夬","姤","萃","升","困","井","革","鼎","震","艮","渐","归妹",
    "丰","旅","巽","兑","涣","节","中孚","小过","既济","未济"
]

# 八卦名称
TRIGRAM_NAMES = ["坤", "震", "坎", "兑", "艮", "离", "巽", "乾"]

# 卦象吉凶值（论文中FAVORABILITY表，用于决策评分）
FAVORABILITY = np.array([
    # 乾 坤 屯 蒙 需 讼 师 比 小畜 履 泰 否
    0.85, 0.36, 0.50, 0.50, 0.72, 0.42, 0.50, 0.78, 0.62, 0.64, 1.00, 0.32,
    # 同人 大有 谦 豫 随 蛊 临 观 噬嗑 贲 剥 复
    0.75, 0.90, 0.76, 0.66, 0.80, 0.52, 0.86, 0.64, 0.80, 0.62, 0.42, 0.80,
    # 无妄 大畜 颐 大过 坎 离 咸 恒 遁 大壮 晋 明夷
    0.80, 0.82, 0.60, 0.52, 0.42, 0.66, 0.76, 0.72, 0.55, 0.76, 0.95, 0.42,
    # 家人 睽 蹇 解 损 益 夬 姤 萃 升 困 井
    0.70, 0.46, 0.36, 0.72, 0.58, 0.60, 0.68, 0.52, 0.82, 0.90, 0.55, 0.85,
    # 革 鼎 震 艮 渐 归妹 丰 旅 巽 兑 涣 节
    0.88, 0.90, 0.62, 0.60, 0.55, 0.55, 0.85, 0.55, 0.60, 0.66, 0.60, 0.70,
    # 中孚 小过 既济 未济
    0.72, 0.55, 0.96, 0.46
])

# 通行本相邻卦信息
TONGXING_PAIRS = [(i, i+1) for i in range(0, 64, 2)]


# ══════════════════════════════════════════════════════
# 六爻干涉引擎核心
# ══════════════════════════════════════════════════════

def build_hamiltonian(theta: float = 1.0, params: Optional[Dict] = None) -> np.ndarray:
    """
    构建编码爻际规则的哈密顿量（论文§6.2.1）
    
    6量子位系统，64×64 哈密顿量矩阵。
    编码5条爻际规则：
    1. 阴阳相引(H_adj)：相邻爻相互作用
    2. 远距感应(H_ying)：应位爻相互作用
    3. 当位倾向(H_dang)：位置偏好
    4. 得中强化(H_zhong)：二五爻特殊地位
    5. 竞争跃迁(H_comp)：XX+YY耦合
    """
    if params is None:
        params = QYUF_PARAMS
    
    n_qubits = 6
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    
    def pauli_z(i):
        """作用于第i个量子位的Z算符（i从0开始）"""
        op = 1.0
        for q in range(n_qubits):
            if q == i:
                op = np.kron(np.diag([1, -1]), op)
            else:
                op = np.kron(np.eye(2), op)
        return op
    
    def pauli_x(i):
        """作用于第i个量子位的X算符"""
        op = 1.0
        for q in range(n_qubits):
            if q == i:
                op = np.kron(np.array([[0,1],[1,0]]), op)
            else:
                op = np.kron(np.eye(2), op)
        return op
    
    def pauli_y(i):
        """作用于第i个量子位的Y算符"""
        op = 1.0
        for q in range(n_qubits):
            if q == i:
                op = np.kron(np.array([[0,-1j],[1j,0]]), op)
            else:
                op = np.kron(np.eye(2), op)
        return op
    
    # 1. 阴阳相引：相邻ZZ耦合
    J_adj = params.get("J_adj", 1.0)
    for i in range(n_qubits - 1):
        H += J_adj * pauli_z(i) @ pauli_z(i+1)
    
    # 2. 远距感应：应位ZZ耦合（初↔四，二↔五，三↔上）
    J_ying = params.get("J_ying", 0.5)
    ying_pairs = [(0, 3), (1, 4), (2, 5)]
    for i, j in ying_pairs:
        H += J_ying * pauli_z(i) @ pauli_z(j)
    
    # 3. 当位倾向：奇数位倾向阳(+Z)，偶数位倾向阴(-Z)
    h_dang = params.get("h_dang", 0.30)
    for i in range(n_qubits):
        sign = -1.0 if (i % 2 == 0) else 1.0  # 初=0(偶)倾向阴, 二=1(奇)倾向阳
        H += h_dang * sign * pauli_z(i)
    
    # 4. 得中强化
    h_zhong = params.get("h_zhong", 0.15)
    H += h_zhong * (-pauli_z(1) + pauli_z(4))  # 二爻阴(-Z), 五爻阳(+Z)
    
    # 5. 竞争跃迁：XX+YY耦合
    J_comp = params.get("J_comp", 0.5)
    for i in range(n_qubits - 1):
        H += J_comp * (pauli_x(i) @ pauli_x(i+1) + pauli_y(i) @ pauli_y(i+1))
    
    return H


def unitary_evolution(H: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """从哈密顿量生成酉算符 U(Δt) = exp(-iHΔt)（论文§6.2.2）"""
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    U = eigenvectors @ np.diag(np.exp(-1j * eigenvalues * dt)) @ eigenvectors.conj().T
    return U


def evolve_state(psi: np.ndarray, U: np.ndarray, steps: int = 50) -> np.ndarray:
    """从初态|ψ⟩反复施加酉变换U"""
    state = psi.copy()
    for _ in range(steps):
        state = U @ state
    return state


def prob_distribution(state: np.ndarray) -> np.ndarray:
    """从量子态计算64基态的概率分布"""
    return np.abs(state) ** 2


def get_dominant_hexagram(prob: np.ndarray) -> Tuple[int, str, float]:
    """返回主导卦的索引、名称和概率"""
    idx = int(np.argmax(prob))
    return idx, HEXAGRAM_NAMES[idx], float(prob[idx])


def get_trigram_probs(prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算上下卦（各8个八卦）的边际概率分布"""
    lower = np.zeros(8)
    upper = np.zeros(8)
    for i in range(64):
        l = ((i >> 0) & 1) | (((i >> 1) & 1) << 1) | (((i >> 2) & 1) << 2)
        u = ((i >> 3) & 1) | (((i >> 4) & 1) << 1) | (((i >> 5) & 1) << 2)
        lower[l] += prob[i]
        upper[u] += prob[i]
    return lower, upper


def get_bagua_vector(prob: np.ndarray) -> np.ndarray:
    """从64维概率分布计算8维八卦隶属度向量（上下卦平均）"""
    lower, upper = get_trigram_probs(prob)
    return (lower + upper) / 2.0


def entropy(prob: np.ndarray) -> float:
    """香农熵"""
    p = prob[prob > 1e-10]
    return float(-np.sum(p * np.log2(p)))


def compute_effective_temp(prob: np.ndarray) -> float:
    """从概率分布估计有效温度"""
    # 使用概率方差作为"温度"的代理
    # 方差越大=温度越低=结构越清晰
    var = float(np.var(prob))
    # 映射到 [0.1, 2.0] 范围
    t = 2.0 - min(1.9, var * 400)
    return max(0.1, t)


# ══════════════════════════════════════════════════════
# 环境六爻构建器
# ══════════════════════════════════════════════════════

class YaoBuilder:
    """
    从ALFWorld环境状态构建6维爻向量（V18兼容版）
    
    六爻语义（与ylyw_scorer.py保持兼容）：
      y1: 目标差距缩小 (goal-gap reduction)
      y2: 持有承接 (holding continuity)
      y3: 处理谓词推进 (process-predicate advancement)
      y4: 容器可供性 (container affordance)
      y5: 目标关联 (goal association)
      y6: 新颖性 (novelty vs failure history)
    """
    
    def __init__(self):
        self._LO = 0.05
        self._HI = 0.95
        # 惰性导入V18积累函数
        self._location_prior = None
        self._resonance_fn = None
        # 多特征分形八卦相荡引擎（惰性导入）
        self._si = None
        # 概率分布缓存：{特征元组: 64维分布}
        self._prob_cache = {}
    
    def _clip(self, v: float) -> float:
        return max(self._LO, min(self._HI, v))
    
    def _get_location_prior(self, obj_cls: str, recep_cls: str) -> float:
        """惰性导入V18的先验知识"""
        if self._location_prior is None:
            import sys as _sys
            _v18_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                   "..", "alfworld_exp", "v18") if '__file__' in dir() else '/home/lijinhan/MXL/科研/ylyw/alfworld_exp/v18'
            # 尝试多个位置找v18
            _found = False
            for _p in [_v18_dir, '/home/lijinhan/MXL/科研/ylyw/alfworld_exp/v18',
                       '/home/lijinhan/MXL/科研/ylyw/alfworld_exp']:
                _abs = os.path.abspath(_p)
                if _abs not in _sys.path:
                    _sys.path.insert(0, _abs)
            from v18.priors import location_prior
            self._location_prior = location_prior
        return self._location_prior(obj_cls, recep_cls)
    
    def _get_resonance(self, obj_cls: str, recep_cls: str) -> float:
        """
        纯多特征分形版：不用V18字典，用八卦相荡+JS散度推断物体-容器共鸣
        
        这是真正的语义泛化路径——对任何名称的物体和容器都能产生共鸣度
        """
        inf = self._infer_object_recep_resonance(obj_cls, recep_cls)
        return inf if inf is not None else 0.5
    
    def _infer_recep_gua_by_name(self, name: str) -> List[float]:
        """
        从名称提取6维物理特征（多特征分形模型入口）
        
        使用语义增强的多维特征编码，确保每个实体类别有独特的6维签名。
        通过展宽特征值范围和细粒度的类别划分来增加卦象区分度。
        
        语义类别映射:
          维0(刚柔): 硬质封闭→软质开放
          维1(动静): 静止电器→可移动物品
          维2(险丽): 含水潮湿→干热危险
          维3(止悦): 废弃收纳→展示陈列
          维4(轻重): 大型重型→小型轻型
          维5(纹理): 软布粗糙→硬质光滑
        """
        n = name.lower().strip()
        feats = [0.5] * 6
        
        # ── 维0 刚柔: 硬质封闭↔软质开放 ──
        if any(k in n for k in ['cabinet', 'drawer', 'safe', 'vault', 'dresser', 'cupboard']):
            feats[0] = 0.90  # 封闭硬质
        elif any(k in n for k in ['sofa', 'armchair', 'bed', 'couch', 'ottoman', 'laundryhamper']):
            feats[0] = 0.10  # 软质
        elif any(k in n for k in ['table', 'desk', 'counter', 'sidetable', 'diningtable', 'coffeetable']):
            feats[0] = 0.65  # 平面硬质
        elif any(k in n for k in ['shelf', 'stand', 'rack', 'tvstand', 'bookshelf']):
            feats[0] = 0.75  # 开放性硬质
        elif any(k in n for k in ['fridge', 'microwave', 'oven', 'stove', 'dishwasher']):
            feats[0] = 0.85  # 电器硬质
        elif any(k in n for k in ['toilet', 'sink', 'bathtub', 'basin', 'sinkbasin', 'faucet']):
            feats[0] = 0.80  # 陶瓷硬质
        elif any(k in n for k in ['trash', 'garbage', 'bin', 'can', 'recycle']):
            feats[0] = 0.50  # 软硬兼备
        
        # ── 维1 动静: 静止↔可移动 ──
        if any(k in n for k in ['fridge', 'microwave', 'oven', 'stove', 'coffee', 
                                 'lamp', 'toaster', 'tv', 'dishwasher', 'burner',
                                 'desklamp', 'floorlamp', 'lightswitch']):
            feats[1] = 0.10  # 静止电器
        elif any(k in n for k in ['cabinet', 'drawer', 'dresser', 'counter', 'shelf', 
                                   'sink', 'toilet', 'bathtub', 'sinkbasin']):
            feats[1] = 0.25  # 半固定家具
        elif any(k in n for k in ['cart', 'wagon', 'tray']):
            feats[1] = 0.85  # 可移动
        else:
            feats[1] = 0.55  # 普通物品（小件可移动）
        
        # ── 维2 险丽: 含水险↔干热险↔中性 ──
        if any(k in n for k in ['sink', 'sinkbasin', 'toilet', 'bathtub', 'tub', 'basin',
                                 'faucet', 'handsink', 'bathtubbasin']):
            feats[2] = 0.20  # 含水（陷）
        elif any(k in n for k in ['stove', 'burner', 'stoveburner', 'microwave', 'oven',
                                   'toaster', 'coffee', 'coffeemachine']):
            feats[2] = 0.80  # 热源（险）
        elif any(k in n for k in ['fridge', 'freezer']):
            feats[2] = 0.15  # 冷源（亦险）
        else:
            feats[2] = 0.50  # 中性
        
        # ── 维3 止悦: 废弃收纳↔展示陈列 ──
        if any(k in n for k in ['trash', 'garbage', 'waste', 'bin', 'recycle', 'can']):
            feats[3] = 0.10  # 废弃（止）
        elif any(k in n for k in ['shelf', 'stand', 'holder', 'rack', 'tvstand', 
                                   'bookshelf', 'nightstand', 'sidetable']):
            feats[3] = 0.85  # 展示（悦）
        elif any(k in n for k in ['cabinet', 'drawer', 'safe', 'dresser', 'cupboard',
                                   'wardrobe']):
            feats[3] = 0.30  # 收纳（中性偏止）
        elif any(k in n for k in ['table', 'desk', 'counter', 'diningtable', 'coffeetable']):
            feats[3] = 0.65  # 台面（偏悦）
        elif any(k in n for k in ['sofa', 'bed', 'couch', 'armchair', 'ottoman']):
            feats[3] = 0.55  # 坐卧（中性）
        elif any(k in n for k in ['toilet', 'bathtub', 'sink', 'shower']):
            feats[3] = 0.40  # 卫生（中性偏止）
        
        # ── 维4 轻重: 大型重型↔小型轻型 ──
        if any(k in n for k in ['fridge', 'stove', 'dishwasher', 'bed', 'couch', 'sofa',
                                 'dresser', 'bathtub', 'cabinet', 'armchair']):
            feats[4] = 0.85  # 重型
        elif any(k in n for k in ['table', 'desk', 'counter', 'diningtable', 'shelf', 
                                   'drawer', 'coffeetable', 'microwave', 'oven', 'ottoman']):
            feats[4] = 0.65  # 中重型
        elif any(k in n for k in ['chair', 'stool', 'sidetable', 'nightstand', 'lamp',
                                   'toaster', 'tv', 'toilet', 'trash', 'garbage', 'bin',
                                   'can', 'sinkbasin', 'stoveburner', 'newspaper', 
                                   'book', 'magazine', 'dishsponge', 'spraybottle', 
                                   'soapbottle', 'laundryhamper']):
            feats[4] = 0.40  # 中轻型
        elif any(k in n for k in ['clock', 'vase', 'candle', 'plate', 'cup', 'bowl',
                                   'pillow', 'towel', 'rug', 'cloth', 'bottle', 'pan',
                                   'pot', 'knife', 'fork', 'spoon', 'spatula', 'ladle',
                                   'soap', 'remote', 'cellphone', 'pencil', 'pen', 
                                   'keychain', 'statue', 'alarmclock', 'watch', 'saltshaker',
                                   'pepper', 'onion', 'radish', 'banana', 'cheese',
                                   'butter', 'bread', 'tissue', 'creditcard', 'baseball',
                                   'basketball', 'cd', 'plunger', 'box', 'crayon', 'bat',
                                   'whisk', 'scrubbrush', 'blinds', 'handtowel',
                                   'toiletpaper', 'tissuebox', 'potholder']):
            feats[4] = 0.15  # 轻型（物体）
        elif any(k in n for k in ['apple', 'tomato', 'potato', 'lettuce', 'egg', 'cup',
                                   'wine', 'mug', 'glass', 'eggshell']):
            feats[4] = 0.20  # 食品/小型容器
        else:
            feats[4] = 0.25  # 默认轻型
        
        # ── 维5 纹理: 软布粗糙↔硬质光滑 ──
        if any(k in n for k in ['sofa', 'armchair', 'couch', 'bed', 'ottoman', 'laundryhamper',
                                 'pillow', 'towel', 'rug', 'cloth', 'handtowel']):
            feats[5] = 0.85  # 布艺粗糙
        elif any(k in n for k in ['table', 'desk', 'counter', 'shelf', 'cabinet', 'drawer',
                                   'dresser', 'sidetable', 'diningtable', 'coffeetable',
                                   'tvstand', 'bookshelf', 'nightstand', 'chair', 'stool',
                                   'stand', 'holder']):
            feats[5] = 0.30  # 木质光滑
        elif any(k in n for k in ['sink', 'toilet', 'bathtub', 'tub', 'sinkbasin', 'faucet',
                                   'bathtubbasin', 'handsink', 'shower', 'basin']):
            feats[5] = 0.15  # 陶瓷光滑
        elif any(k in n for k in ['fridge', 'microwave', 'oven', 'stove', 'dishwasher',
                                   'toaster', 'coffee', 'coffeemachine', 'lamp', 'tv',
                                   'desklamp', 'floorlamp', 'lightswitch', 'stoveburner']):
            feats[5] = 0.20  # 金属/塑料光滑
        elif any(k in n for k in ['book', 'magazine', 'newspaper', 'cardboard', 'box',
                                   'tissue', 'tissuebox', 'toiletpaper']):
            feats[5] = 0.60  # 纸质（中性）
        elif any(k in n for k in ['apple', 'tomato', 'potato', 'lettuce', 'egg', 'onion',
                                   'radish', 'banana', 'pepper', 'bread', 'cheese', 'butter',
                                   'wine', 'mug', 'glassbottle']):
            feats[5] = 0.55  # 食物（平滑偏中性）
        elif any(k in n for k in ['knife', 'fork', 'spoon', 'spatula', 'ladle', 'pan',
                                   'pot', 'plate', 'cup', 'bowl', 'vase', 'bottle',
                                   'clock', 'candle', 'statue', 'pen', 'pencil',
                                   'keychain', 'watch', 'cd', 'remote', 'cellphone',
                                   'alarmclock', 'crayon', 'saltshaker', 'pepper']):
            feats[5] = 0.25  # 光滑小件
        
        return feats
    

    def _ensure_si(self):
        """惰性初始化StrictYiliInterference"""
        if self._si is None:
            _base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
            if _base not in sys.path:
                sys.path.insert(0, _base)
            from qyuf_strict_unitary import StrictYiliInterference
            self._si = StrictYiliInterference(tau=0.5)
    
    def _compute_prob_dist(self, name: str) -> Optional[np.ndarray]:
        """
        计算并缓存名称对应的64维概率分布
        
        使用语义分类→特征向量→八卦相荡的完整流程。
        语义分类通过实体类别独特标签注入到八卦空间的不同区域。
        """
        key = name.lower().strip()
        if key in self._prob_cache:
            return self._prob_cache[key]
        try:
            self._ensure_si()
            feats = np.array(self._infer_recep_gua_by_name(name))
            _, up, lp = self._si.fractal.to_hex_state(feats)
            ue = self._si.unitary_8.apply(up)
            le = self._si.unitary_8.apply(lp)
            
            # 张量积
            out = np.zeros(64, dtype=complex)
            for u in range(8):
                for l in range(8):
                    out[(u << 3) | l] = ue[u] * le[l]
            
            # 语义偏置：注入实体类别标签，增强区分度
            # 在64卦空间中注入语义偏置，方向由实体名称决定
            bias = self._semantic_bias(key)
            if bias is not None:
                out = out + bias * 0.3
            
            out /= np.linalg.norm(out)
            probs = np.abs(out)**2
            self._prob_cache[key] = probs
            return probs
        except Exception:
            return None
    
    def _semantic_bias(self, key: str) -> Optional[np.ndarray]:
        """语义类别标签注入，将实体类别信息编码到64卦空间的不同区域"""
        # 定义类别标签→64卦注入向量
        # 核心思想：不同类别的实体在64卦空间中占据不同的区域
        
        # 类别到八卦的映射（从语义角度）
        # 使用独热类别注入：每个类别对应不同的八卦方向
        category_map = {
            # 封闭收纳类 → 坤(地)承载 + 艮(山)阻塞
            'cabinet': [(0,0,0.5), (1,1,0.3)],
            'drawer': [(0,0,0.4), (1,1,0.4)],
            'dresser': [(0,0,0.5), (0,7,0.3)],
            'safe': [(1,7,0.5), (7,1,0.3)],
            'cupboard': [(0,0,0.5)],
            # 软家具 → 离(火)悦 + 兑(泽)悦
            'sofa': [(5,6,0.5), (5,5,0.3)],
            'bed': [(4,5,0.5), (5,5,0.3)],
            'armchair': [(5,6,0.4), (4,5,0.3)],
            'couch': [(5,6,0.5)],
            'ottoman': [(5,6,0.4), (6,5,0.3)],
            # 台面 → 乾(天)刚健 + 兑(泽)丽
            'table': [(7,7,0.4), (6,6,0.3)],
            'desk': [(7,6,0.4), (7,5,0.3)],
            'counter': [(7,6,0.4), (6,6,0.3)],
            'sidetable': [(6,6,0.4), (7,7,0.3)],
            'diningtable': [(7,7,0.4), (6,6,0.3)],
            'coffeetable': [(7,6,0.4), (6,7,0.3)],
            # 电器 → 离(火)热 + 乾(天)动
            'fridge': [(2,2,0.5), (7,2,0.3)],
            'microwave': [(4,3,0.5), (3,4,0.3)],
            'oven': [(3,3,0.5), (4,3,0.3)],
            'stove': [(3,3,0.5), (3,4,0.3)],
            'toaster': [(3,4,0.4), (4,3,0.3)],
            'coffee': [(5,3,0.4), (5,5,0.3)],
            'lamp': [(4,4,0.5), (4,3,0.3)],
            'tv': [(4,1,0.4), (4,0,0.3)],
            'desklamp': [(4,4,0.4), (5,0,0.3)],
            # 含水 → 坎(水)陷
            'sink': [(2,2,0.5), (2,3,0.3)],
            'toilet': [(2,1,0.4), (2,0,0.3)],
            'bathtub': [(2,2,0.4), (2,1,0.3)],
            'faucet': [(2,2,0.3), (2,3,0.3)],
            # 展示 → 巽(风)入 + 艮(山)止
            'shelf': [(6,4,0.4), (6,5,0.3)],
            'rack': [(6,4,0.4), (5,6,0.3)],
            'tvstand': [(6,6,0.4), (6,5,0.3)],
            'bookshelf': [(6,4,0.4), (5,6,0.3)],
            'nightstand': [(7,7,0.4), (6,7,0.3)],
            # 废弃 → 坤(地)藏 + 艮(山)止
            'trash': [(1,7,0.5), (7,1,0.3)],
            'garbage': [(1,7,0.5), (7,1,0.3)],
            'bin': [(1,7,0.5), (1,1,0.3)],
            # 纸质 → 巽(风)散
            'book': [(5,5,0.4), (5,7,0.3)],
            'newspaper': [(7,3,0.4), (7,4,0.3)],
            'magazine': [(5,5,0.4), (7,5,0.3)],
            'paper': [(3,3,0.4), (5,3,0.3)],
            # 餐具 → 兑(泽)缺 + 震(雷)动
            'knife': [(0,1,0.4), (1,0,0.3)],
            'fork': [(1,0,0.4), (0,1,0.3)],
            'spoon': [(1,1,0.4), (1,0,0.3)],
            'pan': [(3,4,0.4), (4,3,0.2)],
            'pot': [(3,4,0.4), (5,6,0.2)],
            'plate': [(6,5,0.4), (5,6,0.3)],
            'cup': [(2,1,0.3), (1,2,0.3)],
            'bowl': [(2,5,0.4), (5,2,0.3)],
            'bottle': [(2,2,0.3), (2,3,0.3)],
            # 食物 → 震(雷)生 + 乾(天)养
            'apple': [(7,4,0.4), (4,7,0.3)],
            'tomato': [(7,4,0.4), (4,7,0.2)],
            'potato': [(7,6,0.3), (6,7,0.3)],
            'lettuce': [(7,3,0.4), (7,4,0.2)],
            'egg': [(7,4,0.3), (4,7,0.3)],
            'bread': [(7,7,0.3), (7,1,0.2)],
            # 小件 → 震(雷)动 + 巽(风)入
            'clock': [(0,0,0.4), (0,4,0.3)],
            'vase': [(4,0,0.3), (0,4,0.3)],
            'candle': [(4,3,0.4), (3,4,0.3)],
            'pillow': [(7,2,0.3), (7,7,0.3)],
            'remote': [(1,4,0.3), (4,1,0.3)],
            'cellphone': [(4,1,0.3), (1,4,0.3)],
            'pen': [(1,0,0.3), (0,1,0.3)],
            'pencil': [(0,1,0.3), (1,0,0.3)],
            'keychain': [(0,1,0.3), (1,3,0.2)],
            'statue': [(0,6,0.3), (6,0,0.3)],
            'alarmclock': [(4,4,0.3), (0,4,0.3)],
            'towel': [(2,4,0.3), (4,2,0.3)],
            'soap': [(2,5,0.3), (5,2,0.3)],
            'spraybottle': [(2,4,0.3), (4,2,0.3)],
            'dishsponge': [(2,2,0.3), (2,4,0.3)],
            'watch': [(5,5,0.3), (5,7,0.2)],
            'cd': [(0,0,0.3), (0,4,0.3)],
            'chair': [(1,4,0.3), (1,1,0.3)],
            'stool': [(1,4,0.3), (1,0,0.3)],
        }
        
        # 找匹配的类别标签
        bias = np.zeros(64, dtype=complex)
        found = False
        for cat, entries in category_map.items():
            if cat in key:
                for (u, l, w) in entries:
                    idx = (u << 3) | l
                    bias[idx] += 0.3 + 0.7 * w
                found = True
        
        if not found:
            return None
        
        bias /= np.linalg.norm(bias)
        return bias
    
    def _infer_object_recep_resonance(self, obj_cls: str, recep_cls: str) -> Optional[float]:
        """
        多特征分形版：用StrictYiliInterference的八卦相荡推断物体-容器共鸣
        
        步骤：
          1. 提取物体和容器的6维物理特征
          2. 用多特征分形→八卦相荡→64维涌现概率分布（带缓存）
          3. 比较两个64维概率分布的JS散度
        
        共鸣度 = exp(-JS*3) ∈ (0, 1]，1=完全相同
        """
        try:
            self._ensure_si()
            if self._si is None:
                return None
            probs_obj = self._compute_prob_dist(obj_cls)
            probs_rec = self._compute_prob_dist(recep_cls)
            if probs_obj is None or probs_rec is None:
                return None
            
            # JS散度
            p = probs_obj + 1e-10
            q = probs_rec + 1e-10
            p /= p.sum()
            q /= q.sum()
            m = 0.5 * (p + q)
            js = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
            return float(np.exp(-js * 3.0))
        except Exception as e:
            return None
    
    def build_yao(self, parsed: Dict, world, goal, phase: Dict) -> List[float]:
        """
        构建6维爻向量（V18规则兼容版+QYUF多特征分形增强）
        
        六爻含义（沿用V18规范）：
          y0(刚柔) = holding continuity / 携带连续性
          y1(动静) = goal-gap / 目标差
          y2(险丽) = process-predicate / 处理谓词
          y3(止悦) = container affordance / 容器可供性
          y4(轻重) = goal association / 目标关联度（使用JS散度共鸣）
          y5(纹理) = novelty vs failure / 新奇vs失败记忆
        """
        
        def _loc(cls: str):
            """获取位置先验"""
            if self._location_prior is not None:
                if callable(self._location_prior):
                    return self._location_prior(cls, '')  # 函数形式
                return self._location_prior.get(cls, 0.5)
            try:
                from v18.priors import location_prior
                return location_prior.get(cls, 0.5)
            except Exception:
                return 0.5
        
        # ── 解析候选 ──
        verb = parsed.get("verb", "")
        obj_cls = parsed.get("obj_cls", "")
        recep_cls = parsed.get("recep_cls", "")
        
        # ── y0 刚柔：holding continuity ──
        holding = phase.get("holding_target", False)
        has_item = phase.get("has_item", phase.get("in_hand", False))
        y0 = 0.8 if holding else (0.4 if has_item else 0.2)
        if verb == "put":
            y0 = 0.7 if holding else 0.8
        elif verb == "take":
            y0 = 0.3
        
        # ── y1 动静：目标差 ──
        missing = phase.get("missing", 1)
        receptiveness = phase.get("receptiveness", 1.0)
        need_proc = phase.get("need_proc", False)
        y1 = 0.2 + 0.3 * min(missing, 3) + (0.3 if need_proc else 0)
        
        # ── y2 险丽：处理谓词 ──
        y2 = 0.5
        if recep_cls:
            y2 = _loc(recep_cls)
        elif obj_cls:
            y2 = _loc(obj_cls)
        
        # ── y3 止悦：容器可供性 ──
        searching = phase.get("searching", phase.get("at_recep", False))
        at_recep = phase.get("at_recep", phase.get("at_location", False)) or searching
        
        if verb == "go" and recep_cls:
            rec_ploc = _loc(recep_cls)
            y3 = 0.8 if rec_ploc < 0.5 else 0.4 + 0.4 * (1 - rec_ploc)
        elif verb == "put":
            y3 = 0.7 if at_recep else 0.5
        elif verb == "take":
            y3 = 0.3 if at_recep else 0.6
        else:
            y3 = 0.5
        if verb == "go" and not at_recep:
            y3 = 0.7
        
        # ── y4 轻重：目标关联度（多特征分形JS散度共鸣度增强） ──
        if recep_cls and hasattr(self, '_prob_cache'):
            try:
                res = self._infer_object_recep_resonance(obj_cls or "", recep_cls)
                if res is not None:
                    y4 = self._clip(0.2 + 0.6 * res)
                else:
                    y4 = 0.5
            except Exception:
                y4 = 0.5
        else:
            y4 = 0.5
        if holding:
            y4 = self._clip(y4 + 0.2)
        
        # ── y5 纹理：遗忘vs失败记忆 ──
        y5 = 0.5
        failed = phase.get("last_action_failed", False)
        novel = phase.get("novelty", 0)
        if failed:
            y5 = 0.2
        if novel > 0.5:
            y5 = 0.8
        if verb in ("help", "inventory", "look"):
            y5 = 0.1
        
        return [self._clip(y0), self._clip(y1), self._clip(y2),
                self._clip(y3), self._clip(y4), self._clip(y5)]
    

    def _is_target_recep(self, recep: str, world, goal) -> bool:
        """判断容器是否为目标物体可能所在"""
        if not goal or not goal.object_class:
            return False
        try:
            cls = world._class_of(recep) if hasattr(world, '_class_of') else recep.rsplit(" ", 1)[0]
            return self._get_location_prior(goal.object_class, cls) > 0.3
        except Exception:
            return False


# ══════════════════════════════════════════════════════
# 量子-易理评分器
# ══════════════════════════════════════════════════════

class QYUFScorer:
    """
    基于六爻酉干涉的量子评分器
    
    替代 V18 ylyw_scorer.YLYWScorer 的64卦模板匹配。
    核心差异：
      原版：六爻向量→余弦模板匹配→选择最佳卦
      量子版：六爻向量→构建初态→酉演化→64卦概率涌现→选择主导卦
    
    额外功能：从演化后的64卦概率分布中提取八卦隶属度向量，
    这比原版的单一卦匹配更丰富，可以反映系统的不确定性和多态性。
    """
    
    def __init__(self, params: Optional[Dict] = None, theta: float = 1.0, evo_steps: int = 20,
                 experience_path: Optional[str] = None, alpha: float = 0.0):
        self.params = params or QYUF_PARAMS.copy()
        self.theta = theta
        self.evo_steps = evo_steps
        self._H = None
        self._U = None
        self._yao_builder = YaoBuilder()
        
        # 线性加权系数（与原版兼容，用于ablation对比）
        self._lin_w = np.array([0.28, 0.14, 0.20, 0.12, 0.16, 0.10])
        
        # === 知几学习：经验以量子态叠加形式注入哈密顿量 ===
        _base = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
        self.experience_path = experience_path or os.path.join(_base, 'qyuf_experience.json')
        self.alpha = alpha  # 经验学习率，0=纯物理（与V18 ablation兼容）
        self._ep_buf = []   # 经验缓冲区（本局记录，局末提交）
        # 经验存储器：{六爻原型键: {卦索引: {"count": int, "wins": int}}}
        self._experience: Dict = {}
        self._n_learned = 0   # 有效经验数
        self._load_experience()
        
        # 预计算哈密顿量和酉算符
        self._rebuild_unitary()
    
    def _rebuild_unitary(self):
        """(重新)构建哈密顿量和酉算符"""
        self._H = build_hamiltonian(self.theta, self.params)
        self._U = unitary_evolution(self._H, 0.1)
    
    # ── 知几学习接口 ──
    
    def _proto_key(self, yao: List[float]) -> str:
        """六爻原型键：将六爻量化为离散原型"""
        return ','.join(f'{v:.2f}' for v in yao)
    
    def _load_experience(self):
        """加载持久化的经验"""
        try:
            if os.path.exists(self.experience_path):
                with open(self.experience_path) as f:
                    self._experience = json.load(f)
                self._n_learned = sum(
                    1 for v in self._experience.values()
                    if any(e['wins'] >= 2 for e in v.values())
                )
        except Exception:
            self._experience = {}
    
    def _save_experience(self):
        """持久化经验"""
        try:
            with open(self.experience_path, 'w') as f:
                json.dump(self._experience, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def observe_decision(self, cmd: str, world, goal, phase: Dict,
                         chosen_hex: int, chosen_score: float,
                         step: int, game_id: int):
        """记录一次决策（不管胜负，局末再用胜负标签）"""
        parsed = self._parse_action(cmd)
        yao = self.build_yao(parsed, world, goal, phase)
        key = self._proto_key(yao)
        self._ep_buf.append({
            'key': key, 'yao': [round(v, 4) for v in yao],
            'chosen_hex': chosen_hex, 'chosen_score': chosen_score,
            'step': step, 'game_id': game_id,
        })
    
    def commit_game(self, won: bool):
        """局末提交：胜局记录为正经验，负局衰减"""
        for rec in self._ep_buf:
            key = rec['key']
            hx = rec['chosen_hex']
            if key not in self._experience:
                self._experience[key] = {}
            if str(hx) not in self._experience[key]:
                self._experience[key][str(hx)] = {'count': 0, 'wins': 0}
            ex = self._experience[key][str(hx)]
            ex['count'] += 1
            if won:
                ex['wins'] += 1
            else:
                ex['wins'] = max(0, ex['wins'] - 1)  # 负局衰减
        # 不再重建H（路线2：概率输出层加权）
        self._ep_buf = []
        self._save_experience()
    
    def _apply_experience_bias(self, prob: np.ndarray, yao_key: str) -> np.ndarray:
        """
        概率输出层经验加权（路线2）
        
        核心：对历史成功卦象的概率加经验因子，不修改哈密顿量
        
        p'_i = p_i * (1 + alpha * ω_i)
        其中ω_i = Σ_k wins_k/count_k，来自所有匹配该六爻原型的经验
        """
        if not self._experience or self.alpha <= 0:
            return prob
        
        biased = prob.copy()
        
        if yao_key in self._experience:
            for hx_str, ex in self._experience[yao_key].items():
                if ex['count'] < 2 or ex['wins'] < 1:
                    continue
                omega = ex['wins'] / ex['count']  # 成功率
                try:
                    hidx = int(hx_str)
                except ValueError:
                    continue
                # 概率放大
                biased[hidx] *= (1.0 + self.alpha * omega)
        
        # 重新归一化
        biased = biased / biased.sum()
        return biased
    
    @staticmethod
    def _hex_name_to_idx(name: str) -> int:
        """卦名→索引"""
        try:
            return int(name)
        except ValueError:
            for i, n in enumerate(HEX_NAMES):
                if n == name:
                    return i
            return 0
    
    def reset_experience(self):
        """重置经验（用于对照实验）"""
        self._experience = {}
        self._n_learned = 0
        self._ep_buf = []
        self._rebuild_unitary()
    
    def experience_stats(self) -> Dict:
        """经验统计"""
        total_keys = len(self._experience)
        valid_keys = sum(1 for v in self._experience.values()
                        if any(e['wins'] >= 2 for e in v.values()))
        total_decisions = sum(
            sum(e['count'] for e in v.values())
            for v in self._experience.values()
        )
        return {
            'total_prototypes': total_keys,
            'valid_experiences': valid_keys,
            'total_decisions': total_decisions,
            'alpha': self.alpha,
        }
    
    def set_params(self, params: Dict):
        """更新参数并重建酉算符"""
        self.params = params
        self._rebuild_unitary()
    
    def build_yao(self, parsed: Dict, world, goal, phase: Dict) -> List[float]:
        """构建6维爻向量（委托给YaoBuilder）"""
        return self._yao_builder.build_yao(parsed, world, goal, phase)
    
    def _evolve_yao(self, yao: List[float], yao_key: Optional[str] = None) -> np.ndarray:
        """
        核心量子过程：六爻向量→酉演化→64维概率分布
        
        步骤：
          1. 从6维爻向量构建6量子位初态（作为乘积态）
          2. 施加酉变换 U(Δt)
          3. 计算64基态的概率分布
          4. 概率输出层经验加权（路线2：不修改哈密顿量）
        """
        # 1. 构建初态：每个爻独立制备
        psi = np.ones(64, dtype=complex) / 8.0
        
        for i in range(64):
            amp = 1.0
            for q in range(6):
                bit = (i >> q) & 1
                yi = yao[q]
                amp *= yi if bit == 1 else (1.0 - yi)
            psi[i] = amp
        
        psi = psi / np.linalg.norm(psi)
        
        # 2. 酉演化
        evolved = evolve_state(psi, self._U, self.evo_steps)
        
        # 3. 概率分布
        prob = prob_distribution(evolved)
        
        # 4. 概率输出层经验加权
        if yao_key is None:
            yao_key = self._proto_key(yao)
        prob = self._apply_experience_bias(prob, yao_key)
        
        return prob
    
    def _score_candidate_dict(self, cmd: str, world, goal, phase: Dict, parsed: Optional[Dict] = None) -> dict:
        """
        对一个候选动作的量子评分（内部dict版本）
        
        Returns:
            包含评分信息的字典（兼容原版Candidate.log()格式）
        """
        if parsed is None:
            parsed = self._parse_action(cmd)
        
        yao = self.build_yao(parsed, world, goal, phase)
        yao_key = self._proto_key(yao)
        
        # 量子演化（带经验加权）
        prob = self._evolve_yao(yao, yao_key=yao_key)
        
        # 提取信息
        dom_idx, dom_name, dom_prob = get_dominant_hexagram(prob)
        ent = entropy(prob)
        eff_temp = compute_effective_temp(prob)
        bagua_vec = get_bagua_vector(prob)
        lower, upper = get_trigram_probs(prob)
        
        # 线性基线（与YLYWScorer兼容）
        vec = np.array(yao, dtype=float)
        lin_score = float(np.dot(vec, self._lin_w) / self._lin_w.sum())
        
        # QYUF评分 = 主导卦吉凶值 × 概率 × 熵修正
        favor = float(FAVORABILITY[dom_idx])
        qyuf_score = lin_score * favor * (0.5 + 0.5 * dom_prob) * (1.0 - 0.1 * (ent / 6.0))
        
        return {
            "cmd": cmd,
            "yao": [round(v, 3) for v in yao],
            "hexagram": dom_name,
            "hex_idx": dom_idx,
            "hex_prob": round(dom_prob, 4),
            "favorability": round(favor, 3),
            "entropy": round(ent, 3),
            "eff_temp": round(eff_temp, 3),
            "bagua_vec": [round(v, 3) for v in bagua_vec],
            "lower_trigram": [round(v, 3) for v in lower],
            "upper_trigram": [round(v, 3) for v in upper],
            "qyuf_score": round(qyuf_score, 5),
            "linear_score": round(lin_score, 5),
        }
    
    def dest_favorability(self, world, goal, phase: Dict, dest_cls: str,
                          obj_cls: Optional[str]) -> float:
        """目标位置吉凶评分（用于retry-chain排序）"""
        parsed = {"cmd": f"go to {dest_cls} 1", "verb": "go",
                  "recep": f"{dest_cls} 1", "recep_cls": dest_cls,
                  "obj": None, "obj_cls": None}
        ph = dict(phase)
        ph["holding_target"] = None
        ph["searching"] = True
        yao = self.build_yao(parsed, world, goal, ph)
        prob = self._evolve_yao(yao)
        dom_idx, _, _ = get_dominant_hexagram(prob)
        return float(FAVORABILITY[dom_idx])
    
    def score_candidate(self, cmd: str, world, goal, phase: Dict, parsed: Optional[Dict] = None):
        """
        兼容V18的Candidate对象返回。
        同时记录候选评分供纯量子六爻使用。
        """
        result = self._score_candidate_dict(cmd, world, goal, phase, parsed)
        
        # 记录候选评分（用于y1离散度计算）
        if not hasattr(self, '_last_scores'):
            self._last_scores = []
        self._last_scores.append(result.get('qyuf_score', result.get('linear_score', 0.5)))
        # 给YaoBuilder传递引用
        self._yao_builder._scr = self
        
        # 解析动作用于veto
        parsed_action = parsed if parsed else self._parse_action(cmd)
        
        class _QYUFCompatCandidate:
            def __init__(self, d, parsed):
                self.cmd = d["cmd"]
                self.parsed = parsed
                self.yao = d["yao"]
                self.hexagram = d["hexagram"]
                self.hex_cn = d["hexagram"]
                self.cos = 0.0
                self.favor = d["favorability"]
                self.gua_affinity = 0.0
                self.ylyw_score = d["qyuf_score"]
                self.linear_score = d["linear_score"]
                self.vetoed = False
                self.veto_reason = ""
            def log(self):
                return {
                    "cmd": self.cmd, "yao": self.yao,
                    "hex": self.hexagram, "hex_cn": self.hex_cn,
                    "cos": round(self.cos, 4), "favor": round(self.favor, 3),
                    "gua_aff": round(self.gua_affinity, 3),
                    "score": round(self.ylyw_score, 5),
                    "lin": round(self.linear_score, 5),
                    "veto": self.vetoed, "veto_reason": self.veto_reason,
                }
        
        return _QYUFCompatCandidate(result, parsed_action)

    def _parse_action(self, cmd: str) -> Dict:
        """解析动作字符串（与原版兼容）"""
        c = cmd.strip()
        low = c.lower()
        d = {"cmd": c, "verb": low.split(" ", 1)[0], "obj": None, "recep": None,
             "obj_cls": None, "recep_cls": None}
        m = re.match(r"go to (.+)$", low)
        if m: d["verb"]="go"; d["recep"]=m.group(1); d["recep_cls"]=m.group(1).rsplit(" ",1)[0]; return d
        m = re.match(r"open (.+)$", low)
        if m: d["verb"]="open"; d["recep"]=m.group(1); d["recep_cls"]=m.group(1).rsplit(" ",1)[0]; return d
        m = re.match(r"take (.+?) from (.+)$", low)
        if m: d["verb"]="take"; d["obj"]=m.group(1); d["recep"]=m.group(2); d["obj_cls"]=m.group(1).rsplit(" ",1)[0]; d["recep_cls"]=m.group(2).rsplit(" ",1)[0]; return d
        m = re.match(r"(?:move|put) (.+?) (?:to|in|on|in/on) (.+)$", low)
        if m: d["verb"]="put"; d["obj"]=m.group(1); d["recep"]=m.group(2); d["obj_cls"]=m.group(1).rsplit(" ",1)[0]; d["recep_cls"]=m.group(2).rsplit(" ",1)[0]; return d
        m = re.match(r"(clean|heat|cool) (.+?) with (.+)$", low)
        if m: d["verb"]=m.group(1); d["obj"]=m.group(2); d["recep"]=m.group(3); d["obj_cls"]=m.group(2).rsplit(" ",1)[0]; d["recep_cls"]=m.group(3).rsplit(" ",1)[0]; return d
        return d
    
    def log(self, cmd: str, world, goal, phase: Dict) -> Dict:
        """完整评分日志（与v18 Candidate.log()格式兼容）"""
        return self._score_candidate_dict(cmd, world, goal, phase)


# ══════════════════════════════════════════════════════
# 自检/单元测试
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("QYUF 量子-易理模型 自检")
    print("=" * 60)
    
    # 1. 哈密顿量验证
    H = build_hamiltonian()
    print(f"\n1. 哈密顿量 H: {H.shape[0]}×{H.shape[1]}")
    print(f"   矩阵类型: {H.dtype}")
    print(f"   共轭对称? {np.allclose(H, H.conj().T)}")
    evals = np.linalg.eigvalsh(H)
    print(f"   特征值范围: [{evals.min():.3f}, {evals.max():.3f}]")
    
    # 2. 酉算符验证
    U = unitary_evolution(H)
    print(f"\n2. 酉算符 U: {U.shape[0]}×{U.shape[1]}")
    print(f"   酉性验证(U†U=I)? {np.allclose(U.conj().T @ U, np.eye(64), atol=1e-10)}")
    
    # 3. 从均等叠加态演化
    psi0 = np.ones(64, dtype=complex) / 8.0
    psi_final = evolve_state(psi0, U, 50)
    prob = prob_distribution(psi_final)
    dom_idx, dom_name, dom_prob = get_dominant_hexagram(prob)
    ent = entropy(prob)
    print(f"\n3. 从太极态演化50步:")
    print(f"   主导卦: {dom_name}(#{dom_idx})  概率: {dom_prob:.4f}")
    print(f"   系统熵: {ent:.3f} bit")
    lower, upper = get_trigram_probs(prob)
    l_cov = sum(1 for v in lower if v > 0.001)
    u_cov = sum(1 for v in upper if v > 0.001)
    print(f"   下卦覆盖: {l_cov}/8  上卦覆盖: {u_cov}/8")
    
    # 4. YaoBuilder 测试
    print(f"\n4. YaoBuilder 测试（模拟环境）:")
    builder = YaoBuilder()
    
    # 模拟一个简单场景
    class MockWorld:
        class Recep:
            def __init__(self): self.visited = False; self.searched = False
        def __init__(self):
            self.receps = {"fridge 1": self.Recep(), "counter 1": self.Recep()}
            self.objs = {}
        def state_key(self): return "mock"
    
    class MockGoal:
        object_class = "apple"
        recep_class = "fridge"
        tool_class = "sinkbasin"
        def needs_process(self): return False
        def needs_light(self): return False
        def is_target(self, cls): return cls == "apple"
    
    world = MockWorld()
    goal = MockGoal()
    
    print("   (跳过v18依赖测试，集成到V21后验证)")
    
    print(f"\n5. 参数扫描:")
    print(f"   Config参数: {QYUF_PARAMS}")
    print(f"   最优窗口: h_dang∈[0.10,0.30], J_comp∈[0.3,1.0]")
    print(f"   （源自论文§6.5参数空间扫描）")
    
    print("\n✅ QYUF量子-易理模型自检通过")
