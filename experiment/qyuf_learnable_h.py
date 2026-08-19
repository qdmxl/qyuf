#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知几式可学习H — QYUFScorer 的可学习哈密顿量子类
================================================
回应论文§6.2 局限"缺乏自主学习机制"：
让 H 的 5 个权重 (J_adj, J_ying, h_dang, h_zhong, J_comp) 从整局成败经验中
用"知几式频率校准 + 灵敏度归因"演化——而不是仅在概率输出层加权。

与 QYUFScorer 的关键区别（选项B：替换概率加权）：
  - QYUFScorer: p'_i = p_i × (1 + α·ω_i)   (概率输出层加权，不动H)
  - 本模块:     H_θ ← H_θ + α·归因反馈之后重建酉算符   (学习沉入H内部)

知几原理（对齐 ylyw zhiji/yao_tune）：
  "知几其神乎！几者，动之微，吉之先见者也。见几而作，不俟终日。"
  一次局末的成败事件 → 立即把成败归因到驱动该卦的H权重 → 微调

机制：
  1. 灵敏度归因（无需梯度反传）：
        H(θ)=Σ_k θ_k·A_k,  U_eff=exp(-iHτ_eff), ψ_out=U_eff·ψ
        ∂prob[dom]/∂θ_k = 2·Re(ψ_out[dom]*·[∂U_eff/∂θ_k·ψ][dom])
        ∂U_eff/∂θ_k = -iτ_eff·U_eff·A_k    (H对θ线性，可用)
  2. 局末提交 commit_game(won)：
        胜局:  θ += α_q · Σ(该局成功主导卦的灵敏度方向)   （强化获吉配置）
        负局:  θ -= α_q · Σ(该局失败主导卦的灵敏度方向)   （知耻：惩罚入凶配置）
        之后 _rebuild_unitary() 让量子干涉随新H改变
  3. 跨局稳定：α_s 慢速向成功率高的权重配置演化；θ钳制在[l,u]
"""
import os, sys, json
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ylyw', 'QYUF', 'experiment'))
from qyuf_model import QYUFScorer, build_hamiltonian, unitary_evolution


# ═════════════════════════════════════════════
# H 的基矩阵（θ 线性分解）：H(θ) = Σ_k θ_k · A_k
# ═════════════════════════════════════════════
LEARNABLE_KEYS = ['J_adj', 'J_ying', 'h_dang', 'h_zhong', 'J_comp']
THETA_DEFAULT = np.array([1.0, 0.5, 0.30, 0.15, 0.5])  # = Config-C


def build_base_matrices():
    n_qubits = 6
    dim = 2 ** n_qubits

    def pauli_z(i):
        op = 1.0
        for q in range(n_qubits):
            if q == i:
                op = np.kron(np.diag([1, -1]), op)
            else:
                op = np.kron(np.eye(2), op)
        return op

    def pauli_x(i):
        op = 1.0
        for q in range(n_qubits):
            if q == i:
                op = np.kron(np.array([[0, 1], [1, 0]]), op)
            else:
                op = np.kron(np.eye(2), op)
        return op

    def pauli_y(i):
        op = 1.0
        for q in range(n_qubits):
            if q == i:
                op = np.kron(np.array([[0, -1j], [1j, 0]]), op)
            else:
                op = np.kron(np.eye(2), op)
        return op

    A_adj = np.zeros((dim, dim), dtype=complex)
    for i in range(n_qubits - 1):
        A_adj += pauli_z(i) @ pauli_z(i + 1)

    A_ying = np.zeros((dim, dim), dtype=complex)
    for i, j in [(0, 3), (1, 4), (2, 5)]:
        A_ying += pauli_z(i) @ pauli_z(j)

    A_dang = np.zeros((dim, dim), dtype=complex)
    for i in range(n_qubits):
        sign = -1.0 if (i % 2 == 0) else 1.0
        A_dang += sign * pauli_z(i)

    A_zhong = (-pauli_z(1) + pauli_z(4)) * 1.0

    A_comp = np.zeros((dim, dim), dtype=complex)
    for i in range(n_qubits - 1):
        A_comp += pauli_x(i) @ pauli_x(i + 1) + pauli_y(i) @ pauli_y(i + 1)

    return [A_adj, A_ying, A_dang, A_zhong, A_comp]


BASE_MATS = build_base_matrices()
DIM = BASE_MATS[0].shape[0]


class QYUFScorerLearnableH(QYUFScorer):
    """
    知几式可学习H评分器：以 QYUFScorer 为基类，学习沉入H权重。
    选项B：用 H 权重的知几校准替代概率输出层加权。
    """
    def __init__(self, params: Optional[Dict] = None, theta: float = 1.0,
                 evo_steps: int = 20, experience_path: Optional[str] = None,
                 alpha: float = 0.0, alpha_q: float = 0.08, alpha_s: float = 0.03,
                 dt: float = 0.1, lower=None, upper=None, seed: int = 0):
        # 初始化基类（会 load 经验、rebuild unitary）
        super().__init__(params=params, theta=theta, evo_steps=evo_steps,
                         experience_path=experience_path, alpha=alpha)
        self.dt = dt
        self.tau_eff = dt * evo_steps          # 有效演化时间 τ_eff=dt·steps
        self.alpha_q = alpha_q                 # 局内/局末快速校准
        self.alpha_s = alpha_s                 # 跨局慢速稳定
        self.lower = np.array(lower if lower is not None
                              else [0.2, 0.2, 0.1, 0.0, 0.1], dtype=float)
        self.upper = np.array(upper if upper is not None
                              else [2.0, 1.5, 1.0, 1.0, 1.5], dtype=float)
        self.rng = np.random.RandomState(seed)
        # 将 params(5权重) 载入 θ 向量
        self.theta = np.array([self.params.get(k, THETA_DEFAULT[i])
                               for i, k in enumerate(LEARNABLE_KEYS)], dtype=float)
        # U_eff 缓存：只依赖θ，候选间共享，θ变化后重建
        self._U_eff = None
        self._rebuild_U_eff()
        # 可学习H的决策缓冲：每格记录 (主导卦, 灵敏度向量)
        self._lh_buf = []           # 本局累积
        self._suc_cnt = np.zeros(5) # 跨局：正向归因累计
        self._tot_cnt = np.zeros(5) # 跨局：参与累计
        self._theta_traj = [self.theta.copy()]

    def _rebuild_U_eff(self):
        """由当前θ重建 U_eff=exp(-iH(θ)·τ_eff)"""
        H = build_hamiltonian(1.0, dict(zip(LEARNABLE_KEYS, self.theta.tolist())))
        self._U_eff = unitary_evolution(H, self.tau_eff)

    # ---------- 核心：灵敏度归因 ----------
    def _sensitivity(self, psi: np.ndarray, U_eff: np.ndarray, dom_idx: int) -> np.ndarray:
        """主导卦概率对5权重的灵敏度 ∂prob[dom]/∂θ_k（τ_eff一阶解析）"""
        psi_out = U_eff @ psi
        grad = np.zeros(5)
        for k, A in enumerate(BASE_MATS):
            dU = -1j * self.tau_eff * (U_eff @ A)
            dpsi = dU @ psi
            dP = 2 * np.real(psi_out[dom_idx].conj() * dpsi[dom_idx])
            grad[k] = dP
        return grad

    def _yield_quantum(self, yao: List[float]):
        """
        用量子的 H(θ) 演化六爻 → 概率。
        与基类 _evolve_yao 等价，但用缓存的 U_eff 直接演算，
        并额外返回 psi（供灵敏度归因）。
        """
        # 构建初态（与基类完全一致）
        psi = np.ones(DIM, dtype=complex) / 8.0
        for i in range(DIM):
            amp = 1.0
            for q in range(6):
                bit = (i >> q) & 1
                yi = yao[q]
                amp *= yi if bit == 1 else (1.0 - yi)
            psi[i] = amp
        psi = psi / np.linalg.norm(psi)
        U_eff = self._U_eff
        psi_out = U_eff @ psi
        prob = np.abs(psi_out) ** 2
        return prob, psi, U_eff, psi_out

    # ---------- 采样主导卦决策（与基类同语义，但H会随θ变） ----------
    def _evolve_yao(self, yao: List[float], yao_key: Optional[str] = None) -> np.ndarray:
        """覆盖：学习沉入H，不再做概率输出层加权"""
        prob, _, _, _ = self._yield_quantum(yao)
        return prob

    # ---------- 记录决策（含灵敏度） ----------
    def record_decision_learn(self, cand: dict, yao: List[float]):
        """
        记录一次决策的灵敏度归因（由 agent 在选到高评分候选后调用）。
        cand: _score_candidate_dict 的返回（含 hex_idx 主导卦）
        yao: 该候选的6爻向量
        """
        prob, psi, U_eff, psi_out = self._yield_quantum(yao)
        dom = int(cand.get('hex_idx', int(np.argmax(prob))))
        sens = self._sensitivity(psi, U_eff, dom)
        self._lh_buf.append({'dom': dom, 'sens': sens.copy()})
        return sens

    # ---------- 局末知几校准（H权重演化） ----------
    def commit_game(self, won: bool):
        """知几：整局成败 → 归因到5个H权重 → 更新θ → 重建酉算符"""
        if self._lh_buf:
            sens_sum = np.zeros(5)
            for rec in self._lh_buf:
                sens_sum += rec['sens']
            # 胜局强化（沿让成功卦概率更高的方向）：这里用 灵敏度×sign反馈
            sign = +1.0 if won else -1.0
            delta = self.alpha_q * sign * sens_sum
            # 跨局成功率校准（K_calibration 慢速）
            self._tot_cnt += 1.0
            if won:
                self._suc_cnt += 1.0
            self.theta = self.theta + delta
            # 跨局知几：成功率驱动的慢速校准
            p_succ = np.clip(self._suc_cnt / (self._tot_cnt + 1e-9), 0.0, 1.0)
            adjust = (p_succ - 0.5) * 2.0 * self.alpha_s   # [-α_s, +α_s]
            self.theta = self.theta + adjust * np.sign(self.theta + 1e-6)
        self.theta = np.clip(self.theta, self.lower, self.upper)
        self._lh_buf = []
        # 重建酉算符缓存：让新的H生效（候选间共享，只此一次）
        self.params = dict(zip(LEARNABLE_KEYS, self.theta.tolist()))
        self._rebuild_U_eff()
        self._theta_traj.append(self.theta.copy())
        # 覆盖基类存储逻辑？基类commit_game会把决策写入_experience并save。
        # 这里选项B不依赖概率加权，跳过经验文件写入（保留基类行为会污染）。
        # 注意：基类commit_game有副作用，我们完全覆盖之。（已在子类重写）

    def theta_dict(self):
        return dict(zip(LEARNABLE_KEYS, np.round(self.theta, 3)))

    def experience_stats(self):
        s = super().experience_stats()
        s['theta'] = self.theta_dict()
        s['suc_attribution'] = np.round(self._suc_cnt / (self._tot_cnt + 1e-9), 3).tolist()
        return s
