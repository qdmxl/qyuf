# QYUF — Quantum-Yili Unified Framework

量子易理统一框架：严格"八卦相荡"酉变换的实现。

## 版本历史

| 版本 | 文件 | 描述 |
|------|------|------|
| v1.0 | `qyuf_final.py` | Grover振幅放大涌现（Oracle=固定易理评分） |
| v0.4 | `qyuf_prototype_v4.py` | 量子势能涌现（Bohm势能引导概率流） |
| v2.0 | `qyuf_adaptive_oracle.py` | 感知自适应Oracle（Oracle随物体特征变化） |
| v3.0 | `qyuf_multifeature.py` | 多特征分形模型（每维特征独立匹配八卦） |
| v3.5 | `qyuf_multifeature_grover.py` | 多特征分形 + Grover振幅放大 |
| **v4.0** | **`qyuf_strict_unitary.py`** | **严格八卦相荡酉变换（论文对应版本）** |

## 核心理念

- **特征分形**：物的每个特征独立匹配不同八卦，各有匹配系数
- **八卦相荡**：$U_{\text{摩}} = e^{-iH\tau}$ 在 $\mathcal{H}_3$ 上实现概率幅干涉
- **上/下卦独立**：显性/隐性特征分别驱动上卦和下卦的"相荡"
- **张量积涌现**：$(U_{\text{摩}} \otimes U_{\text{摩}}) |\Psi_0\rangle$

## 论文

`易理探讨15-八卦相荡.docx` — 《易经》生成逻辑与量子干涉的严格数学对应
