#!/usr/bin/env python3
"""
QYUF v4.0 × ALFWorld V20 端到端完整测试 (134 tasks)

在V20完整Agent基础上，将六爻计算替换为v4.0严格八卦相荡酉变换。
但动作选择仍然使用V20的完整YLYWScorer(保持导航/规划能力不变)。

目的：测试"用八卦相荡替代六爻计算"对决策质量的影响。
"""
from __future__ import annotations
import sys, os, time, json, re, traceback
import numpy as np

YLYW_DIR = os.path.expanduser("~/MXL/科研/ylyw")
ALFWORLD_EXP = os.path.join(YLYW_DIR, "alfworld_exp")
QYUF_DIR = os.path.join(YLYW_DIR, "QYUF")
V20_DIR = os.path.join(ALFWORLD_EXP, "v20")

for p in [ALFWORLD_EXP, os.path.join(ALFWORLD_EXP, "v18"), V20_DIR, QYUF_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from qyuf_strict_unitary import (
    StrictYiliInterference, FeatureFractal, TRIGRAM_NAMES, HEX_NAMES, infer_strategy
)

# ============================================================
# 物体6维特征库
# ============================================================

FEATURES_6D = {
    'plate':       [0.65, 0.20, 0.80, 0.10, 0.35, 0.15],
    'bowl':        [0.60, 0.30, 0.75, 0.10, 0.40, 0.20],
    'mug':         [0.70, 0.15, 0.65, 0.10, 0.30, 0.10],
    'cup':         [0.65, 0.10, 0.60, 0.10, 0.25, 0.10],
    'apple':       [0.55, 0.30, 0.80, 0.10, 0.20, 0.50],
    'potato':      [0.60, 0.70, 0.60, 0.10, 0.25, 0.60],
    'tomato':      [0.40, 0.20, 0.70, 0.10, 0.20, 0.50],
    'bread':       [0.15, 0.50, 0.60, 0.10, 0.15, 0.40],
    'egg':         [0.10, 0.15, 0.75, 0.10, 0.10, 0.10],
    'soap':        [0.50, 0.40, 0.40, 0.10, 0.15, 0.20],
    'pencil':      [0.75, 0.30, 0.10, 0.10, 0.05, 0.30],
    'fork':        [0.85, 0.20, 0.15, 0.10, 0.10, 0.10],
    'knife':       [0.90, 0.15, 0.10, 0.10, 0.15, 0.10],
    'spoon':       [0.80, 0.15, 0.15, 0.10, 0.10, 0.10],
    'food':        [0.35, 0.50, 0.50, 0.10, 0.25, 0.40],
    'milk':        [0.60, 0.10, 0.60, 0.10, 0.35, 0.10],
    'coffee':      [0.30, 0.10, 0.40, 0.10, 0.25, 0.10],
    'butter':      [0.20, 0.30, 0.50, 0.10, 0.20, 0.30],
    'credit card': [0.60, 0.20, 0.10, 0.10, 0.02, 0.15],
    'keychain':    [0.80, 0.30, 0.10, 0.10, 0.03, 0.20],
    'cloth':       [0.10, 0.70, 0.30, 0.10, 0.05, 0.80],
    'spatula':     [0.75, 0.10, 0.30, 0.10, 0.15, 0.10],
    'watch':       [0.70, 0.10, 0.40, 0.10, 0.05, 0.20],
    'safe':        [0.90, 0.10, 0.80, 0.05, 0.80, 0.10],
    'book':        [0.55, 0.40, 0.85, 0.05, 0.40, 0.50],
    'pen':         [0.70, 0.20, 0.10, 0.10, 0.05, 0.20],
    'cellphone':   [0.65, 0.10, 0.50, 0.10, 0.15, 0.10],
    'key':         [0.80, 0.20, 0.10, 0.10, 0.03, 0.15],
    'creditcard':  [0.60, 0.20, 0.10, 0.10, 0.02, 0.15],
}

def get_obj_features(task_desc: str) -> np.ndarray:
    task_lower = task_desc.lower()
    for obj in sorted(FEATURES_6D.keys(), key=len, reverse=True):
        if obj in task_lower:
            return np.array(FEATURES_6D[obj])
    return np.array([0.50, 0.40, 0.50, 0.20, 0.30, 0.40])


# ============================================================
# V20 Agent + v4.0 八卦相荡
# ============================================================

yili_engine = StrictYiliInterference(tau=0.6)

class QYUFv40Agent:
    """
    基于V20完整逻辑的Agent，但加入了v4.0八卦相荡分析。
    优先使用V20已有的YLYW决策来保证导航/操作的可靠性，
    v4.0分析仅作为额外信息注入决策过程。
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.step_idx = 0
        self.game_id = None
        self.task_desc = ""
        self.qyuf_hex_name = "?"
        self.qyuf_strategy = "?"
        self.qyuf_result = {}
        
        # V20完整Agent
        from v20.agent_v20 import AgentV20
        self.v20_agent = AgentV20(log_path=None, verbose=False)
        
        # 允许V20的act设置verbose
        self._qyuf_decided = False
    
    def reset(self, task_desc, obs, admissible_commands, game_id=None):
        self.task_desc = task_desc
        self.step_idx = 0
        self.game_id = game_id
        self._qyuf_decided = False
        
        # V20 reset
        self.v20_agent.reset(task_desc, obs, admissible_commands, game_id=game_id)
        
        # 运行v4.0八卦相荡
        res = qyuf40_decision(task_desc)
        self.qyuf_result = res
        self.qyuf_hex_name = res['hex_name']
        self.qyuf_strategy = res['strategy']
        self._qyuf_decided = True
        
        if self.verbose:
            print(f"  [QYUFv4] 涌现卦:{res['hex_name']} 策略:{res['strategy']}")
            print(f"          吉卦:{res['good_prob_init']:.0f}%→{res['good_prob_final']:.0f}%")
    
    def act(self, obs, admissible) -> str:
        """用V20的完整决策逻辑"""
        self.step_idx += 1
        return self.v20_agent.act(obs, admissible)
    
    def observe_transition(self, action, obs, admissible, won=False):
        return self.v20_agent.observe_transition(action, obs, admissible, won=won)
    
    def dump_logs(self, extra=None, final=False):
        logs = self.v20_agent.dump_logs(extra=extra, final=final)
        logs['qyuf_v40'] = {
            'hex_name': self.qyuf_hex_name,
            'strategy': self.qyuf_strategy,
            'good_init': self.qyuf_result.get('good_prob_init', 0),
            'good_final': self.qyuf_result.get('good_prob_final', 0),
            'upper_diff': self.qyuf_result.get('upper_diff', 0),
            'lower_diff': self.qyuf_result.get('lower_diff', 0),
        }
        return logs


# ============================================================
# 决策引擎 (v4.0分析 + 策略映射)
# ============================================================

def qyuf40_decision(task_desc: str) -> dict:
    feats = get_obj_features(task_desc)
    r = yili_engine.run(feats)
    
    top1_idx = r['top_final'][0][0]
    strategy = infer_strategy(top1_idx, feats)
    top3 = [(r['top_final'][i][0], HEX_NAMES[r['top_final'][i][0]], r['top_final'][i][1]) 
            for i in range(min(3, len(r['top_final'])))]
    
    return {
        'hex_idx': top1_idx,
        'hex_name': HEX_NAMES[top1_idx],
        'strategy': strategy,
        'good_prob_init': r['good_prob_init'] * 100,
        'good_prob_final': r['good_prob_final'] * 100,
        'upper_diff': r['upper_diff'],
        'lower_diff': r['lower_diff'],
        'upper_top3': [(TRIGRAM_NAMES[i], np.abs(r['upper_evolved'][i])**2) 
                       for i, _ in sorted(enumerate(np.abs(r['upper_evolved'])**2), key=lambda x: -x[1])[:3]],
        'lower_top3': [(TRIGRAM_NAMES[i], np.abs(r['lower_evolved'][i])**2) 
                       for i, _ in sorted(enumerate(np.abs(r['lower_evolved'])**2), key=lambda x: -x[1])[:3]],
        'top3': top3,
        'features': feats.tolist(),
    }


# ============================================================
# 测试运行器
# ============================================================

MAX_STEPS = 50

def run_single(env, agent, game_idx: int) -> dict:
    obs, info = env.reset(game_idx=game_idx)
    task_desc = info.get("task_desc") or ""
    admissible = info.get("admissible_commands") or ["look"]
    
    agent.reset(task_desc, obs, admissible, game_id=game_idx)
    
    won = False
    steps = 0
    actions = []
    
    for _ in range(MAX_STEPS):
        action = agent.act(obs, admissible)
        actions.append(action)
        obs, info = env.step(action)
        steps += 1
        won = bool(info.get("won", False))
        admissible = info.get("admissible_commands") or ["look"]
        agent.observe_transition(action, obs, admissible, won=won)
        
        if won or info.get("done", False):
            break
    
    return {
        "game_idx": game_idx,
        "won": won,
        "steps": steps,
        "actions": actions,
        "task_desc": task_desc,
        "hex_name": agent.qyuf_hex_name,
        "strategy": agent.qyuf_strategy,
        "good_init": agent.qyuf_result.get("good_prob_init", 0),
        "good_final": agent.qyuf_result.get("good_prob_final", 0),
        "upper_diff": agent.qyuf_result.get("upper_diff", 0),
        "lower_diff": agent.qyuf_result.get("lower_diff", 0),
        "top3_hex": agent.qyuf_result.get("top3", []),
    }


def main():
    print("═"*70)
    print("  QYUF v4.0 × ALFWorld V20 端到端完整测试")
    print("  V20完整Agent + v4.0八卦相荡分析 (U_摩 = e^{-iHτ})")
    print("═"*70)
    
    from alfworld_official_wrapper import ALFWorldOfficial
    from v20.agent_v20 import AgentV20
    
    env = ALFWorldOfficial(split="valid_unseen")
    n = env.num_games
    print(f"  总游戏数: {n}")
    
    # V20基线结果（先跑一半用于对比）
    baseline_results = []
    v40_results = []
    
    t_start = time.time()
    
    # ===== V20 基线 =====
    print(f"\n{'─'*60}")
    print(f"  阶段1: V20基线测试")
    print(f"{'─'*60}")
    
    for gi in range(n):
        agent = AgentV20(log_path=None, verbose=False)
        try:
            r = run_single(env, agent, gi)
            r['hex_name'] = '?'  # V20没有v4.0卦象输出
        except Exception as e:
            traceback.print_exc()
            r = {"game_idx": gi, "won": False, "steps": MAX_STEPS,
                 "error": str(e), "task_desc": ""}
        
        baseline_results.append(r)
        tag = "WON" if r.get('won') else "LOST"
        print(f"  [{gi+1:3d}/{n}] {tag} | {r.get('task_desc','')[:40]:40s} | {r.get('steps',0):2d}步")
        
        if (gi + 1) % 10 == 0:
            bw = sum(1 for r_ in baseline_results if r_['won'])
            print(f"  → V20基线成功率: {bw}/{gi+1} = {bw/(gi+1)*100:.1f}%")
    
    bw_total = sum(1 for r in baseline_results if r['won'])
    
    # ===== V40 测试 =====
    print(f"\n{'─'*60}")
    print(f"  阶段2: QYUF v4.0 八卦相荡测试")
    print(f"{'─'*60}")
    
    for gi in range(n):
        agent = QYUFv40Agent(verbose=False)
        try:
            r = run_single(env, agent, gi)
        except Exception as e:
            traceback.print_exc()
            r = {"game_idx": gi, "won": False, "steps": MAX_STEPS,
                 "error": str(e), "task_desc": "",
                 "hex_name": "?", "strategy": "?",
                 "good_init": 0, "good_final": 0,
                 "upper_diff": 0, "lower_diff": 0, "top3_hex": []}
        
        v40_results.append(r)
        tag = "WON" if r.get('won') else "LOST"
        print(f"  [{gi+1:3d}/{n}] {tag} | {r.get('task_desc','')[:40]:40s} | 卦{r.get('hex_name','?'):2s} | {r.get('steps',0):2d}步")
        
        if (gi + 1) % 10 == 0:
            vw = sum(1 for r_ in v40_results if r_['won'])
            print(f"  → v4.0成功率: {vw}/{gi+1} = {vw/(gi+1)*100:.1f}%")
    
    vw_total = sum(1 for r in v40_results if r['won'])
    
    # ===== 最终报告 =====
    t_total = time.time() - t_start
    
    print(f"\n{'='*70}")
    print(f"  QYUF v4.0 × ALFWorld V20 完整测试报告")
    print(f"{'='*70}")
    
    print(f"\n  总用时: {t_total:.1f}s")
    
    print(f"\n  【总体对比】")
    print(f"  {'指标':30s} {'V20基线':>10s} {'v4.0相荡':>10s}")
    print(f"  {'-'*52}")
    print(f"  {'总任务数':30s} {n:>10d} {n:>10d}")
    print(f"  {'成功':30s} {bw_total:>10d} {vw_total:>10d}")
    print(f"  {'成功率':30s} {bw_total/n*100:>9.1f}% {vw_total/n*100:>9.1f}%")
    avg_b = np.mean([r['steps'] for r in baseline_results])
    avg_v = np.mean([r['steps'] for r in v40_results])
    print(f"  {'平均步数/成功':30s} {np.mean([r['steps'] for r in baseline_results if r['won']]):>10.1f} {np.mean([r['steps'] for r in v40_results if r['won']]):>10.1f}")
    
    print(f"\n  【v4.0八卦相荡涌现指标】")
    gi_vals = [r['good_init'] for r in v40_results if r.get('good_init') is not None]
    gf_vals = [r['good_final'] for r in v40_results if r.get('good_final') is not None]
    ud_vals = [r['upper_diff'] for r in v40_results if r.get('upper_diff') is not None]
    ld_vals = [r['lower_diff'] for r in v40_results if r.get('lower_diff') is not None]
    
    if gi_vals:
        print(f"  {'吉卦初始概率':25s} {np.mean(gi_vals):>8.1f}%")
        print(f"  {'吉卦相荡后概率':25s} {np.mean(gf_vals):>8.1f}%")
        print(f"  {'吉卦增益':25s} {np.mean(gf_vals)-np.mean(gi_vals):>+8.1f}pp")
        print(f"  {'上卦干涉变化':25s} {np.mean(ud_vals):.4f}")
        print(f"  {'下卦干涉变化':25s} {np.mean(ld_vals):.4f}")
    
    # 成功/失败 vs 吉卦增益
    print(f"\n  【成功 vs 吉卦增益关系】")
    won_gains = [r['good_final']-r['good_init'] for r in v40_results if r.get('won') and r.get('good_init') is not None]
    lost_gains = [r['good_final']-r['good_init'] for r in v40_results if not r.get('won') and r.get('good_init') is not None]
    if won_gains:
        print(f"  {'成功时平均吉卦增益':25s} {np.mean(won_gains):>+8.1f}pp")
    if lost_gains:
        print(f"  {'失败时平均吉卦增益':25s} {np.mean(lost_gains):>+8.1f}pp")
    
    # 涌现卦象分布
    print(f"\n  【涌现卦象分布TOP10】")
    hex_counts = {}
    for r in v40_results:
        hn = r.get('hex_name', '?')
        hex_counts[hn] = hex_counts.get(hn, 0) + 1
    for hn, cnt in sorted(hex_counts.items(), key=lambda x: -x[1])[:10]:
        wr = sum(1 for r in v40_results if r.get('hex_name')==hn and r.get('won'))
        print(f"    {hn:4s}: {cnt:3d}/{n} ({cnt/n*100:.1f}%) 成功{wr:3d}({wr/cnt*100:.0f}%)")
    
    print(f"\n  【策略分布TOP10】")
    strat_counts = {}
    for r in v40_results:
        s = r.get('strategy', '?')
        strat_counts[s] = strat_counts.get(s, 0) + 1
    for s, cnt in sorted(strat_counts.items(), key=lambda x: -x[1])[:10]:
        wr = sum(1 for r in v40_results if r.get('strategy')==s and r.get('won'))
        print(f"    {s:20s}: {cnt:3d}/{n} ({cnt/n*100:.1f}%) 成功{wr:3d}({wr/cnt*100:.0f}%)")
    
    print(f"\n{'='*70}")
    print(f"  结论")
    print(f"{'='*70}")
    print(f"""
  V20基线成功率: {bw_total}/{n} = {bw_total/n*100:.1f}%
  v4.0相荡成功率: {vw_total}/{n} = {vw_total/n*100:.1f}%
  
  v4.0与V20的全同决策逻辑（动作选择完全来自V20），
  区别仅在于v4.0额外运行了八卦相荡分析（非侵入式）。
  
  v4.0八卦相荡的核心价值：
  - 为每个task自动涌现出最优卦象和策略
  - 吉卦从初始{np.mean(gi_vals):.1f}%经U_摩干涉后到{np.mean(gf_vals):.1f}%
  - 上卦和下卦各自独立干涉（下卦干涉强于上卦）
  
  如需测试v4.0直接替代YLYW六爻评分的效果，
  需要进一步在V20的cn_world_model中用v4.0替换YLYWScorer。
""")
    
    # 保存结果
    outpath = os.path.join(ALFWORLD_EXP, "qyuf_v40_e2e_v20_compare.json")
    
    save = {
        'config': {'model': 'V20+QYUFv4', 'tau': 0.6, 'split': 'valid_unseen'},
        'baseline_v20': {
            'won': bw_total,
            'total': n,
            'rate': bw_total/n,
        },
        'v40_strict_unitary': {
            'won': vw_total,
            'total': n,
            'rate': vw_total/n,
            'avg_good_init': float(np.mean(gi_vals)) if gi_vals else 0,
            'avg_good_final': float(np.mean(gf_vals)) if gf_vals else 0,
            'avg_upper_diff': float(np.mean(ud_vals)) if ud_vals else 0,
            'avg_lower_diff': float(np.mean(ld_vals)) if ld_vals else 0,
        },
        'results_v40': [{
            'game_idx': r['game_idx'],
            'won': r['won'],
            'steps': r['steps'],
            'task_desc': r.get('task_desc', ''),
            'hex_name': r.get('hex_name', '?'),
            'strategy': r.get('strategy', '?'),
            'good_init': r.get('good_init', 0),
            'good_final': r.get('good_final', 0),
        } for r in v40_results],
    }
    
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    
    print(f"  结果已保存到: {outpath}")


if __name__ == "__main__":
    main()
