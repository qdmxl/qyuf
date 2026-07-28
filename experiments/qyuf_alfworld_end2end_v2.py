#!/usr/bin/env python3
"""
QYUF × ALFWorld V20 端到端测试 (v2)
====================================
复用YLYW V4的动作执行逻辑，只替换选卦决策这一步。

实验设计：
  - 原始YLYW: 用汉语引擎+六爻选卦 → 动作
  - QYUF量子: 用物体特征→Grover涌现选卦 → 动作
  - 对比: 完成率、步数
"""

import sys, os, re, math, time
import numpy as np

YLYW_DIR = os.path.expanduser("~/MXL/科研/ylyw")
YLYW_CORE = os.path.join(YLYW_DIR, "api_docs", 'ylyw_core')
ALFWORLD_EXP = os.path.join(YLYW_DIR, "alfworld_exp")
QYUF_SRC = os.path.join(YLYW_DIR, "QYUF", "src")

for p in [ALFWORLD_EXP, YLYW_CORE, QYUF_SRC, YLYW_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from alfworld_official_wrapper import ALFWorldOfficial
from trigram_base import TrigramBase
from qyuf_core import QYUF, HEXAGRAM_NAMES, STRATEGY_MAP


# ========== 物体特征库 ==========
OBJECT_FEATURES = {
    'plate':   {'weight':0.35,'hardness':0.65,'size':0.55,'fragility':0.40},
    'bowl':    {'weight':0.40,'hardness':0.60,'size':0.65,'fragility':0.45},
    'mug':     {'weight':0.30,'hardness':0.70,'size':0.40,'fragility':0.50},
    'cup':     {'weight':0.25,'hardness':0.65,'size':0.35,'fragility':0.55},
    'apple':   {'weight':0.20,'hardness':0.55,'size':0.30,'fragility':0.35},
    'potato':  {'weight':0.25,'hardness':0.60,'size':0.35,'fragility':0.25},
    'tomato':  {'weight':0.20,'hardness':0.40,'size':0.30,'fragility':0.70},
    'bread':   {'weight':0.15,'hardness':0.15,'size':0.45,'fragility':0.20},
    'egg':     {'weight':0.10,'hardness':0.10,'size':0.15,'fragility':0.90},
    'soap':    {'weight':0.15,'hardness':0.50,'size':0.25,'fragility':0.30},
    'pencil':  {'weight':0.05,'hardness':0.75,'size':0.10,'fragility':0.20},
    'fork':    {'weight':0.10,'hardness':0.85,'size':0.15,'fragility':0.10},
    'knife':   {'weight':0.15,'hardness':0.90,'size':0.20,'fragility':0.10},
    'spoon':   {'weight':0.10,'hardness':0.80,'size':0.15,'fragility':0.10},
    'food':    {'weight':0.25,'hardness':0.35,'size':0.40,'fragility':0.50},
    'coffee':  {'weight':0.25,'hardness':0.30,'size':0.20,'fragility':0.60},
    'milk':    {'weight':0.35,'hardness':0.60,'size':0.40,'fragility':0.40},
    'butter':  {'weight':0.20,'hardness':0.20,'size':0.30,'fragility':0.30},
}

def get_features(obj_name):
    obj_key = re.sub(r'\s+\d+$','',obj_name.lower().strip())
    for k in sorted(OBJECT_FEATURES, key=len, reverse=True):
        if k in obj_key:
            return OBJECT_FEATURES[k]
    return OBJECT_FEATURES['food']


# ========== 经典YLYW决策 ==========
trigram_base = TrigramBase()
qyuf = QYUF(good_threshold=0.0)

def classic_decision(features):
    memberships = trigram_base.get_all_memberships(features)
    scores = np.zeros(64)
    for idx in range(64):
        scores[idx] = memberships[idx>>3] * memberships[idx&0x7]
    best = int(np.argmax(scores))
    return HEXAGRAM_NAMES[best], STRATEGY_MAP.get(HEXAGRAM_NAMES[best],"standard_grasp")

def quantum_decision(features, iters=1):
    memberships = trigram_base.get_all_memberships(features)
    psi = np.zeros(64, dtype=complex)
    for idx in range(64):
        u, l = idx>>3, idx&0x7
        m = memberships[u]*memberships[l] + 0.01
        sb = 0.8 + (qyuf.scores[idx]+8)/22*0.4
        psi[idx] = math.sqrt(m) * sb
    psi /= np.linalg.norm(psi)
    psi = qyuf.amplify(psi, iters)
    best = int(np.argmax(qyuf.prob(psi)))
    return HEXAGRAM_NAMES[best], STRATEGY_MAP.get(HEXAGRAM_NAMES[best],"standard_grasp")


# ========== 卦象→动作参数 ==========
HEX_ACTION_PARAMS = {
    'power_grasp':        {'explore':'fast',   'take_priority':'direct','speed':'fast'},
    'strong_grasp':       {'explore':'normal', 'take_priority':'direct','speed':'fast'},
    'precision_grasp':    {'explore':'slow',   'take_priority':'careful','speed':'slow'},
    'precise_grasp':      {'explore':'slow',   'take_priority':'careful','speed':'slow'},
    'gentle_grasp':       {'explore':'slow',   'take_priority':'careful','speed':'slow'},
    'soft_grasp':         {'explore':'slow',   'take_priority':'careful','speed':'slow'},
    'cautious_grasp':     {'explore':'slow',   'take_priority':'explore','speed':'slow'},
    'adaptive_grasp':     {'explore':'normal', 'take_priority':'explore','speed':'medium'},
    'balanced_grasp':     {'explore':'normal', 'take_priority':'direct','speed':'medium'},
    'risky_grasp':        {'explore':'fast',   'take_priority':'direct','speed':'fast'},
    'biting_grasp':       {'explore':'normal', 'take_priority':'direct','speed':'medium'},
    'decorative_grasp':   {'explore':'slow',   'take_priority':'careful','speed':'medium'},
    'extrication_grasp':  {'explore':'slow',   'take_priority':'careful','speed':'slow'},
    'monitoring_grasp':   {'explore':'normal', 'take_priority':'careful','speed':'medium'},
    'tactile_feedback_grasp':{'explore':'slow','take_priority':'careful','speed':'slow'},
    'compliant_grasp':    {'explore':'slow',   'take_priority':'careful','speed':'slow'},
    'standard_grasp':     {'explore':'normal', 'take_priority':'direct','speed':'medium'},
    'retry_grasp':        {'explore':'normal', 'take_priority':'explore','speed':'medium'},
    'top_down_grasp':     {'explore':'normal', 'take_priority':'direct','speed':'medium'},
    'quick_grasp':        {'explore':'fast',   'take_priority':'direct','speed':'fast'},
    'corrective_grasp':   {'explore':'normal', 'take_priority':'direct','speed':'medium'},
    'dynamic_grasp':      {'explore':'fast',   'take_priority':'direct','speed':'fast'},
    'stable_grasp':       {'explore':'normal', 'take_priority':'direct','speed':'medium'},
    'progressive_grasp':  {'explore':'slow',   'take_priority':'explore','speed':'slow'},
    'accumulate_grasp':   {'explore':'normal', 'take_priority':'direct','speed':'medium'},
    'waiting_grasp':      {'explore':'slow',   'take_priority':'explore','speed':'slow'},
    'abort_or_retry':     {'explore':'normal', 'take_priority':'explore','speed':'medium'},
    'nurture_grasp':      {'explore':'slow',   'take_priority':'careful','speed':'slow'},
    'robust_power_grasp': {'explore':'fast',   'take_priority':'direct','speed':'fast'},
    'robust_grasp':       {'explore':'fast',   'take_priority':'direct','speed':'fast'},
    'injury_grasp':       {'explore':'slow',   'take_priority':'careful','speed':'slow'},
    'conflict_grasp':     {'explore':'fast',   'take_priority':'direct','speed':'fast'},
    'conditional_grasp':  {'explore':'normal', 'take_priority':'explore','speed':'medium'},
    'reduced_force_grasp':{'explore':'slow',   'take_priority':'careful','speed':'slow'},
    'retreat_grasp':      {'explore':'slow',   'take_priority':'explore','speed':'slow'},
    'coordinated_grasp':  {'explore':'normal', 'take_priority':'direct','speed':'medium'},
    'dual_grasp':         {'explore':'normal', 'take_priority':'direct','speed':'medium'},
    'competitive_grasp':  {'explore':'fast',   'take_priority':'direct','speed':'fast'},
    'difficult_grasp':    {'explore':'slow',   'take_priority':'careful','speed':'slow'},
    'direct_grasp':       {'explore':'fast',   'take_priority':'direct','speed':'fast'},
    'sequential_grasp':   {'explore':'normal', 'take_priority':'direct','speed':'medium'},
    'support_grasp':      {'explore':'normal', 'take_priority':'direct','speed':'medium'},
    'prepared_grasp':     {'explore':'normal', 'take_priority':'direct','speed':'medium'},
    'advance_grasp':      {'explore':'fast',   'take_priority':'direct','speed':'fast'},
    'observation':        {'explore':'slow',   'take_priority':'explore','speed':'slow'},
}

DEFAULT_PARAMS = {'explore':'normal','take_priority':'direct','speed':'medium'}


# ========== 动作选择器 (YLYW V4核心逻辑) ==========
def build_action_selector(strategy):
    """根据策略生成动作选择参数"""
    return HEX_ACTION_PARAMS.get(strategy, DEFAULT_PARAMS)

def pick_action(phase, admissible, target_obj, target_loc, preproc_loc, params):
    """YLYW V4的核心动作选择逻辑 + 策略参数影响"""
    
    has_take   = any(c.startswith('take ') for c in admissible)
    has_put    = any(c.startswith('put ') for c in admissible)
    has_open   = any(c.startswith('open ') for c in admissible)
    has_close  = any(c.startswith('close ') for c in admissible)
    has_clean  = any(c.startswith('clean ') for c in admissible)
    has_heat   = any(c.startswith('heat ') for c in admissible)
    has_cool   = any(c.startswith('cool ') for c in admissible)
    has_goto   = any(c.startswith('go to') for c in admissible)
    
    P_EXPLORE, P_TAKE, P_GOTO_PREPROC, P_PREPROC = 0, 1, 2, 3
    P_RETRIEVE, P_GOTO_TARGET, P_PLACE = 4, 5, 6
    
    if phase == P_EXPLORE:
        if has_take: return _take(admissible, target_obj, params)
        if has_goto:
            # 快速探索 vs 慢速探索
            if params['explore'] == 'fast':
                return admissible[0] if admissible else 'look'
            for loc_kw in ['cabinet','countertop','shelf','drawer','fridge']:
                for c in admissible:
                    if c.startswith('go to') and loc_kw in c:
                        return c
            return _goto(admissible)
        return 'look'
    
    elif phase == P_TAKE:
        if has_take: return _take(admissible, target_obj, params)
        if has_open: return _open(admissible)
        if has_goto: return _goto(admissible)
        return 'look'
    
    elif phase == P_GOTO_PREPROC:
        if preproc_loc and has_goto:
            for c in admissible:
                if c.startswith('go to') and preproc_loc in c:
                    return c
        return _preproc(admissible, preproc_loc, has_put, has_open, has_close,
                       has_clean, has_heat, has_cool, has_goto)
    
    elif phase == P_PREPROC:
        return _preproc(admissible, preproc_loc, has_put, has_open, has_close,
                       has_clean, has_heat, has_cool, has_goto)
    
    elif phase == P_RETRIEVE:
        if has_take: return _take(admissible, target_obj, params)
        if has_open and preproc_loc:
            for c in admissible:
                if c.startswith('open') and preproc_loc in c: return c
        if has_goto and preproc_loc:
            for c in admissible:
                if c.startswith('go to') and preproc_loc in c: return c
        return 'look'
    
    elif phase == P_GOTO_TARGET:
        if target_loc and has_goto:
            for c in admissible:
                if c.startswith('go to') and target_loc in c: return c
        if has_put: return _put(admissible, target_loc)
        if has_goto: return _goto(admissible)
        return 'look'
    
    elif phase == P_PLACE:
        if has_open and target_loc:
            for c in admissible:
                if c.startswith('open') and target_loc in c: return c
        if has_put: return _put(admissible, target_loc)
        if target_loc and has_goto:
            for c in admissible:
                if c.startswith('go to') and target_loc in c: return c
        return 'look'
    
    return admissible[0] if admissible else 'look'


def _take(admissible, target_obj, params):
    """拿物体——受策略影响"""
    if target_obj:
        # 策略影响选物顺序
        if params['take_priority'] == 'direct':
            # 直接选目标物体
            for c in admissible:
                if c.startswith('take ') and target_obj in c.lower():
                    return c
        elif params['take_priority'] == 'careful':
            # 谨慎选——可能先确认
            for c in admissible:
                if c.startswith('take ') and target_obj in c.lower():
                    return c
        elif params['take_priority'] == 'explore':
            # 探索性——先拿别的试试
            for c in admissible:
                if c.startswith('take '):
                    if target_obj not in c.lower():
                        return c
                    return c
    for c in admissible:
        if c.startswith('take '): return c
    return None

def _put(admissible, target_loc):
    if target_loc:
        for c in admissible:
            if c.startswith('put ') and target_loc in c: return c
    for c in admissible:
        if c.startswith('put '): return c
    return None

def _open(admissible):
    for c in admissible:
        if c.startswith('open '): return c
    return None

def _goto(admissible):
    for c in admissible:
        if c.startswith('go to'): return c
    return None

def _preproc(admissible, preproc_loc, has_put, has_open, has_close,
             has_clean, has_heat, has_cool, has_goto):
    if preproc_loc:
        if has_put:
            for c in admissible:
                if c.startswith('put ') and preproc_loc in c: return c
        if has_open:
            for c in admissible:
                if c.startswith('open ') and preproc_loc in c: return c
        if has_clean:
            for c in admissible:
                if c.startswith('clean '): return c
        if has_heat:
            for c in admissible:
                if c.startswith('heat '): return c
        if has_cool:
            for c in admissible:
                if c.startswith('cool '): return c
        if has_close:
            for c in admissible:
                if c.startswith('close ') and preproc_loc in c: return c
        if has_goto:
            for c in admissible:
                if c.startswith('go to') and preproc_loc in c: return c
    return 'look'


# ========== 任务执行 ==========
def run_with_decision_fn(game_idx, decision_fn, label, max_steps=30):
    """用指定的决策函数运行ALFWorld任务"""
    env = ALFWorldOfficial()
    obs, info = env.reset(game_idx=game_idx)
    task_desc = info.get('task_desc', '')
    
    # 解析任务
    obj_name = 'food'
    for o in ['plate','bowl','mug','cup','apple','potato','tomato',
              'bread','egg','soap','pencil','fork','knife','spoon',
              'coffee','milk','butter','food']:
        if o in task_desc.lower():
            obj_name = o; break
    
    need_clean = 'clean' in task_desc.lower() or 'wash' in task_desc.lower()
    need_heat  = 'heat' in task_desc.lower()
    need_cool  = 'cool' in task_desc.lower() or 'chill' in task_desc.lower()
    need_preproc = need_clean or need_heat or need_cool
    preproc_loc = 'sinkbasin' if need_clean else ('microwave' if need_heat else ('fridge' if need_cool else None))
    
    target_obj = obj_name
    target_loc = 'countertop'
    for tl in ['countertop','cabinet','shelf','drawer','desk','table',
               'garbagecan','diningtable']:
        if tl in task_desc.lower():
            target_loc = tl; break
    
    # 获取决策策略
    features = get_features(obj_name)
    hex_name, strategy = decision_fn(features)
    params = build_action_selector(strategy)
    
    # 按YLYW V4的phase系统执行
    phases = {0:'探索',1:'拿物',2:'去预处理',3:'预处理',4:'取出',5:'去目标',6:'放置'}
    P_EXPLORE, P_TAKE, P_GOTO_PREPROC, P_PREPROC = 0,1,2,3
    P_RETRIEVE, P_GOTO_TARGET, P_PLACE = 4,5,6
    phase = P_EXPLORE
    history = []
    carried = None
    success = False
    
    for step in range(max_steps):
        admissible = info.get('admissible_commands', ['look'])
        
        # ===== Phase转换 =====
        has_take = any(c.startswith('take ') for c in admissible)
        has_clean = any(c.startswith('clean ') for c in admissible)
        has_heat = any(c.startswith('heat ') for c in admissible)
        has_cool = any(c.startswith('cool ') for c in admissible)
        has_put = any(c.startswith('put ') for c in admissible)
        has_goto = any(c.startswith('go to') for c in admissible)
        
        if phase == P_EXPLORE and has_take:
            phase = P_TAKE
        elif phase == P_TAKE and ('pick up' in obs.lower() or 'you take' in obs.lower()):
            carried = obj_name
            phase = P_GOTO_PREPROC if need_preproc else P_GOTO_TARGET
        elif phase == P_GOTO_PREPROC:
            if preproc_loc and any(preproc_loc in c for c in admissible if c.startswith('put')):
                phase = P_PREPROC
            elif not has_goto:
                phase = P_PREPROC
        elif phase == P_PREPROC and not has_clean and not has_heat and not has_cool and not has_put:
            phase = P_RETRIEVE
        elif phase == P_RETRIEVE:
            if has_take:
                phase = P_GOTO_TARGET
                carried = None
            elif not has_open and not has_goto:
                phase = P_GOTO_TARGET
        elif phase == P_GOTO_TARGET:
            if target_loc and any(target_loc in c for c in admissible if c.startswith('put ')):
                phase = P_PLACE
            elif not has_goto:
                phase = P_PLACE
        
        # ===== 选动作 =====
        action = pick_action(phase, admissible, target_obj, target_loc, preproc_loc, params)
        if not action:
            break
        
        # ===== 执行 =====
        try:
            obs, _, done, info = env.step(action)
        except:
            break
        
        history.append(action)
        
        if 'pick up' in obs.lower():
            carried = obj_name
        if 'You are no longer' in obs:
            carried = None
        
        if done:
            success = True
            break
    
    return success, len(history), hex_name, strategy


# ========== 主测试 ==========
def main():
    print("="*70)
    print("  QYUF × ALFWorld V20 端到端对比 (完全体)")
    print("  经典YLYW vs QYUF量子涌现 — 相同动作执行器")
    print("="*70)
    
    num_tasks = 30
    results = {'classic': [], 'quantum': []}
    
    for method, label, decision_fn in [
        ('classic', '经典YLYW', classic_decision),
        ('quantum', 'QYUF量子', lambda f: quantum_decision(f, 1))
    ]:
        print(f"\n--- [{label}] {num_tasks} tasks ---")
        for gi in range(num_tasks):
            try:
                success, steps, hex_name, strat = run_with_decision_fn(gi, decision_fn, label)
            except Exception as e:
                success, steps, hex_name, strat = False, 0, "?", "?"
            
            results[method].append((success, steps, hex_name, strat))
            mark = "✓" if success else "✗"
            print(f"  #{gi:2d} {mark} steps={steps:2d} {hex_name:4s}({strat:20s})")
        
        succ = sum(1 for r in results[method] if r[0])
        avg_s = np.mean([r[1] for r in results[method]]) if results[method] else 0
        avg_s_succ = np.mean([r[1] for r in results[method] if r[0]]) if succ else 0
        print(f"  [{label}] 完成: {succ}/{num_tasks} ({succ/num_tasks*100:.1f}%) 步数: {avg_s:.1f}(成功:{avg_s_succ:.1f})")
    
    # 总结
    c_ok = sum(1 for r in results['classic'] if r[0])
    q_ok = sum(1 for r in results['quantum'] if r[0])
    print(f"\n{'='*70}")
    print(f"  总结")
    print(f"{'='*70}")
    print(f"  {'指标':25s} {'经典YLYW':>12s} {'QYUF量子':>12s}")
    print(f"  {'-'*51}")
    print(f"  {'任务数':25s} {num_tasks:>12d} {num_tasks:>12d}")
    print(f"  {'完成':25s} {c_ok:>12d} {q_ok:>12d}")
    print(f"  {'完成率':25s} {c_ok/num_tasks*100:>11.1f}% {q_ok/num_tasks*100:>11.1f}%")
    
    c_steps = np.mean([r[1] for r in results['classic']]) or 0
    q_steps = np.mean([r[1] for r in results['quantum']]) or 0
    print(f"  {'平均步数':25s} {c_steps:>12.1f} {q_steps:>12.1f}")
    
    # 卦象质量
    print(f"\n  卦象质量:")
    for method, label in [('classic', '经典YLYW'), ('quantum', 'QYUF量子')]:
        scores = []
        for _, _, hn, _ in results[method]:
            if hn:
                idx = HEXAGRAM_NAMES.index(hn) if hn in HEXAGRAM_NAMES else -1
                if idx >= 0:
                    scores.append(qyuf.scores[idx])
        avg = np.mean(scores) if scores else 0
        print(f"  {label:12s}: 平均易理评分 {avg:+.1f}")
    
    # 分策略的完成率
    strat_stats = {}
    for method in ['classic', 'quantum']:
        for ok, st, hn, strat in results[method]:
            if strat not in strat_stats:
                strat_stats[strat] = {'classic_ok':0,'classic_n':0,'quantum_ok':0,'quantum_n':0}
            if method == 'classic':
                strat_stats[strat]['classic_n'] += 1
                if ok: strat_stats[strat]['classic_ok'] += 1
            else:
                strat_stats[strat]['quantum_n'] += 1
                if ok: strat_stats[strat]['quantum_ok'] += 1
    
    print(f"\n  策略完成率:")
    print(f"  {'策略':25s} {'经典完成':>8s} {'量子完成':>8s}")
    print(f"  {'-'*43}")
    for s, st in sorted(strat_stats.items()):
        cr = f"{st['classic_ok']}/{st['classic_n']}" if st['classic_n'] else "-"
        qr = f"{st['quantum_ok']}/{st['quantum_n']}" if st['quantum_n'] else "-"
        print(f"  {s:25s} {cr:>8s} {qr:>8s}")


if __name__ == "__main__":
    main()
