"""Diff the predicted vs gold agent DB for a task, exactly as the evaluator does
(but printing the differing tables/rows instead of just hashing).

    uv run python dbdiff.py <results_dir> <task_id> [task_id ...]
"""

import glob
import json
import sys

from a2a_hack.domain import get_hack_environment, get_hack_tasks
from tau2.data_model.simulation import SimulationRun


def build_gold(task):
    env = get_hack_environment()
    init = task.initial_state
    env.set_state(
        initialization_data=init.initialization_data if init else None,
        initialization_actions=init.initialization_actions if init else None,
        message_history=(init.message_history if init and init.message_history else []),
    )
    for a in task.evaluation_criteria.actions or []:
        try:
            env.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
        except Exception as e:
            print(f"  [gold action error] {a.name}: {e}")
    return env


def build_predicted(task, sim):
    env = get_hack_environment(solo_mode=False)
    init = task.initial_state
    env.set_state(
        initialization_data=init.initialization_data if init else None,
        initialization_actions=init.initialization_actions if init else None,
        message_history=list(sim.messages or []),
    )
    return env


def diff_tables(gold: dict, pred: dict):
    for table in sorted(set(gold) | set(pred)):
        g = (gold.get(table) or {}).get("data", {}) if isinstance(gold.get(table), dict) else {}
        p = (pred.get(table) or {}).get("data", {}) if isinstance(pred.get(table), dict) else {}
        if not isinstance(g, dict) or not isinstance(p, dict):
            continue
        only_pred = set(p) - set(g)
        only_gold = set(g) - set(p)
        changed = [k for k in set(g) & set(p) if json.dumps(g[k], sort_keys=True, default=str) != json.dumps(p[k], sort_keys=True, default=str)]
        if not (only_pred or only_gold or changed):
            continue
        print(f"  ▸ table '{table}':")
        for k in sorted(only_pred):
            print(f"      + PREDICTED-ONLY {k}: {json.dumps(p[k], default=str)[:160]}")
        for k in sorted(only_gold):
            print(f"      - GOLD-ONLY      {k}: {json.dumps(g[k], default=str)[:160]}")
        for k in sorted(changed):
            print(f"      ~ CHANGED        {k}")
            print(f"          gold: {json.dumps(g[k], default=str)[:200]}")
            print(f"          pred: {json.dumps(p[k], default=str)[:200]}")


def main():
    results_dir, task_ids = sys.argv[1], sys.argv[2:]
    tasks = {t.id: t for t in get_hack_tasks()}
    sims = {}
    for f in glob.glob(f"{results_dir}/simulations/*.json"):
        d = json.load(open(f))
        sims[d["task_id"]] = f
    for tid in task_ids:
        print("=" * 70)
        print(tid)
        task = tasks[tid]
        sim = SimulationRun.model_validate(json.load(open(sims[tid])))
        gold = build_gold(task).tools.db.model_dump()
        pred = build_predicted(task, sim).tools.db.model_dump()
        if gold == pred:
            print("  DB MATCH (no differences) — reward should be 1.0")
        else:
            print("  DB MISMATCH — differing tables below:")
            diff_tables(gold, pred)


if __name__ == "__main__":
    main()
