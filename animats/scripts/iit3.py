"""IIT 3.0 measures along the lines of descent, as in Albantakis et al. (2014), Fig. 3.

For an animat brain in one state, `brain_measures` computes
    n_concepts, sum_phi        concepts of the whole brain (sensors, motors, hidden) and their Σφ^Max
    mc_elements, mc_concepts   size of the main complex (MC) and the number of its concepts
    phi_max                    Φ^Max of the MC
All are 0 when there is no complex. `lod_measures` averages them over the states each sampled
animat visited during the 128 trials, weighted by probability of occurrence, as the paper does.

Each (tpm_hash, state) is computed once, in parallel worker processes, and appended to a cache
CSV that is shared across seeds and tasks; an interrupted run resumes where it stopped.

Whole-brain concepts only consider mechanisms made of hidden nodes: sensors have no inputs and
motors no outputs within the brain, so they can't be (part of) a concept (paper, Fig. 2B). This
gives identical concepts (checked on 80 random brain-states) and is ~8x faster.

validate_states: pyphi refuses to analyse a system whose state has no possible predecessor (for
a candidate complex, given the rest of the brain as background), and such a candidate drops out.
The paper evaluates every state the animat experienced, which includes the post-reset state at
the start of each trial, so the default (False) turns that check off. In a 150-state sample this
changed Φ in 8 states. True gives the old phi_iit3 behaviour; it uses a separate cache.
"""

import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyphi
from pyphi.utils import powerset

from common import seed_dir
from load import list_animats, list_seeds, to_substrate

MEASURES = ["n_concepts", "sum_phi", "mc_elements", "mc_concepts", "phi_max"]


def brain_measures(substrate, state, hidden, validate_states=False):
    """The MEASURES of one brain in one state, as a dict. hidden: indices of the hidden nodes."""
    with pyphi.config.override(**pyphi.iit3, validate_system_states=validate_states):
        concepts = substrate.ces(state, mechanisms=list(powerset(hidden, nonempty=True))).concepts
        # complexes() rather than maximal_complex(): with validation on, the latter raises on
        # unreachable states when there is no complex
        found = substrate.complexes(state)
        mc_concepts = 0
        if found:
            distinctions = found[0].sia.distinctions
            if distinctions is None:  # only if the analysis short-circuited, which Φ > 0 rules out
                distinctions = substrate.ces(state, found[0].node_indices).concepts
            mc_concepts = sum(1 for c in distinctions if c.phi > 0)
    return {
        "n_concepts": sum(1 for c in concepts if c.phi > 0),
        "sum_phi": float(sum(c.phi for c in concepts)),
        "mc_elements": len(found[0].node_indices) if found else 0,
        "mc_concepts": mc_concepts,
        "phi_max": float(found[0].phi) if found else 0.0,
    }


def _hidden(labels):
    return [i for i, label in enumerate(labels) if str(label).startswith("H")]


def _measure_brain(path, states, validate_states):
    """Worker: the MEASURES for each state (a bit string) of the brain stored in `path`."""
    pyphi.config.progress_bars = False
    with np.load(path) as z:
        animat = {k: z[k] for k in ("tpm", "cm", "node_labels")}
    substrate = to_substrate(animat)
    hidden = _hidden(animat["node_labels"])
    return [{"state": s, **brain_measures(substrate, tuple(int(c) for c in s), hidden, validate_states)}
            for s in states]


def _compute(todo, cache, path, jobs, validate_states, batch=500):
    """Compute the brains in `todo` in parallel, adding results to `cache` and to the CSV at `path`.

    A fresh worker pool per batch of brains keeps pyphi's per-process caches from growing for
    the whole run. (Not max_tasks_per_child: with it, ProcessPoolExecutor stopped replacing
    exited workers after ~500 brains and hung with no workers left.)
    """
    ctx = multiprocessing.get_context("forkserver")  # fork is unsafe from a threaded Jupyter kernel
    items = list(todo.items())
    start, done = time.time(), 0
    for i in range(0, len(items), batch):
        pending = []
        with ProcessPoolExecutor(jobs, mp_context=ctx) as pool:
            futures = {pool.submit(_measure_brain, str(npz), sorted(states), validate_states): h
                       for h, (npz, states) in items[i:i + batch]}
            try:
                for f in as_completed(futures):
                    h = futures[f]
                    for r in f.result():
                        cache[(h, r["state"])] = {m: r[m] for m in MEASURES}
                        pending.append({"tpm_hash": h, **r})
                    done += 1
                    elapsed = time.time() - start
                    print(f"\r{done}/{len(items)} brains, {elapsed / 60:.0f} min, "
                          f"eta {elapsed / done * (len(items) - done) / 60:.0f} min ", end="", flush=True)
            except KeyboardInterrupt:
                for f in futures:
                    f.cancel()
                raise
            finally:  # saved after every batch, and on an interruption
                if pending:
                    pd.DataFrame(pending, columns=["tpm_hash", "state", *MEASURES]).to_csv(
                        path, mode="a", header=not path.exists(), index=False)
    print(file=sys.stdout)


def cache_path(output, validate_states=False):
    return output / ("iit3_validated_cache.csv" if validate_states else "iit3_cache.csv")


def lod_measures(tasks, output, jobs=10, validate_states=False, seeds=None):
    """Weighted means of MEASURES for every sampled animat on every LOD of each task.

    Returns {task: DataFrame} with one row per (seed, generation): score and the MEASURES.
    Each table is also saved as <output>/<task>_iit3.csv (_iit3_validated.csv with validation
    on, and a _seeds_... suffix if `seeds` restricts the seeds, e.g. for a quick test).
    """
    path = cache_path(output, validate_states)
    cache = {}
    if path.exists():
        saved = pd.read_csv(path, dtype={"tpm_hash": str, "state": str})
        cache = {(h, s): row for h, s, row in zip(saved["tpm_hash"], saved["state"],
                                                   saved[MEASURES].to_dict("records"))}

    # one pass over the substrates: what each animat visited, and which brain-states are missing
    animats, todo = [], {}  # todo: tpm_hash -> (npz path, set of states)
    for task in tasks:
        for seed in seeds or list_seeds(task):
            for row in list_animats(task, seed).itertuples():
                npz = seed_dir(task, seed) / "substrates" / f"gen_{row.generation:05d}.npz"
                with np.load(npz) as z:
                    states = ["".join(map(str, s)) for s in z["visited_states"]]
                    counts = z["visited_counts"]
                animats.append((task, seed, row.generation, row.score, row.tpm_hash, states,
                                counts / counts.sum()))
                missing = [s for s in states if (row.tpm_hash, s) not in cache]
                if missing:
                    todo.setdefault(row.tpm_hash, (npz, set()))[1].update(missing)
    n_todo = sum(len(states) for _, states in todo.values())
    print(f"{len(cache)} brain-states cached, {n_todo} to compute over {len(todo)} brains", flush=True)

    if todo:
        _compute(todo, cache, path, jobs, validate_states)

    rows = []
    for task, seed, generation, score, h, states, probs in animats:
        values = np.array([[cache[(h, s)][m] for m in MEASURES] for s in states])
        rows.append({"task": task, "seed": seed, "generation": generation, "score": score,
                     **dict(zip(MEASURES, probs @ values))})
    df = pd.DataFrame(rows)
    suffix = "_iit3_validated" if validate_states else "_iit3"
    if seeds:
        suffix += "_seeds_" + "_".join(map(str, sorted(seeds)))
    tables = {}
    for task, table in df.groupby("task", sort=False):
        table = table.reset_index(drop=True)
        table.to_csv(output / f"{task}{suffix}.csv", index=False)
        tables[task] = table
    return tables
