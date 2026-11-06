"""Shared machinery for the per-version IIT scripts (iit3.py, iit4.py): the measures of
Albantakis et al. (2014), Fig. 3, along the lines of descent, and the figure itself.

A version module defines
    NAME            "iit3" or "iit4", also its module name; outputs go to OUTPUT/<NAME>/
    MEASURES        names of the values brain_measures returns
    FIGURES         {file name: rows}, one Fig. 3 per entry, rows being {column: (y label,
                    y range or None)}, "fitness" included
    brain_measures(substrate, state, hidden) -> {measure: value} for one brain-state

Brain-states: the substrate has the real motor TPM columns and each state has the motors'
actual outputs (load.with_motor_outputs). Each animat's measures are the means over the states
it visited during the 128 trials, weighted by probability of occurrence, as in the paper.

Each (tpm_hash, state) is computed once, in parallel worker processes, and appended to
OUTPUT/<NAME>/cache.csv, shared across seeds and tasks, so an interrupted run resumes where it
stopped and a rerun only reads files. OUTPUT/<NAME>/info.json records the pyphi version the cache
was computed with; another version refuses to add to it (delete the folder to start over).
Per task, OUTPUT/<NAME>/<task>.csv has one row per animat on each LOD; fig3*.png are the figures.
Runs over a subset of seeds get a _seeds_... suffix, so they don't overwrite the full set.

Progress: in a terminal, an in-place bar; with output redirected to a file, a timestamped line
every 5 minutes, so `tail -f` on the log shows how far a long run is. Brains are computed in
decreasing order of connectivity, the slowest first, so that the last brains of each batch are
quick ones and few workers sit idle while a batch finishes. The ETA is based on the brain-states
done, so it starts pessimistic.
"""

import argparse
import importlib
import json
import multiprocessing
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyphi

from common import format_seeds, parse_seeds
from load import list_animats, list_seeds, load_animat, to_substrate, with_motor_outputs

OUTPUT = Path(__file__).resolve().parent.parent / "notebooks" / "output"
TASKS = ["task1", "task2", "task3", "task4"]
TASK_STYLE = {  # title colour and blocks to catch / avoid, as in the paper's header
    "task1": ("black", "catch 1, avoid 3"),
    "task2": ("#000080", "catch 1, avoid 2"),
    "task3": ("#0000ff", "catch 1+4, avoid 2+3"),
    "task4": ("#1e90ff", "catch 3+6, avoid 4+5"),
}
SUBSET_COLOR = {"task3": "#8b0000", "task4": "#ff0000"}


def hidden_nodes(labels):
    return [i for i, label in enumerate(labels) if str(label).startswith("H")]


def state_string(state):
    return "".join(str(int(x)) for x in state)


# ---- computing brain-states ----

def _measure_brain(name, task, seed, generation, states):
    """Worker: MEASURES of module `name` for each state (a bit string) of one animat's brain."""
    module = importlib.import_module(name)
    pyphi.config.progress_bars = False
    animat = with_motor_outputs(load_animat(task, seed, generation))
    substrate = to_substrate(animat)
    hidden = hidden_nodes(animat["node_labels"])
    return [{"state": s, **module.brain_measures(substrate, tuple(int(c) for c in s), hidden)}
            for s in states]


def _hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


class _Progress:
    """An in-place bar in a terminal; otherwise (a log file) a timestamped line every `interval` s."""

    BAR = 20

    def __init__(self, name, brains, states, interval=300):
        self.name, self.brains, self.states, self.interval = name, brains, states, interval
        self.done_brains = self.done_states = 0
        self.start = self.last = time.time()
        self.tty = sys.stdout.isatty()
        self._show()

    def update(self, n_states):
        self.done_brains += 1
        self.done_states += n_states
        if self.tty or time.time() - self.last >= self.interval or self.done_brains == self.brains:
            self._show()

    def _show(self):
        self.last = time.time()
        elapsed = self.last - self.start
        frac = self.done_states / self.states if self.states else 1
        eta = _hms(elapsed / frac * (1 - frac)) if frac > 0 else "?"
        bar = "█" * int(frac * self.BAR) + "░" * (self.BAR - int(frac * self.BAR))
        line = (f"{self.name} [{bar}] {frac:4.0%}  eta {eta}  elapsed {_hms(elapsed)}  "
                f"{self.done_states}/{self.states} brain-states, {self.done_brains}/{self.brains} brains")
        if self.tty:  # cut to the terminal width: a wrapped line can't be redrawn in place
            print("\r" + line[:shutil.get_terminal_size().columns - 1] + "\033[K", end="", flush=True)
        else:
            print(time.strftime("%Y-%m-%d %H:%M ") + line, flush=True)

    def close(self):
        if self.tty:
            print(flush=True)


def _compute(module, todo, cache, path, jobs, batch=500):
    """Compute the brains in `todo` in parallel, adding results to `cache` and to the CSV at `path`.

    todo: tpm_hash -> ((task, seed, generation), set of states, #connections). The most connected
    brains go first, as they are the slowest (up to ~10x the mean time per state under IIT 4.0).

    A fresh worker pool per batch of brains keeps pyphi's per-process caches from growing for
    the whole run. (Not max_tasks_per_child: with it, ProcessPoolExecutor stopped replacing
    exited workers after ~500 brains and hung with no workers left.)
    """
    ctx = multiprocessing.get_context("forkserver")  # fork is unsafe from a threaded Jupyter kernel
    items = sorted(todo.items(), key=lambda item: (item[1][2], len(item[1][1])), reverse=True)
    progress = _Progress(module.NAME, len(items), sum(len(states) for _, (_, states, _) in items))
    try:
        for i in range(0, len(items), batch):
            pending = []
            with ProcessPoolExecutor(jobs, mp_context=ctx) as pool:
                futures = {pool.submit(_measure_brain, module.NAME, *where, sorted(states)): h
                           for h, (where, states, _) in items[i:i + batch]}
                try:
                    for f in as_completed(futures):
                        h = futures[f]
                        results = f.result()
                        for r in results:
                            cache[(h, r["state"])] = {m: r[m] for m in module.MEASURES}
                            pending.append({"tpm_hash": h, **r})
                        progress.update(len(results))
                except KeyboardInterrupt:
                    for f in futures:
                        f.cancel()
                    raise
                finally:  # saved after every batch, and on an interruption
                    if pending:
                        pd.DataFrame(pending, columns=["tpm_hash", "state", *module.MEASURES]).to_csv(
                            path, mode="a", header=not path.exists(), index=False)
    finally:
        progress.close()


def _check_cache_info(folder):
    """Refuse to mix pyphi versions in one cache (the cache is shared across machines)."""
    info_path = folder / "info.json"
    info = {"pyphi": pyphi.__version__}
    if info_path.exists():
        saved = json.loads(info_path.read_text())
        if saved != info:
            sys.exit(f"{folder}/cache.csv was computed with {saved}, this is {info}. Install the same "
                     f"pyphi (uv sync), or delete {folder} to recompute everything.")
    else:
        info_path.write_text(json.dumps(info, indent=2) + "\n")


def suffix(task, seeds):
    """'' if `seeds` is every finished seed of `task`, else _seeds_..."""
    return "" if sorted(seeds) == list_seeds(task) else "_seeds_" + format_seeds(seeds).replace(",", "_")


def lod_measures(module, tasks=TASKS, seeds=None, jobs=10):
    """Weighted means of module.MEASURES for every sampled animat on every LOD of each task.

    seeds: None for every finished seed, else a list (seeds a task lacks are skipped).
    Returns {task: DataFrame} with one row per (seed, generation): score and the MEASURES, each
    also saved as OUTPUT/<NAME>/<task>[_seeds_...].csv.
    """
    folder = OUTPUT / module.NAME
    folder.mkdir(parents=True, exist_ok=True)
    _check_cache_info(folder)
    path = folder / "cache.csv"
    cache = {}
    if path.exists():
        saved = pd.read_csv(path, dtype={"tpm_hash": str, "state": str})
        missing = [m for m in module.MEASURES if m not in saved.columns]
        if missing:
            sys.exit(f"{path} has no {missing} (computed before they were added): delete {folder} "
                     f"to recompute everything.")
        cache = {(h, s): row for h, s, row in zip(saved["tpm_hash"], saved["state"],
                                                   saved[module.MEASURES].to_dict("records"))}

    # one pass over the substrates: what each animat visited, and which brain-states are missing
    animats, todo = [], {}  # todo: tpm_hash -> ((task, seed, generation), set of states, #connections)
    task_seeds = {task: [s for s in list_seeds(task) if seeds is None or s in seeds] for task in tasks}
    for task, ts in task_seeds.items():
        for seed in ts:
            for row in list_animats(task, seed).itertuples():
                a = with_motor_outputs(load_animat(task, seed, row.generation))
                states = [state_string(s) for s in a["visited_states"]]
                animats.append((task, seed, row.generation, row.score, row.tpm_hash, states,
                                a["visited_probs"]))
                missing = [s for s in states if (row.tpm_hash, s) not in cache]
                if missing:
                    where = ((task, seed, row.generation), set(), int(a["cm"].sum()))
                    todo.setdefault(row.tpm_hash, where)[1].update(missing)
    n_todo = sum(len(states) for _, states, _ in todo.values())
    print(f"{module.NAME}: {len(cache)} brain-states cached, {n_todo} to compute over {len(todo)} brains",
          flush=True)
    if todo:
        _compute(module, todo, cache, path, jobs)

    rows = []
    for task, seed, generation, score, h, states, probs in animats:
        values = np.array([[cache[(h, s)][m] for m in module.MEASURES] for s in states])
        rows.append({"task": task, "seed": seed, "generation": generation, "score": score,
                     **dict(zip(module.MEASURES, probs @ values))})
    tables = {}
    for task, table in pd.DataFrame(rows).groupby("task", sort=False):
        table = table.reset_index(drop=True)
        table.to_csv(folder / f"{task}{suffix(task, task_seeds[task])}.csv", index=False)
        tables[task] = table
    return tables


# ---- statistics and the figure ----

def late_mean(table, column, window):
    """Mean per seed over the last `window` generations."""
    late = table[table["generation"] > table["generation"].max() - window]
    return late.groupby("seed")[column].mean()


def fittest_subset(table, target, window=5000):
    """The k fittest seeds, with k chosen so their mean fitness over the last `window`
    generations is closest to `target` (the paper's matched subsets of Tasks 3 and 4)."""
    fitness = late_mean(table, "score", window).sort_values(ascending=False)
    k = min(range(1, len(fitness) + 1), key=lambda k: abs(fitness.iloc[:k].mean() - target))
    return sorted(fitness.index[:k])


def fittest_subsets(tables):
    """{task: seeds} for Tasks 3 and 4, matched to Task 1's mean fitness over the last 5,000
    generations. Empty without Task 1, or with a single seed (nothing to choose)."""
    if "task1" not in tables or tables["task1"]["seed"].nunique() < 2:
        return {}
    target = late_mean(tables["task1"], "score", 5000).mean()
    return {task: fittest_subset(tables[task], target) for task in ("task3", "task4") if task in tables}


def late_summary(tables, measures, subsets):
    """Mean ± SEM across LODs of each measure over the last 3,000 generations (the paper's
    statistics window), per task and fittest subset. Fitness in %."""
    summary = {}
    for task, table in tables.items():
        groups = [(task, table)]
        if task in subsets:
            groups.append((f"{task} ({len(subsets[task])} fittest)", table[table["seed"].isin(subsets[task])]))
        for label, t in groups:
            late = pd.DataFrame({c: late_mean(t, c, 3000) for c in ["score", *measures]})
            late["score"] *= 100
            summary[label] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(late.mean(), late.sem())]
    return pd.DataFrame(summary, index=["fitness (%)", *measures])


def plot_fig3(tables, rows, subsets, path=None):
    """The paper's Fig. 3: one column per task, one row per entry of `rows` ({column: (y label,
    y range or None)}; "fitness" is the score in %). Task 1 is repeated in black in every
    column and the fittest subsets of Tasks 3 and 4 are in red, as in the paper; bands are ±1
    SEM across LODs. A y range is extended if the data goes beyond it; None starts at 0 and
    fits the data. Saved to `path` if given."""
    tasks = [t for t in TASKS if t in tables]
    fig, axes = plt.subplots(len(rows), len(tasks), figsize=(3.2 * len(tasks), 2.2 * len(rows)),
                             sharex=True, squeeze=False)
    for col, task in enumerate(tasks):
        n = task.removeprefix("task")
        series = [(tables["task1"], "black", "Task 1")] if "task1" in tables else []
        if task != "task1":
            series.append((tables[task], TASK_STYLE[task][0], f"Task {n}"))
        if task in subsets:
            subset = tables[task][tables[task]["seed"].isin(subsets[task])]
            series.append((subset, SUBSET_COLOR[task], f"Task {n} ({len(subsets[task])} fittest)"))
        for row, (column, (label, yrange)) in enumerate(rows.items()):
            ax = axes[row, col]
            top = 0
            for table, color, name in series:
                stats = table.assign(fitness=table["score"] * 100).groupby("generation")[column].agg(["mean", "sem"])
                stats["sem"] = stats["sem"].fillna(0)  # a single seed has no SEM
                gen = stats.index / 1e4
                ax.plot(gen, stats["mean"], color=color, linewidth=1, label=name)
                ax.fill_between(gen, stats["mean"] - stats["sem"], stats["mean"] + stats["sem"],
                                color=color, alpha=0.2, linewidth=0)
                top = max(top, (stats["mean"] + stats["sem"]).max() * 1.05)
            lo, hi = yrange if yrange else (0, 0)
            ax.set_ylim(lo, max(hi, top) or 1)
            ax.set_ylabel(label)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, col].set_title(f"Task {n}\n{TASK_STYLE[task][1]}", color=TASK_STYLE[task][0], fontweight="bold")
        axes[0, col].legend(fontsize="small", frameon=False, loc="lower right")
        axes[-1, col].set_xlabel("#Generations (×10$^4$)")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


# ---- command line ----

def main(module, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--seeds", type=parse_seeds, help="e.g. 1-50 or 1,4,7 (default: every finished seed)")
    parser.add_argument("--jobs", type=int, default=10, help="worker processes (default 10)")
    args = parser.parse_args()

    tables = lod_measures(module, args.tasks, args.seeds, args.jobs)
    subsets = fittest_subsets(tables)
    for task, seeds in subsets.items():
        print(f"{task}: {len(seeds)} fittest seeds {seeds}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(late_summary(tables, module.MEASURES, subsets))
    seeds = "_seeds_" + format_seeds(args.seeds).replace(",", "_") if args.seeds else ""
    for name, rows in module.FIGURES.items():
        path = OUTPUT / module.NAME / f"{name}{seeds}.png"
        plot_fig3(tables, rows, subsets, path)
        plt.close("all")
        print(f"saved {path}")
