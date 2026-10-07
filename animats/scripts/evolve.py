#!/usr/bin/env python3
"""Run independent evolutionary runs (one per seed) for a task, in parallel.

Each seed runs MABE in animats/<task>/runs/seed_NN/ with that task's configs and
GLOBAL-randomSeed = NN. A run is finished once run_info.json exists; finished seeds
are skipped unless --force is given. Unfinished seed directories are cleared and rerun.

Progress: in a terminal, each running seed gets a progress bar that updates in place.
When output goes to a file (e.g. nohup), a status summary is printed every 10 minutes
instead. MABE's own console output goes to runs/seed_NN/raw/evolve.log.

Examples:
    python animats/scripts/evolve.py task1 --seeds 1-50 --jobs 10
    python animats/scripts/evolve.py task1 --seeds 1-2 --updates 2000    # quick test
"""

import argparse
import gzip
import json
import math
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import REPO, config_args, config_value, git_state, parse_seeds, run_mabe, seed_dir


def evolve(task, seed, updates, force, progress):
    out = seed_dir(task, seed)
    if (out / "run_info.json").exists() and not force:
        return "skipped (already finished)"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    args = [*config_args(task), "-p", "GLOBAL-randomSeed", str(seed)]
    if updates is not None:
        args += ["-p", "GLOBAL-updates", str(updates)]

    start = time.time()
    progress.started(seed)
    run_mabe(args, cwd=out, log_path=out / "raw" / "evolve.log")
    # genomes are ~80 KB each by mid-run; gzip cuts the committed file ~3x (only extract.py reads it)
    with open(out / "LOD_organisms.csv", "rb") as src, gzip.open(out / "LOD_organisms.csv.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    (out / "LOD_organisms.csv").unlink()
    info = {
        "task": task,
        "seed": seed,
        "updates_override": updates,
        "mabe_args": [a.replace(str(REPO) + "/", "") for a in args],  # repo-relative
        "elapsed_seconds": round(time.time() - start, 1),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        **git_state(),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return f"done in {hms(info['elapsed_seconds'])}"


def hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def fitness_from_max(task):
    """A function turning MABE's printed `max` (selection score S) back into fitness F, if possible.

    With optimizeValue = POW[b,MULT[DM_AVE[score],n]], S = b^(n*F), so F = ln(S) / ln(b) / n.
    """
    optimizer = config_value(task, "OPTIMIZER", "optimizer")
    formula = config_value(task, f"OPTIMIZER_{optimizer.upper()}", "optimizeValue")
    m = re.fullmatch(r"POW\[([\d.]+),MULT\[DM_AVE\[score\],(\d+)\]\]", formula)
    if not m:
        return None
    base, trials = float(m.group(1)), int(m.group(2))
    return lambda s: math.log(s) / math.log(base) / trials


class Progress:
    """Status of all seeds, drawn as in-place progress bars in a terminal."""

    BAR = 20
    UPDATE_RE = re.compile(r"update: (\d+)\s+max = ([\d.eE+-]+)")

    def __init__(self, task, n_seeds, total_updates):
        self.task, self.n_seeds, self.total = task, n_seeds, total_updates
        self.to_fitness = fitness_from_max(task)
        self.tty = sys.stdout.isatty()
        self.lock = threading.Lock()
        self.running = {}  # seed -> start time
        self.done = self.failed = 0
        self.start = time.time()
        self.drawn = 0     # lines of the current bar block, so it can be redrawn in place
        self.stop = threading.Event()

    def started(self, seed):
        with self.lock:
            self.running[seed] = time.time()

    def finished(self, seed, message, failed=False):
        with self.lock:
            self.running.pop(seed, None)
            if failed:
                self.failed += 1
            else:
                self.done += 1
            self._erase()
            print(f"seed {seed:02d}: {message}", flush=True)
            self._draw()

    def loop(self):
        """Redraw every second in a terminal; print a summary every 10 minutes otherwise."""
        last_summary = time.time()
        while not self.stop.wait(1):
            with self.lock:
                if self.tty:
                    self._erase()
                    self._draw()
                elif time.time() - last_summary >= 600:
                    print("\n".join(self._lines()), flush=True)
                    last_summary = time.time()
        with self.lock:
            self._erase()

    def _latest(self, seed):
        """Last generation and best score MABE has printed for this seed."""
        log = seed_dir(self.task, seed) / "raw" / "evolve.log"
        try:
            with open(log, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                matches = self.UPDATE_RE.findall(f.read().decode(errors="replace"))
        except FileNotFoundError:
            matches = []
        return (int(matches[-1][0]), float(matches[-1][1])) if matches else (0, None)

    def _lines(self):
        lines = [f"{self.task}: {self.done}/{self.n_seeds} done, {len(self.running)} running, "
                 f"{self.failed} failed, elapsed {hms(time.time() - self.start)}"]
        for seed, started in sorted(self.running.items()):
            gen, best = self._latest(seed)
            frac = min(gen / self.total, 1.0)
            filled = int(frac * self.BAR)
            bar = "█" * filled + "░" * (self.BAR - filled)
            if best is None:
                score = ""
            elif self.to_fitness:
                score = f"best F {self.to_fitness(best):.3f}"
            else:
                score = f"max {best:.3g}"
            elapsed = time.time() - started
            eta = hms(elapsed / frac * (1 - frac)) if frac > 0 else "?"
            lines.append(f"  seed {seed:02d} [{bar}] {gen:>6}/{self.total} {frac:4.0%}  {score}  eta {eta}")
        return lines

    def _draw(self):
        if not self.tty:
            return
        width = shutil.get_terminal_size().columns - 1
        lines = [line[:width] for line in self._lines()]  # wrapped lines would break the redraw
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        self.drawn = len(lines)

    def _erase(self):
        if self.tty and self.drawn:
            sys.stdout.write(f"\033[{self.drawn}F\033[J")  # cursor up to the block's start, clear below
            self.drawn = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", help="task name, e.g. task1 (configs read from animats/<task>/configs/)")
    parser.add_argument("--seeds", required=True, help="seeds to run, e.g. 1-50 or 1,3,5")
    parser.add_argument("--jobs", type=int, default=1, help="number of runs in parallel (one core each)")
    parser.add_argument("--updates", type=int, help="override GLOBAL-updates (for short test runs)")
    parser.add_argument("--force", action="store_true", help="rerun seeds that already finished")
    args = parser.parse_args()

    if git_state()["dirty"]:
        print("warning: working tree has uncommitted changes; run_info.json records the commit, "
              "so commit config changes before real runs", file=sys.stderr)

    seeds = parse_seeds(args.seeds)
    total = args.updates if args.updates is not None else int(config_value(args.task, "GLOBAL", "updates"))
    progress = Progress(args.task, len(seeds), total)
    drawer = threading.Thread(target=progress.loop, daemon=True)
    drawer.start()

    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(evolve, args.task, s, args.updates, args.force, progress): s for s in seeds}
        for f in as_completed(futures):
            seed = futures[f]
            try:
                progress.finished(seed, f.result())
            except Exception as e:
                failed.append(seed)
                progress.finished(seed, f"FAILED: {e}", failed=True)
    progress.stop.set()
    drawer.join()
    if failed:
        sys.exit(f"failed seeds: {sorted(failed)}")


if __name__ == "__main__":
    main()
