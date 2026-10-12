"""Paths and helpers shared by evolve.py, extract.py and load.py.

Layout (see CLAUDE.md):
    animats/configs/                 settings.cfg, settings_organism.cfg, shared by all tasks
    animats/<task>/configs/          settings_world.cfg; a file here overrides the shared one
    animats/<task>/runs/seed_NN/     one evolutionary run per seed
        run_info.json                written by evolve.py when the run finishes
        LOD_organisms.csv.gz         genomes along the line of descent, every 512 generations
        LOD_data.csv, pop.csv        per-generation stats
        animats.csv                  written by extract.py: one row per sampled animat
        substrates/gen_NNNNN.npz     written by extract.py: TPM, CM, visited states
        raw/                         (gitignored) MABE logs and intermediate files
"""

import os
import re
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANIMATS = REPO / "animats"
MABE = REPO / "work" / "mabe"
SHARED_CONFIGS = ANIMATS / "configs"
CONFIG_FILES = ("settings.cfg", "settings_world.cfg", "settings_organism.cfg")


def task_dir(task):
    return ANIMATS / task


def runs_dir(task):
    return task_dir(task) / "runs"


def seed_dir(task, seed):
    return runs_dir(task) / f"seed_{seed:02d}"


def config_path(task, name):
    """A task's own copy of a config file if it has one, else the shared one."""
    for p in (task_dir(task) / "configs" / name, SHARED_CONFIGS / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} in neither {task_dir(task) / 'configs'} nor {SHARED_CONFIGS}")


def config_args(task):
    """The `-f ...` arguments that load a task's config files."""
    return ["-f", *(str(config_path(task, name)) for name in CONFIG_FILES)]


def config_value(task, section, name, file="settings.cfg"):
    """A setting from a task's config file, e.g. config_value("task1", "GLOBAL", "updates").

    Names repeat across sections (optimizeValue is under both OPTIMIZER_ROULETTE and
    OPTIMIZER_TOURNAMENT), so the section is required.
    """
    current = None
    for line in config_path(task, file).read_text().splitlines():
        if line.startswith("% "):
            current = line[2:].strip()
        elif current == section:
            m = re.match(rf"\s*{re.escape(name)}\s*=\s*(\S+)", line)
            if m:
                return m.group(1)
    raise KeyError(f"{section}-{name} not found in {file}")


def parse_seeds(spec):
    """'1-5,8,10-12' -> [1, 2, 3, 4, 5, 8, 10, 11, 12]"""
    seeds = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def format_seeds(seeds):
    """[1, 2, 3, 5, 7, 8] -> '1-3,5,7-8', the inverse of parse_seeds."""
    parts, seeds = [], sorted(seeds)
    i = 0
    while i < len(seeds):
        j = i
        while j + 1 < len(seeds) and seeds[j + 1] == seeds[j] + 1:
            j += 1
        parts.append(str(seeds[i]) if i == j else f"{seeds[i]}-{seeds[j]}")
        i = j + 1
    return ",".join(parts)


def existing_seeds(task):
    """Seeds whose evolutionary run has finished."""
    return sorted(
        int(d.name.split("_")[1])
        for d in runs_dir(task).glob("seed_*")
        if (d / "run_info.json").exists()
    )


# Ctrl-C handling for run_parallel. Once STOPPING is set no new MABE process starts, and the
# running ones (tracked in _running) are terminated.
STOPPING = threading.Event()
_running = set()
_running_lock = threading.Lock()


class Stopped(Exception):
    """MABE was stopped, or never started, because the batch was cancelled."""


def run_mabe(args, cwd, log_path):
    """Run MABE in `cwd` with `args`, sending its console output to `log_path`."""
    if not MABE.exists():
        raise FileNotFoundError(f"{MABE} not found; run ./mbuild from the repo root")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        with _running_lock:
            if STOPPING.is_set():
                raise Stopped()
            proc = subprocess.Popen([str(MABE), *args], cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
            _running.add(proc)
        try:
            returncode = proc.wait()
        finally:
            with _running_lock:
                _running.discard(proc)
    if -returncode in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        # Ctrl-C reaches MABE as well as Python; set STOPPING here too, so this worker
        # thread can't start the next seed before the main thread has reacted
        STOPPING.set()
        raise Stopped()
    if returncode != 0:
        raise RuntimeError(f"MABE exited with {returncode}; see {log_path}")


def run_parallel(fn, seeds, jobs, report):
    """Call fn(seed) for each seed on `jobs` threads, and report(seed, result, error) as each ends.

    On Ctrl-C (or if MABE is killed) seeds not yet started are dropped, running MABE processes
    are terminated, and KeyboardInterrupt is raised once all workers have stopped.
    """
    pool = ThreadPoolExecutor(max_workers=jobs)
    futures = {pool.submit(fn, seed): seed for seed in seeds}
    try:
        for f in as_completed(futures):
            try:
                result, error = f.result(), None
            except Stopped:
                continue
            except Exception as e:  # report and carry on with the other seeds
                result, error = None, e
            report(futures[f], result, error)
    except KeyboardInterrupt:
        STOPPING.set()
    pool.shutdown(wait=not STOPPING.is_set(), cancel_futures=True)
    if STOPPING.is_set():
        with _running_lock:
            for proc in _running:
                proc.terminate()
        pool.shutdown(wait=True)
        raise KeyboardInterrupt


def exit_interrupted():
    """Exit as if killed by Ctrl-C, so that a shell loop running this script stops too."""
    sys.stdout.flush()
    sys.stderr.flush()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGINT)


def git_state():
    """Current commit hash and whether the working tree has uncommitted changes.

    Run outputs and notebooks are left out: a running batch creates the former itself, and
    neither affects a run.
    """
    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = git("status", "--porcelain", "--", ".", ":(exclude,glob)animats/*/runs/**",
                ":(exclude)animats/notebooks")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(dirty)}
