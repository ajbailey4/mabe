"""Paths and helpers shared by evolve.py, extract.py and load.py.

Layout (see CLAUDE.md):
    animats/<task>/configs/          settings.cfg, settings_world.cfg, settings_organism.cfg
    animats/<task>/runs/seed_NN/     one evolutionary run per seed
        run_info.json                written by evolve.py when the run finishes
        LOD_organisms.csv.gz         genomes along the line of descent, every 512 generations
        LOD_data.csv, pop.csv        per-generation stats
        animats.csv                  written by extract.py: one row per sampled animat
        networks/gen_NNNNN.npz       written by extract.py: TPM, CM, visited states
        raw/                         (gitignored) MABE logs and intermediate files
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANIMATS = REPO / "animats"
MABE = REPO / "work" / "mabe"
CONFIG_FILES = ("settings.cfg", "settings_world.cfg", "settings_organism.cfg")


def task_dir(task):
    return ANIMATS / task


def runs_dir(task):
    return task_dir(task) / "runs"


def seed_dir(task, seed):
    return runs_dir(task) / f"seed_{seed:02d}"


def config_args(task):
    """The `-f ...` arguments that load a task's config files."""
    paths = [task_dir(task) / "configs" / name for name in CONFIG_FILES]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    return ["-f", *map(str, paths)]


def config_value(task, section, name, file="settings.cfg"):
    """A setting from a task's config file, e.g. config_value("task1", "GLOBAL", "updates").

    Names repeat across sections (optimizeValue is under both OPTIMIZER_ROULETTE and
    OPTIMIZER_TOURNAMENT), so the section is required.
    """
    current = None
    for line in (task_dir(task) / "configs" / file).read_text().splitlines():
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


def existing_seeds(task):
    """Seeds whose evolutionary run has finished."""
    return sorted(
        int(d.name.split("_")[1])
        for d in runs_dir(task).glob("seed_*")
        if (d / "run_info.json").exists()
    )


def run_mabe(args, cwd, log_path):
    """Run MABE in `cwd` with `args`, sending its console output to `log_path`."""
    if not MABE.exists():
        raise FileNotFoundError(f"{MABE} not found; run ./mbuild from the repo root")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        result = subprocess.run([str(MABE), *args], cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"MABE exited with {result.returncode}; see {log_path}")


def git_state():
    """Current commit hash and whether the working tree has uncommitted changes."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
