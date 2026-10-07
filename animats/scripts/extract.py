#!/usr/bin/env python3
"""Turn each finished run's line of descent into pyphi-ready networks.

For every animat in runs/seed_NN/LOD_organisms.csv.gz (one per 512 generations) this:
  1. runs MABE's TPM_GENERATOR world to get the brain's response to every
     (sensor, hidden) state,
  2. reruns the BlockCatch task (analyze mode) to record the brain state at
     every step of all 128 trials,
  3. writes runs/seed_NN/networks/gen_NNNNN.npz and a summary row in
     runs/seed_NN/animats.csv.

Contents of each .npz (node order everywhere is MABE's: sensors, motors, hidden,
labelled S0 S1 M0 M1 H0 H1 H2 H3 for task 1):
  tpm             (2^N, N) state-by-node TPM, little-endian state index (pyphi's
                  convention): row = current state, column j = P(node j is on next step)
  cm              (N, N) connectivity matrix inferred from the TPM: cm[i, j] = 1 if
                  node i's state can change node j's next state
  node_labels     (N,) str
  visited_states  (K, N) distinct states the brain was in during the 128 trials
  visited_counts  (K,) how many time steps each visited state occurred
  activity        (T, N) every time step of every trial, with activity_trial and
                  activity_t giving the trial number and step within the trial
  seed, generation, mabe_id, score   scalars (score = fraction of trials correct)

Conventions:
  - A "state" at step t is sensors at t, motors 0, hidden at t. Motors are stored
    as 0 to follow the paper, which zeroes them "after the movement was performed".
    The brain never reads its motors, so this only affects how states are labelled;
    the motor output at step t is still in the TPM: tpm[state_t, motor columns].
  - Motors don't feed back (BRAIN_MARKOV-recurrentOutput = 0), so TPM rows that
    differ only in motor state are identical.
  - Nothing inside the brain drives the sensors, so their TPM columns are 0.

Examples:
    .venv/bin/python animats/scripts/extract.py task1             # all finished seeds
    .venv/bin/python animats/scripts/extract.py task1 --seeds 3 --force
"""

import argparse
import csv
import gzip
import hashlib
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from common import config_args, existing_seeds, parse_seeds, run_mabe, seed_dir

csv.field_size_limit(sys.maxsize)  # genome columns are very long


def read_lod_organisms(path):
    """(original ID, generation) for each row, in file order."""
    with open(path, newline="") as f:
        return [(int(r["ID"]), int(r["update"])) for r in csv.DictReader(f)]


def build_tpm(tpm_csv):
    """Expand TPM_GENERATOR's packed output into a full 2^N x N state-by-node TPM.

    Packed rows look like  before="s0,s1,-1,-1,h0,h1,h2,h3"  after="-1,-1,m0,m1,h0,h1,h2,h3",
    where -1 marks values that don't exist (motors don't feed back; sensors have no inputs).
    """
    rows = {}
    with open(tpm_csv, newline="") as f:
        for r in csv.DictReader(f):
            before = tuple(int(x) for x in r["before"].split(","))
            after = [int(x) for x in r["after"].split(",")]
            rows[before] = after
    befores = np.array(list(rows))
    afters = np.array(list(rows.values()))
    n = befores.shape[1]
    not_read = np.all(befores == -1, axis=0)   # motors: current state never set
    not_driven = np.all(afters == -1, axis=0)  # sensors: next state never produced

    tpm = np.zeros((2 ** n, n))
    for index in range(2 ** n):
        state = [(index >> i) & 1 for i in range(n)]
        key = tuple(-1 if not_read[i] else state[i] for i in range(n))
        tpm[index] = np.where(not_driven, 0, rows[key])
    return tpm, not_read, not_driven


def infer_cm(tpm):
    n = tpm.shape[1]
    index = np.arange(tpm.shape[0])
    cm = np.zeros((n, n), dtype=int)
    for i in range(n):
        flipped = tpm[index ^ (1 << i)]
        cm[i] = np.any(flipped != tpm, axis=0)
    return cm


def state_index(states):
    return (states * (1 << np.arange(states.shape[1]))).sum(axis=1)


def check_activity_against_tpm(tpm, trial, activity, not_driven):
    """Every recorded step must be the TPM's prediction from the previous step."""
    same_trial = trial[1:] == trial[:-1]
    predicted = tpm[state_index(activity[:-1])][same_trial][:, ~not_driven]
    observed = activity[1:][same_trial][:, ~not_driven]
    return int(np.sum(np.any(predicted != observed, axis=1)))


def extract(task, seed, force):
    out = seed_dir(task, seed)
    if (out / "animats.csv").exists() and not force:
        return seed, "skipped (already extracted)"

    raw = out / "raw" / "extract"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)
    # MABE can't read gzip, so unpack the genomes for it
    with gzip.open(out / "LOD_organisms.csv.gz", "rb") as src, open(raw / "LOD_organisms.csv", "wb") as dst:
        shutil.copyfileobj(src, dst)
    lod = read_lod_organisms(raw / "LOD_organisms.csv")

    load = [*config_args(task), "-p", "GLOBAL-mode", "analyze", "-p", "GLOBAL-initPop", "'LOD_organisms.csv'"]
    run_mabe(load + [
        "-p", "WORLD-worldType", "TPM_GENERATOR",
        "-p", "WORLD_TPM_GENERATOR-worldName", "BlockCatch",
        "-p", "WORLD_TPM_GENERATOR-numberOfHidden", "4",
        "-p", "WORLD_TPM_GENERATOR-outputMode", "packed",
    ], cwd=raw, log_path=raw / "tpm_generator.log")
    run_mabe(load + [
        "-p", "WORLD_BLOCKCATCH_ANALYZE-saveBrainActivity", "1",
        "-p", "WORLD_BLOCKCATCH_ANALYZE-saveBrainStructureAndConnectome", "0",
        "-p", "WORLD_BLOCKCATCH_ANALYZE-saveStateToState", "0",
    ], cwd=raw, log_path=raw / "blockcatch_analyze.log")
    for f in raw.glob("CatchPassVisualize_*.txt"):  # ~0.5 MB per animat, not needed
        f.unlink()

    # MABE gives loaded organisms new IDs 0..n-1 in file order; map them back.
    # check_activity_against_tpm catches any mismatch between the two MABE passes.
    with open(raw / "brainActivity_scores.csv", newline="") as f:
        scores = {int(r["ID"]): float(r["score"]) for r in csv.DictReader(f)}
    if len(scores) != len(lod) or len(list(raw.glob("TPM_id_*.csv"))) != len(lod):
        raise RuntimeError(f"expected {len(lod)} animats from MABE; see logs in {raw}")

    networks = out / "networks"
    if networks.exists():
        shutil.rmtree(networks)
    networks.mkdir()
    summary = []
    for new_id, (mabe_id, generation) in enumerate(lod):
        tpm, not_read, not_driven = build_tpm(raw / f"TPM_id_{new_id}.csv")
        n = tpm.shape[1]
        n_sensors, n_motors = int(not_driven.sum()), int(not_read.sum())
        labels = ([f"S{i}" for i in range(n_sensors)] + [f"M{i}" for i in range(n_motors)]
                  + [f"H{i}" for i in range(n - n_sensors - n_motors)])

        table = np.loadtxt(raw / f"brainActivity_id_{new_id}.csv", delimiter=",", skiprows=1, dtype=int)
        trial, t, activity = table[:, 0], table[:, 1], table[:, 2:]
        # the check needs the recorded motor outputs, so zero motors only afterwards
        mismatches = check_activity_against_tpm(tpm, trial, activity, not_driven)
        if mismatches:
            raise RuntimeError(f"generation {generation}: {mismatches} recorded steps disagree with the TPM")
        activity[:, not_read] = 0
        visited, counts = np.unique(activity, axis=0, return_counts=True)

        cm = infer_cm(tpm)
        np.savez_compressed(
            networks / f"gen_{generation:05d}.npz",
            tpm=tpm, cm=cm, node_labels=np.array(labels),
            visited_states=visited.astype(np.int8), visited_counts=counts,
            activity=activity.astype(np.int8), activity_trial=trial.astype(np.int16),
            activity_t=t.astype(np.int16),
            seed=seed, generation=generation, mabe_id=mabe_id, score=scores[new_id],
        )
        summary.append({
            "generation": generation, "mabe_id": mabe_id, "score": scores[new_id],
            "n_connections": int(cm.sum()), "n_visited_states": len(visited),
            # identical brains share a hash, so analysis can reuse results across generations
            "tpm_hash": hashlib.sha1(tpm.tobytes()).hexdigest()[:12],
        })

    with open(out / "animats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return seed, f"{len(summary)} animats extracted"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", help="task name, e.g. task1")
    parser.add_argument("--seeds", help="seeds to extract (default: every finished run)")
    parser.add_argument("--jobs", type=int, default=1, help="number of seeds in parallel")
    parser.add_argument("--force", action="store_true", help="re-extract seeds that already have animats.csv")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds) if args.seeds else existing_seeds(args.task)
    if not seeds:
        sys.exit("no finished runs found")
    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(extract, args.task, s, args.force): s for s in seeds}
        for f in as_completed(futures):
            try:
                seed, status = f.result()
                print(f"seed {seed:02d}: {status}", flush=True)
            except Exception as e:
                failed.append(futures[f])
                print(f"seed {futures[f]:02d}: FAILED: {e}", flush=True)
    if failed:
        sys.exit(f"failed seeds: {sorted(failed)}")


if __name__ == "__main__":
    main()
