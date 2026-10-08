"""Load extracted animats for analysis (e.g. from a notebook).

    import sys; sys.path.insert(0, "../scripts")   # from animats/notebooks/
    from load import list_animats, load_animat, to_substrate

    lod = list_animats("task1", seed=1)            # DataFrame, one row per sampled generation
    a = load_animat("task1", seed=1, generation=59904)
    substrate = to_substrate(a)                     # pyphi.Substrate
    for state, p in zip(a["visited_states"], a["visited_probs"]):
        ...                                         # state-dependent measure, weighted by p

Only to_substrate() imports pyphi, so everything else works without it installed.
"""

import json

import numpy as np
import pandas as pd
import pyphi

from common import existing_seeds, seed_dir


def list_seeds(task):
    """Seeds with a finished evolutionary run."""
    return existing_seeds(task)


def list_animats(task, seed):
    """animats.csv for one run: generation, mabe_id, score, n_connections, n_visited_states, tpm_hash."""
    return pd.read_csv(seed_dir(task, seed) / "animats.csv")


def all_animats(task):
    """animats.csv for every extracted run, with a seed column."""
    frames = [list_animats(task, s).assign(seed=s)
              for s in list_seeds(task) if (seed_dir(task, s) / "animats.csv").exists()]
    return pd.concat(frames, ignore_index=True)


def run_info(task, seed):
    return json.loads((seed_dir(task, seed) / "run_info.json").read_text())


def lod_data(task, seed):
    """MABE's LOD_data.csv: score and other stats along the line of descent."""
    return pd.read_csv(seed_dir(task, seed) / "LOD_data.csv")


def load_animat(task, seed, generation):
    """Everything extract.py saved for one animat, as a dict of numpy arrays and scalars.

    Adds visited_probs = visited_counts / total steps, for weighting by probability of occurrence.
    """
    path = seed_dir(task, seed) / "substrates" / f"gen_{generation:05d}.npz"
    with np.load(path) as data:
        a = {k: (data[k].item() if data[k].ndim == 0 else data[k]) for k in data.files}
    a["node_labels"] = [str(x) for x in a["node_labels"]]
    a["visited_probs"] = a["visited_counts"] / a["visited_counts"].sum()
    return a


def to_substrate(animat):
    """A pyphi.Substrate built from the animat's TPM, connectivity matrix and node labels."""
    return pyphi.Substrate(animat["tpm"], cm=animat["cm"], node_labels=animat["node_labels"])
