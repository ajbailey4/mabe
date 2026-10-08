# MABE — Albantakis Animat Replication

## Purpose of this repo

This is a working copy of [MABE](https://github.com/Hintzelab/MABE) (Modular Agent Based
Evolver) being used to replicate the animat experiments from Albantakis et al.'s work on
integrated information (IIT) in evolved agents — the classic "catch/avoid falling blocks"
task controlled by a Markov Brain. MABE is a reasonable fit for this because it's built by
the same lab lineage that produced the original animat codebase, and its `BlockCatchWorld` +
`MarkovBrain` + `CircularGenome` + `RouletteOptimizer` combination maps closely onto the
paper's setup natively (deterministic hidden Markov gates, byte-valued circular genome with
start-codon-delimited genes, motor encoding where both-on/both-off are redundant, periodic
1D world, roulette-wheel selection without elitism).

Longer term, the intent is to evolve animats under this task, then run IIT/Φ analysis
(connectome + state-transition extraction) on animats sampled from the line of descent —
mirroring the original paper's methodology.

## MABE's architecture

MABE is a plugin-style framework. Each conceptual piece has an abstract base class in
`code/<Module>/Abstract<Module>.h` and swappable concrete implementations in subfolders:

- **World** (`code/World/`) — the task/environment; owns `evaluate()`, which drives
  organisms through the brain and assigns fitness. `BlockCatchWorld/` is the animat task.
- **Brain** (`code/Brain/`) — maps sensor inputs to motor outputs each timestep.
  `MarkovBrain/` implements the deterministic hidden Markov gate network used by the paper.
- **Genome** (`code/Genome/`) — the evolvable representation. `CircularGenome/` is the
  byte-valued circular genome that encodes Markov Brain gates via start codons.
- **Organism** (`code/Organism/`) — wraps one genome+brain instance as an individual.
- **Optimizer** (`code/Optimizer/`) — selection/reproduction algorithm.
  `RouletteOptimizer/` implements fitness-proportionate selection without elitism.
- **Archivist** (`code/Archivist/`) — output/logging and line-of-descent tracking.
- **Group** (`code/Group/`) — bundles a population + optimizer + archivist.
- **Analyze** (`code/Analyze/`) — entropy/state-transition/connectome tooling; the closest
  existing bridge toward IIT-style analysis of evolved brains.

The main update loop lives directly in `code/main.cpp` (not its own module): each tick calls
`world->evaluate()`, then per-group `optimize()` and `archive()`.

## Two-layer configuration — important, easy to get wrong

**Compile-time** (`modules.txt` + `./mbuild`): decides which module implementations are
compiled into the `mabe` binary at all. Each module subfolder's `CMakeLists.txt` is gated
behind an `enable_<Group>_<Name>` option, off by default. `modules.txt` marks each module
`*` (built + default), `+` (built, selectable), or `-` (excluded). Run `./mbuild` from the
repo root after editing it — this regenerates `code/module_factories.{h,cpp}` and rebuilds.
A module not enabled here doesn't exist in the binary; setting its name as a runtime
parameter will fail with "could not find" errors.

**Runtime** (`settings.cfg`, `settings_world.cfg`, `settings_organism.cfg`): once a module
is compiled in, which one runs and its parameter values are chosen here.

**Critical gotcha, confirmed by testing**: running `./mabe` with no arguments loads **no
config files at all** — it only uses the compiled-in defaults. Config files are only read
when passed explicitly:

```
./mabe -f <settings.cfg> <settings_world.cfg> <settings_organism.cfg>
```

Editing a `.cfg` file does nothing unless the run that follows uses `-f` to point at it.
`-p PARAM_NAME value` overrides a single parameter on top of loaded files; `-s [prefix]`
regenerates fresh default settings files (reflecting whatever `modules.txt` currently
enables) and exits without running.

## Repo layout conventions

- `animats/<task>/configs/` — tracked parameter set for one task (e.g. `animats/task1/configs/`
  holds `settings.cfg`, `settings_world.cfg`, `settings_organism.cfg`). Hand-tuned settings
  live and get committed here.
- `animats/<task>/runs/seed_NN/` — one evolutionary run per random seed. **Run results are
  committed** so analysis can happen from any device. Only `runs/*/raw/` (MABE console logs
  and intermediate files) is gitignored. If the data gets too large for the main branch, it
  may move to a separate branch or LFS. Check before assuming it's tracked.
- `animats/scripts/` — the pipeline (see below). `animats/notebooks/` — analysis notebooks.
- `work/` — gitignored: the compiled `mabe` binary (`./mbuild` writes `work/mabe`) and
  ad-hoc scratch runs. Nothing here needs preserving.
- `.venv/` — gitignored Python environment. Run everything with `uv run`, which finds this
  `.venv` from anywhere in the repo (even with conda's base env active). To rebuild it:
  ```
  rm -rf .venv
  uv venv --python 3.13                      # pyphi 2.0 requires Python >= 3.13
  uv pip install numpy pandas matplotlib ipykernel jupyterlab
  uv pip install -e "$HOME/pyphi[emd]"       # user's pyphi 2.0 clone (not on PyPI), editable
  ```
  The `emd` extra (POT, the earth mover's distance) is required for IIT 3.0; without it
  IIT 3.0 calls fail with "re-install PyPhi with `pyphi[emd]`". Code here uses pyphi 2.0
  names only (`Substrate`, `analyze`, `complexes`). If VS Code's Pylint can't resolve
  `pyphi` after a reinstall, restart the Pylint server.
- `build/`, `mbuild`-generated files, `code/module_factories.*`, `code/Utilities/gitversion.h`
  — all gitignored, regenerated by `./mbuild`.
- `modules.txt` — tracked (not gitignored despite living alongside other ignored build
  artifacts); this is a real source-controlled file, edit and commit it deliberately.

## Experiment and analysis pipeline

```
python3 animats/scripts/evolve.py task1 --seeds 1-50 --jobs 10      # MABE run mode, BlockCatch
.venv/bin/python animats/scripts/extract.py task1 --jobs 10         # MABE analyze mode, twice
# then animats/notebooks/explore.ipynb, using animats/scripts/load.py
```

- **evolve.py** runs MABE once per seed in `runs/seed_NN/` with `GLOBAL-randomSeed = NN`.
  The LODwAP archivist writes `LOD_organisms.csv` (gzipped afterwards), which holds the
  genome of the line-of-descent animat every 512 generations, and `LOD_data.csv`, one row
  of stats (score, per-pattern correct, gate count, genome length...) for the same animats. `run_info.json` (commit
  hash, args, timing) marks a finished run; finished seeds are skipped unless `--force`.
  `--updates N` overrides the run length for quick tests. A full 60,000-generation run
  takes on the order of 1.5+ hours on one core (about 90 ms/generation early on).
  In a terminal it shows one in-place progress bar per running seed (generation, best
  fitness, ETA), read from each seed's `raw/evolve.log`. With output redirected (nohup),
  it prints a status summary every 10 minutes instead, so long runs are best started in
  tmux/screen to keep the live bars.
- **extract.py** loads those genomes back into MABE in analyze mode twice:
  1. `WORLD-worldType = TPM_GENERATOR` sets each brain into every (sensor, hidden) state
     and records the next state. It only instantiates BlockCatch to learn the brain's I/O
     count, so it must get the same three config files as evolution.
  2. `BlockCatch` with `WORLD_BLOCKCATCH_ANALYZE-saveBrainActivity = 1` (added in this
     repo) replays the 128 trials and dumps the brain state at every step.

  It writes `substrates/gen_NNNNN.npz` per animat (state-by-node little-endian TPM for
  pyphi, CM inferred from it, `motor_tpm` with the real motor outputs, visited states and
  counts, full trial activity, score) and an
  `animats.csv` summary with a `tpm_hash` column for caching across identical brains. It
  checks that every recorded trial step matches the TPM, and fails loudly if not.
- **Node order** is MABE's everywhere: sensors, motors, hidden (`S0 S1 M0 M1 H0..H3` for
  task 1). A recorded "state" at step t is sensors at t, motors 0, hidden at t. Motors are
  stored as 0 following the paper's Methods ("zeroing out ... the motors ... after the
  movement was performed"); the brain never reads its motors (`recurrentOutput = 0`), so
  this only affects state labels. TPM rows differing only in motor state are identical.
- **Sensor and motor TPM columns are 0.5** in the TPM given to pyphi; the real motor
  outputs are kept separately in `motor_tpm`. Why: pyphi drops any candidate system whose
  state it deems unreachable, silently, so a complex just disappears (Φ 0). With 0 sensor
  columns, every state with a sensor on was rejected; with real (deterministic) motor
  columns plus zeroed motor states, many states were too. Tested on random animat-shaped
  brains (Oct 2026): 0 sensors lost all complexes in sensor-on states; zeroed motors with
  real motor columns lost 25 of 73 nonzero states on simulated trials. Any value strictly
  between 0 and 1 gave identical main complexes (400/400 states, values 0.1–0.9) and
  identical whole-brain concept counts and Σφ (45/45), so 0.5 is a convention, not a
  modelling choice. Hidden states with no possible predecessor (e.g. the post-reset 0000
  in some brains) are still unreachable and count as Φ 0.
- **TPMs are deterministic** (0/1), as in the paper. Fine for IIT 3.0; may need handling
  for IIT 4.0.
- **pyphi 2.0 under IIT 3.0**: select it with `with pyphi.config.override(**pyphi.iit3):`
  (or `formalism="IIT_3_0"` on `pyphi.analyze`). On IIT 3.0 results `.phi` is Φ (the
  paper's Φ^Max); on IIT 4.0 results `.phi` is φ_s and `.big_phi` is Φ. Use
  `substrate.complexes(state)` (empty = no complex = Φ 0), not `maximal_complex()`: when no
  complex exists, the latter validates the whole-brain state and raises
  `StateUnreachableForwardsError` on unreachable states. Under IIT 3.0, `complexes()` skips
  overlapping candidates tied for top Φ as indeterminate (pyphi has no IIT 3.0 tie-break).

MABE gotchas found while building this (all handled in extract.py):
- TPM_GENERATOR and other analysis worlds need `GLOBAL-mode analyze`. In `run` mode MABE
  calls the world inside the evolution loop once per generation.
- MABE's loader merges `X_organisms.csv` with a neighbouring `X_data.csv` by ID and errors
  if the data file is missing any organism. So `ARCHIVIST_LODWAP-dataSequence` must include
  every `organismsSequence` generation; both are `:512` in the task configs. (MABE's
  default `:100` vs `:1000` doesn't satisfy this.)
- Loaded organisms get new IDs `0..n-1` in file row order. Output files like
  `TPM_id_<ID>.csv` use the new IDs, not the original ones.
- `MarkovBrain::getTPMforTimepoint()` / `getsampledTPM()` look like a ready-made TPM
  exporter but are stale: they were written for an older `update()` that swapped
  `nodes`/`nextNodes` and produce wrong output now. Use TPM_GENERATOR.

## The paper

Albantakis, Hintze, Koch, Adami & Tononi (2014), "Evolution of Integrated Causal Structures
in Animats Exposed to Environments of Increasing Complexity", PLOS Comput Biol 10(12):
e1003966. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003966

**Full text: [docs/albantakis2014.md](docs/albantakis2014.md)** (CC BY 4.0, converted from
the PLOS XML; equations transcribed to LaTeX, figures linked). Read the relevant section from
there instead of asking the user to paste it. Methods (Animats, Environment, Fitness and
Genetic Algorithm, IIT analysis) is the part the replication follows.

Protocol points that drive the pipeline: 50 independent runs per task, 60,000 generations
each, one line of descent per run; along each LOD, every 512 generations from 0, a TPM is
generated and causal measures (concepts, Φ) are averaged over all brain states the animat
experienced during the 128 test trials, weighted by probability of occurrence.

## Where things currently stand

- **Evolution setup (task 1)**: parameters matched to the paper paragraph by paragraph and
  committed in `animats/task1/configs/`. See git history for the reasoning behind each value.
  The duplication/deletion rate magnitude (per-site vs per-genome) was discussed and left
  as is. Genomes hit the ~20,000-site cap within a few hundred generations under these rates.
- **Pipeline**: evolve → extract → notebook is built and tested end-to-end on short runs.
  Extracted substrates load into pyphi 2.0 as `pyphi.Substrate` (`load.to_substrate`).
- **Next**: launch the 50 full task-1 runs, then exploratory analysis in notebooks with
  pyphi 2.0 (IIT 3.0, IIT 4.0 and other measures). Other tasks from the paper will get
  their own `animats/<task>/` directories.
