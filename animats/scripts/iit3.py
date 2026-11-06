"""IIT 3.0 measures along the lines of descent, as in Albantakis et al. (2014), Fig. 3.

    uv run animats/scripts/iit3.py                         # all tasks and finished seeds
    uv run animats/scripts/iit3.py --tasks task1 --seeds 1 --jobs 6

For an animat brain in one state, `brain_measures` computes
    n_concepts, sum_phi        concepts of the whole brain (sensors, motors, hidden) and their Σφ^Max
    mc_elements, mc_concepts   size of the main complex (MC) and the number of its concepts
    phi_max                    Φ^Max of the MC
All are 0 when there is no complex. Averaging over visited states, caching, the figure and the
outputs (animats/notebooks/output/iit3/) are in iit_lod.py; so is the brain-state convention
(real motor TPM columns, motors at their actual output).

pyphi's IIT 3.0 preset is used as is. Where it differs from the paper's MATLAB program, the
difference is left in (the aim is to see how robust the results are to the formulation). The one
known difference: elements outside a candidate complex are held at their current state on the
cause side too, where the MATLAB program held them at their actual past state.

Whole-brain concepts only consider mechanisms made of hidden nodes: sensors have no inputs and
motors no outputs within the brain, so they can't be (part of) a concept (paper, Fig. 2B). This
gives identical concepts (checked on 80 random brain-states) and is ~8x faster.

pyphi's state check is off (validate_system_states): pyphi refuses to analyse a system whose state
has no possible predecessor given the background, and such a candidate complex drops out. The
paper evaluates every state the animat experienced, including the post-reset state at the start of
each trial. In a 150-state sample (motors zeroed) the check changed Φ in 8 states.
"""

import pyphi
from pyphi.system import System
from pyphi.utils import powerset

import iit_lod

NAME = "iit3"
MEASURES = ["n_concepts", "sum_phi", "mc_elements", "mc_concepts", "phi_max"]
ROWS = {  # column: (y label, the paper's y range)
    "fitness": ("Fitness (%)", (40, 100)),
    "n_concepts": ("⟨#concepts⟩", (0, 6)),
    "sum_phi": ("⟨Σφ$^{Max}$⟩", (0, 1.2)),
    "mc_elements": ("⟨#elements⟩", (0, 2)),
    "mc_concepts": ("⟨#concepts⟩", (0, 2.5)),
    "phi_max": ("⟨Φ$^{Max}$⟩", (0, 0.65)),
}
FIGURES = {"fig3": ROWS}


def brain_measures(substrate, state, hidden):
    """The MEASURES of one brain in one state, as a dict. hidden: indices of the hidden nodes."""
    with pyphi.config.override(**pyphi.iit3, validate_system_states=False):
        system = System(substrate, state)
        concepts = [system.distinction(m) for m in powerset(hidden, nonempty=True)]
        found = substrate.complexes(state)  # empty if there is no complex
        mc_concepts = 0
        if found:
            distinctions = found[0].sia.distinctions
            if distinctions is None:  # only if the analysis short-circuited, which Φ > 0 rules out
                distinctions = substrate.ces(state, subset=found[0].node_indices).distinctions
            mc_concepts = sum(1 for c in distinctions if c.phi > 0)
    return {
        "n_concepts": sum(1 for c in concepts if c.phi > 0),
        "sum_phi": float(sum(c.phi for c in concepts)),
        "mc_elements": len(found[0].node_indices) if found else 0,
        "mc_concepts": mc_concepts,
        "phi_max": float(found[0].phi) if found else 0.0,
    }


def lod_measures(tasks=iit_lod.TASKS, seeds=None, jobs=10):
    """{task: DataFrame} of the MEASURES along every LOD; see iit_lod.lod_measures."""
    import iit3  # this module by name, also when run as a script, so workers can import it
    return iit_lod.lod_measures(iit3, tasks, seeds, jobs)


if __name__ == "__main__":
    import iit3
    iit_lod.main(iit3, "IIT 3.0 measures along the lines of descent and the paper's Fig. 3.")
