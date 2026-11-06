"""IIT 4.0 counterparts of the Fig. 3 measures of Albantakis et al. (2014), along the lines of descent.

    uv run animats/scripts/iit4.py                         # all tasks and finished seeds
    uv run animats/scripts/iit4.py --tasks task1 --seeds 1 --jobs 6

Formulation: pyphi's IIT_4_0_2023 preset (Albantakis et al. 2023), used as is. pyphi's default,
IIT_4_0_2026, differs from it in one setting only: φ_s also requires intrinsic information, which a
deterministic system lacks, so every animat would get φ_s = 0. The background is causally
marginalized on the cause side, as IIT 4.0 specifies (IIT 3.0 held it fixed).

For an animat brain in one state, `brain_measures` computes
    n_distinctions, sum_phi_d  distinctions of the whole brain and their Σφ_d (as #concepts, Σφ^Max)
    mc_units                   size of the main complex (MC)
    mc_distinctions            the distinctions in the MC's Φ-structure
    phi_s                      the MC's system integrated information, which selects the complex
                               (in place of Φ^Max)
    big_phi                    Φ, the MC's structure integrated information: Σφ over its
                               distinctions and relations
    unreachable                1 if the whole brain's state has no possible predecessor, else 0
All are 0 when there is no complex. Averaging over visited states, caching, the figures and the
outputs (animats/notebooks/output/iit4/) are in iit_lod.py; so is the brain-state convention
(real motor TPM columns, motors at their actual output).

Two variants of the MC measures, one figure each (fig3.png, fig3_noselfloop.png):
- IIT 4.0 as is: a single unit with a self-loop can be a complex, so mc_units can be 1. In the
  animats such a unit (a memory bit) is usually the main complex, with φ_s ~0.8.
- `_noselfloop` (mc_units_noselfloop, ...): pyphi's single_micro_nodes_with_selfloops_have_phi off,
  so a complex has at least 2 units, as under IIT 3.0.
The setting only decides whether a one-unit system can have φ_s > 0, so the whole-brain
distinctions are the same in both and are computed once.

Other differences from IIT 3.0 that show up in these measures, all part of IIT 4.0:
- Whole-brain distinctions are computed as for IIT 3.0 (hidden mechanisms, whole brain as the
  system), without filtering them for congruence: the whole brain is not a complex, so it has no
  specified state to be congruent with. The MC's distinctions are its Φ-structure's (congruent).
- pyphi's state check is on, as IIT 4.0 requires: a state with no possible predecessor has no
  cause and no Φ-structure. If the whole brain's state is unreachable, its distinctions count as 0
  and `unreachable` is 1, so its weighted mean is the fraction of time spent in such states.
  Candidate complexes whose state is unreachable (with the background marginalized) drop out.
"""

import pyphi
from pyphi.exceptions import StateUnreachableError
from pyphi.system import System
from pyphi.utils import powerset

import iit_lod

NAME = "iit4"
# measure-name suffix: value of single_micro_nodes_with_selfloops_have_phi for the MC measures
VARIANTS = {"": True, "_noselfloop": False}
MC_MEASURES = ["mc_units", "mc_distinctions", "phi_s", "big_phi"]
MEASURES = ["n_distinctions", "sum_phi_d", *(m + v for v in VARIANTS for m in MC_MEASURES), "unreachable"]
WHOLE_BRAIN_ROWS = {  # column: (y label, y range); None fits the data, as there is no paper to match
    "fitness": ("Fitness (%)", (40, 100)),
    "n_distinctions": ("⟨#distinctions⟩", None),
    "sum_phi_d": ("⟨Σφ$_d$⟩", None),
}
FIGURES = {  # one figure per variant; the whole-brain rows are the same in both
    f"fig3{v}": {**WHOLE_BRAIN_ROWS,
                 f"mc_units{v}": ("⟨MC #units⟩", None),
                 f"mc_distinctions{v}": ("⟨MC #distinctions⟩", None),
                 f"phi_s{v}": ("⟨φ$_s$⟩", None),
                 f"big_phi{v}": ("⟨Φ⟩", None)}
    for v in VARIANTS
}


def brain_measures(substrate, state, hidden):
    """The MEASURES of one brain in one state, as a dict. hidden: indices of the hidden nodes."""
    with pyphi.config.override(**pyphi.iit4_2023):
        try:
            system = System(substrate, state)
            phis = [float(system.distinction(m).phi) for m in powerset(hidden, nonempty=True)]
            unreachable = 0
        except StateUnreachableError:
            phis, unreachable = [], 1
    measures = {"n_distinctions": sum(1 for p in phis if p > 0), "sum_phi_d": sum(phis)}
    for variant, selfloops in VARIANTS.items():
        with pyphi.config.override(**pyphi.iit4_2023, single_micro_nodes_with_selfloops_have_phi=selfloops):
            found = substrate.complexes(state)  # empty if there is no complex
            mc = {"mc_units": 0, "mc_distinctions": 0, "phi_s": 0.0, "big_phi": 0.0}
            if found:
                analysis = pyphi.analyze(substrate, state, subset=found[0].node_indices)
                mc = {"mc_units": len(found[0].node_indices),
                      "mc_distinctions": sum(1 for d in analysis.ces.distinctions if d.phi > 0),
                      "phi_s": analysis.phi,
                      "big_phi": analysis.big_phi}
        measures.update({m + variant: v for m, v in mc.items()})
    return {**measures, "unreachable": unreachable}


def lod_measures(tasks=iit_lod.TASKS, seeds=None, jobs=10):
    """{task: DataFrame} of the MEASURES along every LOD; see iit_lod.lod_measures."""
    import iit4  # this module by name, also when run as a script, so workers can import it
    return iit_lod.lod_measures(iit4, tasks, seeds, jobs)


if __name__ == "__main__":
    import iit4
    iit_lod.main(iit4, "IIT 4.0 counterparts of the paper's Fig. 3 measures along the lines of descent.")
