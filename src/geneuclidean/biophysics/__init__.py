"""Differentiable 3D Protein–Pocket–Ligand Biophysical & Electrical Force-Field Digital Twin.

Implements PEP 562 module-level lazy loading (``__getattr__`` / ``__dir__``) with full
PEP 561 static type support (``if TYPE_CHECKING:``) so ``import geneuclidean.biophysics``
completes in ``< 5 ms`` with **zero** eager imports of ``torch``, ``numpy``, ``scipy``,
``sklearn``, ``rdkit``, ``matplotlib``, ``Bio``, ``transformers``, or ``peft``.

Public Pillars:
1. **Differentiable ``torch.float32`` Force-Field & Electrostatics Engine**
   (:mod:`geneuclidean.biophysics.forcefield`):
   :class:`BiophysicalAtomParams`, :class:`BiophysicalHamiltonian`,
   :class:`ComplexEnergyBreakdown`, :func:`parameterize_ligand_biophysics`,
   and :func:`parameterize_pocket_biophysics`.
2. **Stateful 3D Protein–Ligand Digital Twin, Relaxer, Langevin Sampler & OpenMM Bridge**
   (:mod:`geneuclidean.biophysics.protein_ligand_twin`):
   :class:`ProteinLigandDigitalTwin`, :class:`TwinPhysicsInspection`,
   :class:`TwinRelaxationResult`, :class:`TwinEnsembleSummary`,
   and :class:`OpenMMTwinAdapter`.
3. **3D ESP Scalar/Vector Field, Typed Interaction Network, Publication Figure & WebGL Studio**
   (:mod:`geneuclidean.biophysics.interaction_viz`):
   :class:`ElectrostaticPotentialGrid`, :class:`TypedBiophysicalInteraction`,
   :func:`compute_electrostatic_potential_grid`, :func:`extract_typed_interaction_network`,
   :func:`render_biophysical_twin_figure`, :func:`export_web_twin_scene_json`,
   and :func:`export_interactive_twin_html`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # --- Item B: forcefield, protein_ligand_twin, interaction_viz ---
    from geneuclidean.biophysics.forcefield import (
        COULOMB_KCAL_ANGSTROM as COULOMB_KCAL_ANGSTROM,
    )
    from geneuclidean.biophysics.forcefield import (
        KB_KCAL_MOL_K as KB_KCAL_MOL_K,
    )
    from geneuclidean.biophysics.forcefield import (
        KCAL_MOL_E_TO_VOLT as KCAL_MOL_E_TO_VOLT,
    )
    from geneuclidean.biophysics.forcefield import (
        BiophysicalAtomParams as BiophysicalAtomParams,
    )
    from geneuclidean.biophysics.forcefield import (
        BiophysicalHamiltonian as BiophysicalHamiltonian,
    )
    from geneuclidean.biophysics.forcefield import (
        ComplexEnergyBreakdown as ComplexEnergyBreakdown,
    )
    from geneuclidean.biophysics.forcefield import (
        parameterize_ligand_biophysics as parameterize_ligand_biophysics,
    )
    from geneuclidean.biophysics.forcefield import (
        parameterize_pocket_biophysics as parameterize_pocket_biophysics,
    )
    from geneuclidean.biophysics.interaction_viz import (
        ElectrostaticPotentialGrid as ElectrostaticPotentialGrid,
    )
    from geneuclidean.biophysics.interaction_viz import (
        TypedBiophysicalInteraction as TypedBiophysicalInteraction,
    )
    from geneuclidean.biophysics.interaction_viz import (
        compute_electrostatic_potential_grid as compute_electrostatic_potential_grid,
    )
    from geneuclidean.biophysics.interaction_viz import (
        export_interactive_twin_html as export_interactive_twin_html,
    )
    from geneuclidean.biophysics.interaction_viz import (
        export_web_twin_scene_json as export_web_twin_scene_json,
    )
    from geneuclidean.biophysics.interaction_viz import (
        extract_typed_interaction_network as extract_typed_interaction_network,
    )
    from geneuclidean.biophysics.interaction_viz import (
        render_biophysical_twin_figure as render_biophysical_twin_figure,
    )
    from geneuclidean.biophysics.protein_ligand_twin import (
        OpenMMTwinAdapter as OpenMMTwinAdapter,
    )
    from geneuclidean.biophysics.protein_ligand_twin import (
        ProteinLigandDigitalTwin as ProteinLigandDigitalTwin,
    )
    from geneuclidean.biophysics.protein_ligand_twin import (
        TwinEnsembleSummary as TwinEnsembleSummary,
    )
    from geneuclidean.biophysics.protein_ligand_twin import (
        TwinPhysicsInspection as TwinPhysicsInspection,
    )
    from geneuclidean.biophysics.protein_ligand_twin import (
        TwinRelaxationResult as TwinRelaxationResult,
    )

_LAZY_SUBMODULES: frozenset[str] = frozenset(
    {
        # Item B submodules
        "forcefield",
        "interaction_viz",
        "protein_ligand_twin",
    }
)

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Item B: Force-field engine (`geneuclidean.biophysics.forcefield`)
    "BiophysicalAtomParams": ("geneuclidean.biophysics.forcefield", "BiophysicalAtomParams"),
    "BiophysicalHamiltonian": ("geneuclidean.biophysics.forcefield", "BiophysicalHamiltonian"),
    "COULOMB_KCAL_ANGSTROM": ("geneuclidean.biophysics.forcefield", "COULOMB_KCAL_ANGSTROM"),
    "ComplexEnergyBreakdown": ("geneuclidean.biophysics.forcefield", "ComplexEnergyBreakdown"),
    "KB_KCAL_MOL_K": ("geneuclidean.biophysics.forcefield", "KB_KCAL_MOL_K"),
    "KCAL_MOL_E_TO_VOLT": ("geneuclidean.biophysics.forcefield", "KCAL_MOL_E_TO_VOLT"),
    "parameterize_ligand_biophysics": (
        "geneuclidean.biophysics.forcefield",
        "parameterize_ligand_biophysics",
    ),
    "parameterize_pocket_biophysics": (
        "geneuclidean.biophysics.forcefield",
        "parameterize_pocket_biophysics",
    ),
    # Item B: Stateful digital twin (`geneuclidean.biophysics.protein_ligand_twin`)
    "OpenMMTwinAdapter": ("geneuclidean.biophysics.protein_ligand_twin", "OpenMMTwinAdapter"),
    "ProteinLigandDigitalTwin": (
        "geneuclidean.biophysics.protein_ligand_twin",
        "ProteinLigandDigitalTwin",
    ),
    "TwinEnsembleSummary": ("geneuclidean.biophysics.protein_ligand_twin", "TwinEnsembleSummary"),
    "TwinPhysicsInspection": (
        "geneuclidean.biophysics.protein_ligand_twin",
        "TwinPhysicsInspection",
    ),
    "TwinRelaxationResult": (
        "geneuclidean.biophysics.protein_ligand_twin",
        "TwinRelaxationResult",
    ),
    # Item B: ESP grid, typed interaction network & WebGL/Matplotlib viz (`interaction_viz`)
    "ElectrostaticPotentialGrid": (
        "geneuclidean.biophysics.interaction_viz",
        "ElectrostaticPotentialGrid",
    ),
    "TypedBiophysicalInteraction": (
        "geneuclidean.biophysics.interaction_viz",
        "TypedBiophysicalInteraction",
    ),
    "compute_electrostatic_potential_grid": (
        "geneuclidean.biophysics.interaction_viz",
        "compute_electrostatic_potential_grid",
    ),
    "export_interactive_twin_html": (
        "geneuclidean.biophysics.interaction_viz",
        "export_interactive_twin_html",
    ),
    "export_web_twin_scene_json": (
        "geneuclidean.biophysics.interaction_viz",
        "export_web_twin_scene_json",
    ),
    "extract_typed_interaction_network": (
        "geneuclidean.biophysics.interaction_viz",
        "extract_typed_interaction_network",
    ),
    "render_biophysical_twin_figure": (
        "geneuclidean.biophysics.interaction_viz",
        "render_biophysical_twin_figure",
    ),
}

__all__ = sorted(set(_LAZY_SUBMODULES) | set(_LAZY_ATTRS.keys()))


def __getattr__(name: str) -> Any:
    """Lazily resolve ``geneuclidean.biophysics`` classes, functions, and submodules (PEP 562)."""
    if name in _LAZY_SUBMODULES:
        submod = importlib.import_module(f"geneuclidean.biophysics.{name}")
        globals()[name] = submod
        return submod
    if name in _LAZY_ATTRS:
        mod_path, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(mod_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose complete lazy biophysics surface to ``dir(geneuclidean.biophysics)``."""
    return sorted(set(globals().keys()) | set(__all__))
