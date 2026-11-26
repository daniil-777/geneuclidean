"""Reaction-space Combinatorial Thompson Sampling over :math:`N_A \\times N_B` virtual spaces.

Implements the Klaric, Goldman, Kramer, Riley & Walters (*JCIM* 2024) factorized
Bayesian reagent-posterior paradigm directly atop GenEuclidean's 16 curated
:data:`~geneuclidean.synthesis.reactions.ROBUST_MEDCHEM_REACTIONS`:

- Maintains independent conjugate Normal-Normal posteriors :math:`(\\mu_r, \\sigma_r^2)`
  over individual reagents :math:`r \\in \\mathcal{A} \\cup \\mathcal{B}` rather than
  enumerating the :math:`O(N_A N_B)` Cartesian product.
- **Strict Ground-Truth Isolation:** Never enumerates :math:`N_A \\times N_B` and never
  invokes ``oracle_fn`` on any molecule outside the budgeted proposals
  (:math:`\\le \\text{total\\_budget}` calls).
- Supports physiological reagent uncharging (``rdMolStandardize.Uncharger`` — essential
  for pre-protonated ZINC250k carboxylates), balanced warm-up (:math:`w` passes),
  thermal exploration floor, ``"argmax"`` and ``"boltzmann_roulette"`` (AstraZeneca 2024)
  selection, ``triage_molecule`` MedChem hygiene gating, and an **observed-only
  additive main-effects** :math:`R^2` epistasis diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

from geneuclidean.cns.medchem_triage import triage_molecule
from geneuclidean.synthesis.reactions import ROBUST_MEDCHEM_REACTIONS

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]
RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]

TSSelectionMode = Literal["argmax", "boltzmann_roulette"]


@dataclass
class ReagentPosterior:
    """Conjugate Normal-Normal posterior state for a single building-block reagent."""

    reagent_id: str
    smiles: str
    role: Literal["A", "B"]
    mean: float
    std: float
    n_obs: int = 0
    sum_y: float = 0.0
    sum_y2: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "reagent_id": self.reagent_id,
            "smiles": self.smiles,
            "role": self.role,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "n_obs": self.n_obs,
        }


@dataclass(frozen=True)
class CombinatorialTSProduct:
    """Single synthesized & oracle-evaluated virtual product with full reagent provenance."""

    product_smiles: str
    reaction_name: str
    reactant_a_id: str
    reactant_a_smiles: str
    reactant_b_id: str
    reactant_b_smiles: str
    score: float
    step_index: int
    phase: Literal["warmup", "sampling"]
    triage_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_smiles": self.product_smiles,
            "reaction_name": self.reaction_name,
            "reactant_a_id": self.reactant_a_id,
            "reactant_a_smiles": self.reactant_a_smiles,
            "reactant_b_id": self.reactant_b_id,
            "reactant_b_smiles": self.reactant_b_smiles,
            "score": round(self.score, 4),
            "step_index": self.step_index,
            "phase": self.phase,
            "triage_passed": self.triage_passed,
        }


@dataclass(frozen=True)
class CombinatorialTSResult:
    """Complete result & epistasis telemetry from a Combinatorial Thompson Sampling run."""

    reaction_name: str
    selection_mode: TSSelectionMode
    n_reactants_a: int
    n_reactants_b: int
    n_virtual_space: int
    n_oracle_calls: int
    n_warmup_calls: int
    n_sampling_calls: int
    n_reaction_failures: int
    n_triage_filtered: int
    fraction_explored: float
    best_score: float
    best_product: CombinatorialTSProduct | None
    top_products: tuple[CombinatorialTSProduct, ...]
    all_products: tuple[CombinatorialTSProduct, ...]
    reagent_additivity_r2: float
    epistasis_warning: bool
    reagent_posteriors_a: tuple[ReagentPosterior, ...]
    reagent_posteriors_b: tuple[ReagentPosterior, ...]
    ledger_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_name": self.reaction_name,
            "selection_mode": self.selection_mode,
            "n_reactants_a": self.n_reactants_a,
            "n_reactants_b": self.n_reactants_b,
            "n_virtual_space": self.n_virtual_space,
            "n_oracle_calls": self.n_oracle_calls,
            "n_warmup_calls": self.n_warmup_calls,
            "n_sampling_calls": self.n_sampling_calls,
            "n_reaction_failures": self.n_reaction_failures,
            "n_triage_filtered": self.n_triage_filtered,
            "fraction_explored": round(self.fraction_explored, 6),
            "best_score": round(self.best_score, 4) if math.isfinite(self.best_score) else None,
            "best_product": self.best_product.to_dict() if self.best_product is not None else None,
            "top_products": [p.to_dict() for p in self.top_products],
            "reagent_additivity_r2": (
                round(self.reagent_additivity_r2, 4)
                if math.isfinite(self.reagent_additivity_r2)
                else None
            ),
            "epistasis_warning": self.epistasis_warning,
            "ledger_sha256": self.ledger_sha256,
            "metadata": dict(self.metadata),
        }


def _resolve_reaction(reaction_name: str, custom_smirks: str | None) -> tuple[str, str, Any]:
    if custom_smirks is not None:
        rxn = AllChem.ReactionFromSmarts(custom_smirks)  # type: ignore[attr-defined]
        if rxn is None:
            raise ValueError(f"Invalid custom_smirks: {custom_smirks!r}")
        return reaction_name, custom_smirks, rxn
    for spec in ROBUST_MEDCHEM_REACTIONS:
        if spec.name == reaction_name:
            rxn = AllChem.ReactionFromSmarts(spec.smirks)  # type: ignore[attr-defined]
            if rxn is None:
                raise RuntimeError(f"Failed to compile built-in SMIRKS for {reaction_name!r}.")
            return spec.name, spec.smirks, rxn
    known = ", ".join(r.name for r in ROBUST_MEDCHEM_REACTIONS)
    raise ValueError(f"Unknown reaction_name {reaction_name!r}. Available: {known}")


def _standardize_reagents(
    raw_smiles: Sequence[str],
    *,
    role: Literal["A", "B"],
    uncharge: bool,
    template_mol: Any | None,
) -> list[tuple[str, str, Chem.Mol]]:
    """Parse, optionally uncharge, deduplicate, and template-check reagents in O(N)."""
    uncharger = rdMolStandardize.Uncharger() if uncharge else None
    seen: set[str] = set()
    parsed: list[tuple[str, str, Chem.Mol]] = []
    for idx, raw in enumerate(raw_smiles):
        if not isinstance(raw, str) or not raw.strip():
            continue
        mol = Chem.MolFromSmiles(raw.strip())
        if mol is None:
            continue
        if uncharger is not None:
            try:
                mol = uncharger.uncharge(mol)
                Chem.SanitizeMol(mol)
            except Exception:
                continue
        if template_mol is not None and not mol.HasSubstructMatch(template_mol):
            continue
        can = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        if can in seen:
            continue
        seen.add(can)
        parsed.append((f"{role}_{idx:04d}", can, mol))
    return parsed


def _compute_additive_r2(
    obs_triples: Sequence[tuple[int, int, float]],
    n_a: int,
    n_b: int,
) -> float:
    """Compute observed-only additive main-effects :math:`R^2` (y ~ mu + alpha_a + beta_b)."""
    n_obs = len(obs_triples)
    if n_obs < 6:
        return float("nan")
    y = np.array([t[2] for t in obs_triples], dtype=np.float64)
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 1e-12:
        return 1.0

    used_a = sorted({t[0] for t in obs_triples})
    used_b = sorted({t[1] for t in obs_triples})
    map_a = {a: i for i, a in enumerate(used_a)}
    map_b = {b: i for i, b in enumerate(used_b)}
    na = len(used_a)
    nb = len(used_b)
    if n_obs <= na + nb:
        # Ridge-regularized design matrix avoids trivial 1.0 overfit when under-determined
        x_mat = np.zeros((n_obs, 1 + na + nb), dtype=np.float64)
        for row_i, (ai, bi, _) in enumerate(obs_triples):
            x_mat[row_i, 0] = 1.0
            x_mat[row_i, 1 + map_a[ai]] = 1.0
            x_mat[row_i, 1 + na + map_b[bi]] = 1.0
        reg = 0.5 * np.eye(1 + na + nb, dtype=np.float64)
        reg[0, 0] = 0.0
        coef = np.linalg.solve(x_mat.T @ x_mat + reg, x_mat.T @ y)
        y_hat = x_mat @ coef
    else:
        # Unregularized exact OLS ANOVA main effects
        x_mat = np.zeros((n_obs, 1 + na + nb), dtype=np.float64)
        for row_i, (ai, bi, _) in enumerate(obs_triples):
            x_mat[row_i, 0] = 1.0
            x_mat[row_i, 1 + map_a[ai]] = 1.0
            x_mat[row_i, 1 + na + map_b[bi]] = 1.0
        coef, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
        y_hat = x_mat @ coef

    ss_res = float(np.sum((y - y_hat) ** 2))
    return float(max(0.0, min(1.0, 1.0 - ss_res / ss_tot)))


class CombinatorialThompsonSampler:
    """Factorized :math:`O(N_A + N_B)` Thompson Sampler over 2-component MedChem reactions."""

    def __init__(
        self,
        reaction_name: str = "amide_coupling_acid_amine",
        *,
        custom_smirks: str | None = None,
        warmup_per_reagent: int = 3,
        warmup_cap_fraction: float = 0.50,
        temperature: float = 1.0,
        roulette_temperature: float = 0.5,
        min_std_floor_ratio: float = 0.10,
        selection_mode: TSSelectionMode = "argmax",
        triage_filter: bool = True,
        uncharge_reactants: bool = True,
        seed: int = 42,
    ) -> None:
        self.reaction_name, self.reaction_smirks, self._rxn_obj = _resolve_reaction(
            reaction_name, custom_smirks
        )
        self.warmup_per_reagent = max(1, int(warmup_per_reagent))
        self.warmup_cap_fraction = float(np.clip(warmup_cap_fraction, 0.05, 0.80))
        self.temperature = max(0.05, float(temperature))
        self.roulette_temperature = max(0.05, float(roulette_temperature))
        self.min_std_floor_ratio = max(0.01, float(min_std_floor_ratio))
        self.selection_mode: TSSelectionMode = selection_mode
        self.triage_filter = bool(triage_filter)
        self.uncharge_reactants = bool(uncharge_reactants)
        self.seed = int(seed)

    def _react_pair(self, mol_a: Chem.Mol, mol_b: Chem.Mol) -> tuple[str | None, bool, bool]:
        """Run 2-component RDKit reaction on ``(mol_a, mol_b)``.

        Returns ``(product_smiles_or_none, filtered_by_triage, triage_passed)``.
        """
        try:
            prods = self._rxn_obj.RunReactants((mol_a, mol_b))
        except Exception:
            return None, False, False
        for prod_tuple in prods:
            if not prod_tuple:
                continue
            pmol = prod_tuple[0]
            try:
                Chem.SanitizeMol(pmol)
                psmi = str(Chem.MolToSmiles(pmol, canonical=True, isomericSmiles=True))
            except Exception:
                continue
            if not psmi or "." in psmi:
                continue
            tr_ok = bool(triage_molecule(psmi).triage_pass)
            if self.triage_filter and not tr_ok:
                return None, True, False
            return psmi, False, tr_ok
        return None, False, False

    def run(
        self,
        reactants_a: Sequence[str],
        reactants_b: Sequence[str],
        oracle_fn: Callable[[Sequence[str]], Sequence[float]],
        *,
        total_budget: int = 400,
        batch_size: int = 16,
        top_k: int = 25,
    ) -> CombinatorialTSResult:
        """Search ``reactants_a x reactants_b`` in at most ``total_budget`` oracle calls."""
        tmpl_a = (
            self._rxn_obj.GetReactantTemplate(0)
            if self._rxn_obj.GetNumReactantTemplates() >= 2
            else None
        )
        tmpl_b = (
            self._rxn_obj.GetReactantTemplate(1)
            if self._rxn_obj.GetNumReactantTemplates() >= 2
            else None
        )
        reag_a = _standardize_reagents(
            reactants_a, role="A", uncharge=self.uncharge_reactants, template_mol=tmpl_a
        )
        reag_b = _standardize_reagents(
            reactants_b, role="B", uncharge=self.uncharge_reactants, template_mol=tmpl_b
        )
        n_a, n_b = len(reag_a), len(reag_b)
        if n_a == 0 or n_b == 0:
            raise ValueError(
                f"No reaction-compatible reagents found for {self.reaction_name!r}: "
                f"n_a={n_a}, n_b={n_b}."
            )

        n_space = n_a * n_b
        budget = min(int(total_budget), n_space)
        bs = max(1, int(batch_size))
        rng = np.random.default_rng(self.seed)

        post_a = [ReagentPosterior(rid, smi, "A", 0.0, 1.0) for rid, smi, _ in reag_a]
        post_b = [ReagentPosterior(rid, smi, "B", 0.0, 1.0) for rid, smi, _ in reag_b]

        attempted: set[tuple[int, int]] = set()
        seen_products: set[str] = set()
        records: list[CombinatorialTSProduct] = []
        obs_triples: list[tuple[int, int, float]] = []
        n_rxn_fail = 0
        n_triage_filt = 0

        # ------------------------------------------------------------------
        # Phase 1: Balanced O(N_A + N_B) Warm-up (capped at warmup_cap_fraction)
        # ------------------------------------------------------------------
        target_warmup = min(
            budget,
            max(min(bs, budget), int(round(self.warmup_cap_fraction * budget))),
            self.warmup_per_reagent * max(n_a, n_b),
        )
        warmup_queue: list[tuple[int, int]] = []
        max_dim = max(n_a, n_b)
        for _pass in range(self.warmup_per_reagent + 2):
            perm_a = rng.permutation(n_a).tolist()
            perm_b = rng.permutation(n_b).tolist()
            for slot in range(max_dim):
                ai = perm_a[slot % n_a]
                bi = perm_b[slot % n_b]
                warmup_queue.append((ai, bi))
            if len(warmup_queue) >= target_warmup * 3:
                break

        def _flush_batch(
            pending_batch: list[tuple[int, int, str, bool]],
            phase: Literal["warmup", "sampling"],
            mu0: float | None,
            var0: float | None,
            var_obs: float | None,
            std_floor: float,
        ) -> None:
            if not pending_batch:
                return
            smis = [item[2] for item in pending_batch]
            vals = [float(v) for v in oracle_fn(smis)]
            if len(vals) != len(pending_batch):
                raise ValueError(
                    f"oracle_fn returned {len(vals)} scores for {len(pending_batch)} products."
                )
            for (ai, bi, psmi, tr_ok), score in zip(pending_batch, vals, strict=True):
                if not math.isfinite(score):
                    continue
                step_num = len(records) + 1
                records.append(
                    CombinatorialTSProduct(
                        product_smiles=psmi,
                        reaction_name=self.reaction_name,
                        reactant_a_id=reag_a[ai][0],
                        reactant_a_smiles=reag_a[ai][1],
                        reactant_b_id=reag_b[bi][0],
                        reactant_b_smiles=reag_b[bi][1],
                        score=score,
                        step_index=step_num,
                        phase=phase,
                        triage_passed=tr_ok,
                    )
                )
                obs_triples.append((ai, bi, score))
                pa = post_a[ai]
                pb = post_b[bi]
                pa.n_obs += 1
                pa.sum_y += score
                pa.sum_y2 += score * score
                pb.n_obs += 1
                pb.sum_y += score
                pb.sum_y2 += score * score
                if mu0 is not None and var0 is not None and var_obs is not None:
                    for p in (pa, pb):
                        tau = (1.0 / var0) + (float(p.n_obs) / var_obs)
                        post_var = 1.0 / tau
                        p.mean = post_var * ((mu0 / var0) + (p.sum_y / var_obs))
                        p.std = max(std_floor, math.sqrt(post_var))

        batch_buf: list[tuple[int, int, str, bool]] = []
        for ai, bi in warmup_queue:
            if len(records) + len(batch_buf) >= target_warmup:
                break
            if (ai, bi) in attempted:
                continue
            attempted.add((ai, bi))
            psmi, triaged, tr_ok = self._react_pair(reag_a[ai][2], reag_b[bi][2])
            if psmi is None:
                if triaged:
                    n_triage_filt += 1
                else:
                    n_rxn_fail += 1
                continue
            if psmi in seen_products:
                continue
            seen_products.add(psmi)
            batch_buf.append((ai, bi, psmi, tr_ok))
            if len(batch_buf) >= bs:
                _flush_batch(batch_buf, "warmup", None, None, None, 0.1)
                batch_buf = []
        _flush_batch(batch_buf, "warmup", None, None, None, 0.1)
        n_warmup = len(records)

        # Initialize empirical global prior and conjugate posteriors from warm-up scores
        if records:
            warm_ys = np.array([r.score for r in records], dtype=np.float64)
            mu0 = float(np.mean(warm_ys))
            var0 = max(float(np.var(warm_ys)), 1e-4)
        else:
            mu0, var0 = 0.0, 1.0
        var_obs = var0
        std_floor = max(1e-3, self.min_std_floor_ratio * math.sqrt(var0))

        for p in (*post_a, *post_b):
            tau = (1.0 / var0) + (float(p.n_obs) / var_obs)
            post_var = 1.0 / tau
            p.mean = post_var * ((mu0 / var0) + (p.sum_y / var_obs))
            p.std = max(std_floor, math.sqrt(post_var))

        # ------------------------------------------------------------------
        # Phase 2: O(N_A + N_B) Factorized Thompson Sampling
        # ------------------------------------------------------------------
        max_stalls = max(200, (budget - len(records)) * 8)
        stalls = 0
        top_frontier = min(12, max(n_a, n_b))

        while len(records) < budget and stalls < max_stalls and len(attempted) < n_space:
            need = min(bs, budget - len(records))
            samp_buf: list[tuple[int, int, str, bool]] = []
            mu_a = np.array([p.mean for p in post_a], dtype=np.float64)
            sd_a = np.array([p.std for p in post_a], dtype=np.float64)
            mu_b = np.array([p.mean for p in post_b], dtype=np.float64)
            sd_b = np.array([p.std for p in post_b], dtype=np.float64)

            inner_tries = 0
            max_inner = max(need * 25, 100)
            while len(samp_buf) < need and inner_tries < max_inner and len(attempted) < n_space:
                inner_tries += 1
                theta_a = mu_a + (self.temperature * sd_a) * rng.standard_normal(n_a)
                theta_b = mu_b + (self.temperature * sd_b) * rng.standard_normal(n_b)

                chosen_pair: tuple[int, int] | None = None
                if self.selection_mode == "boltzmann_roulette":
                    # Scale-invariant z-scored Boltzmann roulette (works across [0,1] & pChEMBL)
                    sa_scale = max(float(np.std(theta_a)), 1e-6)
                    sb_scale = max(float(np.std(theta_b)), 1e-6)
                    la = ((theta_a - float(np.max(theta_a))) / sa_scale) / self.roulette_temperature
                    lb = ((theta_b - float(np.max(theta_b))) / sb_scale) / self.roulette_temperature
                    pa_prob = np.exp(np.clip(la, -40.0, 0.0))
                    pa_prob /= float(np.sum(pa_prob))
                    pb_prob = np.exp(np.clip(lb, -40.0, 0.0))
                    pb_prob /= float(np.sum(pb_prob))
                    ai = int(rng.choice(n_a, p=pa_prob))
                    bi = int(rng.choice(n_b, p=pb_prob))
                    if (ai, bi) not in attempted:
                        chosen_pair = (ai, bi)
                else:
                    # O(N_A + N_B): inspect only top-k_frontier x top-k_frontier sums
                    ka = min(n_a, top_frontier)
                    kb = min(n_b, top_frontier)
                    top_a = np.argpartition(-theta_a, ka - 1)[:ka]
                    top_a = top_a[np.argsort(-theta_a[top_a])]
                    top_b = np.argpartition(-theta_b, kb - 1)[:kb]
                    top_b = top_b[np.argsort(-theta_b[top_b])]
                    best_sum = float("-inf")
                    for ai_np in top_a:
                        ai_i = int(ai_np)
                        for bi_np in top_b:
                            bi_i = int(bi_np)
                            if (ai_i, bi_i) in attempted:
                                continue
                            s_val = float(theta_a[ai_i] + theta_b[bi_i])
                            if s_val > best_sum:
                                best_sum = s_val
                                chosen_pair = (ai_i, bi_i)
                                # top_b is descending, so first unattempted beats rest for ai_i
                                break

                if chosen_pair is None:
                    # Fallback random unattempted probe prevents stalls in exhausted top corner
                    ai_r = int(rng.integers(0, n_a))
                    bi_r = int(rng.integers(0, n_b))
                    if (ai_r, bi_r) in attempted:
                        stalls += 1
                        continue
                    chosen_pair = (ai_r, bi_r)

                attempted.add(chosen_pair)
                ai_c, bi_c = chosen_pair
                psmi, triaged, tr_ok = self._react_pair(reag_a[ai_c][2], reag_b[bi_c][2])
                if psmi is None:
                    if triaged:
                        n_triage_filt += 1
                    else:
                        n_rxn_fail += 1
                    stalls += 1
                    continue
                if psmi in seen_products:
                    stalls += 1
                    continue
                seen_products.add(psmi)
                samp_buf.append((ai_c, bi_c, psmi, tr_ok))

            if not samp_buf:
                break
            _flush_batch(samp_buf, "sampling", mu0, var0, var_obs, std_floor)

        ranked = sorted(records, key=lambda r: r.score, reverse=True)
        best_prod = ranked[0] if ranked else None
        best_score = best_prod.score if best_prod is not None else float("nan")
        add_r2 = _compute_additive_r2(obs_triples, n_a, n_b)
        epistasis_warn = bool(math.isfinite(add_r2) and add_r2 < 0.50)

        digest = hashlib.sha256(
            json.dumps(
                {
                    "reaction": self.reaction_name,
                    "n_a": n_a,
                    "n_b": n_b,
                    "n_calls": len(records),
                    "products": [(r.product_smiles, round(r.score, 6)) for r in ranked[:top_k]],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        return CombinatorialTSResult(
            reaction_name=self.reaction_name,
            selection_mode=self.selection_mode,
            n_reactants_a=n_a,
            n_reactants_b=n_b,
            n_virtual_space=n_space,
            n_oracle_calls=len(records),
            n_warmup_calls=n_warmup,
            n_sampling_calls=len(records) - n_warmup,
            n_reaction_failures=n_rxn_fail,
            n_triage_filtered=n_triage_filt,
            fraction_explored=float(len(records) / max(n_space, 1)),
            best_score=best_score,
            best_product=best_prod,
            top_products=tuple(ranked[:top_k]),
            all_products=tuple(records),
            reagent_additivity_r2=add_r2,
            epistasis_warning=epistasis_warn,
            reagent_posteriors_a=tuple(post_a),
            reagent_posteriors_b=tuple(post_b),
            ledger_sha256=digest,
            metadata={
                "prior_mean": round(mu0, 4),
                "prior_std": round(math.sqrt(var0), 4),
                "temperature": self.temperature,
                "warmup_per_reagent": self.warmup_per_reagent,
                "triage_filter": self.triage_filter,
            },
        )


__all__ = [
    "CombinatorialTSProduct",
    "CombinatorialTSResult",
    "CombinatorialThompsonSampler",
    "ReagentPosterior",
    "TSSelectionMode",
]
