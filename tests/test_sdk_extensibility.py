"""Contract, plugin extensibility, bug-fix & metric non-regression tests for GenEuclidean SDK.

Validates:
1. Zero-PyTorch PEP 562 lazy import across all new registry/protocol/SDK entry points.
2. Thread-safe generic ``Registry[T]``, scoped ``temporary()``, and entry-point discovery.
3. Custom 3D encoders, autoregressive decoders, sampling strategies, targets, and backbones.
4. Structured dual-inheritance exception hierarchy (100% stdlib backward compatibility).
5. Bug-fix regression guards: cofactor retention in ``Pocket.featurize``, 28-target atlas
   discovery, deep immutability of ``MoleculeReport.custom_scores``, and Tourette circuit SDK.
6. Immutable fluent ``DesignPipeline``, custom scorers/filters, callbacks, ``decorate()``,
   and multi-format exports (``to_sdf``, ``to_csv``, ``to_json``, ``to_dataframe``).
7. Exact numerical metric non-regression against verified ``v4.0.0`` clinical baselines and
   SE(3) rotational/translational invariance through registry-dispatched ``PaiNN-SE3``.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Chem import AllChem

import geneuclidean as ge
from geneuclidean import api as ge_api
from geneuclidean.cns.targets import CNSTargetSpec, list_cns_targets
from geneuclidean.config import (
    CVAEConfig,
    DecoderConfig,
    EncoderConfig,
    ExperimentConfig,
    MultiTaskConfig,
    SamplingConfig,
    seed_everything,
)
from geneuclidean.models.encoders import PocketEncoderOutput, build_encoder
from geneuclidean.models.geneuclidean import GenEuclideanModel
from geneuclidean.protocols import (
    DesignCallbackProtocol,
    MoleculeFilterProtocol,
    MoleculeScorerProtocol,
    PocketEncoderProtocol,
)
from geneuclidean.sampling.decoder_strategies import compute_strategy_probs, select_next_tokens

ECOPIPAM = "CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21"
DRD1_1 = "CN1CCN(C(=O)c2ccc3ccccc3n2)CC1"
DONEPEZIL = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"
DCU = "O=C(NC1CCCCC1)NC1CCCCC1"


def _write_maob_pocket_with_fad(tmp_path: Path) -> Path:
    """Write a minimal MAOB-like pocket PDB containing both protein residues and FAD cofactor."""
    pdb = tmp_path / "2V5Z_MAOB_mini_pocket.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   TYR A 398       1.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  TYR A 398       2.200   0.800   0.000  1.00 20.00           C",
                "ATOM      3  CB  TYR A 398       2.000   1.900   1.000  1.00 20.00           C",
                "ATOM      4  OH  TYR A 398       0.800   2.800   1.200  1.00 20.00           O",
                "ATOM      5  N   TYR A 435      -2.000   0.000   0.000  1.00 20.00           N",
                "ATOM      6  CA  TYR A 435      -2.500   1.300   0.000  1.00 20.00           C",
                "ATOM      7  OH  TYR A 435      -1.200   2.500   0.500  1.00 20.00           O",
                "HETATM    8  N5  FAD A 600       0.000   0.000   2.100  1.00 20.00           N",
                "HETATM    9  C4X FAD A 600       0.500   1.100   2.500  1.00 20.00           C",
                "HETATM   10  O4  FAD A 600      -0.400   1.800   2.900  1.00 20.00           O",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return pdb


def _write_mini_drd1_pocket(tmp_path: Path) -> tuple[Path, Path]:
    """Write a minimal DRD1 pocket PDB + 3D SDF for offline pipeline & export tests."""
    pdb = tmp_path / "9LLG_DRD1_pipe_pocket.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ASP R 103       2.800   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  ASP R 103       3.500   1.200   0.000  1.00 20.00           C",
                "ATOM      3  OD1 ASP R 103       0.900   0.900   1.200  1.00 20.00           O",
                "ATOM      4  OD2 ASP R 103       1.700   1.800   3.000  1.00 20.00           O",
                "ATOM      5  CA  PHE R 288      -3.600   1.400   0.000  1.00 20.00           C",
                "ATOM      6  CZ  PHE R 288       0.400   2.100   1.300  1.00 20.00           C",
                "ATOM      7  OG  SER R 198       0.100  -1.000   0.900  1.00 20.00           O",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    mol = Chem.AddHs(Chem.MolFromSmiles("CNCCc1ccccc1"))
    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    params.randomSeed = 42
    assert AllChem.EmbedMolecule(mol, params) == 0  # type: ignore[attr-defined]
    mol = Chem.RemoveHs(mol)
    sdf = tmp_path / "9LLG_DRD1_pipe_lig.sdf"
    writer = Chem.SDWriter(str(sdf))
    writer.write(mol)
    writer.close()
    return pdb, sdf


# ---------------------------------------------------------------------------
# 1. Zero-PyTorch Lazy Surface & Public __all__ Integrity
# ---------------------------------------------------------------------------


def test_lazy_import_extensibility_surface_stays_zero_torch() -> None:
    """Importing geneuclidean + querying all registries loads ZERO heavy ML libraries."""
    probe = """
import json, sys, time
t0 = time.perf_counter()
import geneuclidean as ge
elapsed_ms = (time.perf_counter() - t0) * 1000.0
info = {
    "elapsed_ms": round(elapsed_ms, 2),
    "encoders": ge.list_encoders(),
    "decoders": ge.list_decoders(),
    "strategies": ge.list_sampling_strategies(),
    "filters": ge.list_filters(),
    "backbones": len(ge.list_backbones()),
    "err_ok": issubclass(ge.TargetNotFoundError, (ge.GenEuclideanError, KeyError)),
    "loaded_heavy": [
        m for m in ("torch", "transformers", "peft", "matplotlib", "sklearn", "rdkit", "meeko")
        if m in sys.modules
    ],
}
print(json.dumps(info))
"""
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout.strip())
    assert info["elapsed_ms"] < 150.0, f"Slow cold import: {info['elapsed_ms']} ms"
    assert info["loaded_heavy"] == [], f"Eager heavy imports leaked: {info['loaded_heavy']}"
    assert set(info["encoders"]) >= {"painn", "egnn", "e3nn", "schnet"}
    assert set(info["decoders"]) >= {"transformer", "lstm"}
    assert set(info["strategies"]) >= {"min_p", "top_p", "top_k", "temperature", "greedy"}
    assert set(info["filters"]) >= {"triage", "mpo", "admet", "basic_amine", "qed"}
    assert info["backbones"] >= 9
    assert info["err_ok"] is True


def test_public_all_and_backward_compatible_api_facade() -> None:
    """Every symbol in geneuclidean.__all__ and geneuclidean.api.__all__ resolves cleanly."""
    for name in ge.__all__:
        assert hasattr(ge, name), f"Missing symbol in geneuclidean: {name}"
    for name in ge_api.__all__:
        assert hasattr(ge_api, name), f"Missing symbol in geneuclidean.api: {name}"
    assert ge.Pocket is ge_api.Pocket
    assert ge.DesignPipeline is ge_api.DesignPipeline
    assert ge.score_molecules is ge_api.score_molecules


# ---------------------------------------------------------------------------
# 2. Thread-Safe Registry[T], Temporary Context, and Entry-Point Discovery
# ---------------------------------------------------------------------------


def test_registry_lifecycle_thread_safety_and_temporary_rollback() -> None:
    """Verify Registry decorator, alias resolution, collision guard, rollback, and concurrency."""
    reg: ge.Registry[Any] = ge.Registry("demo_plugin")

    @reg.register("alpha", aliases=("a1", "first"), metadata={"tier": "core"})
    def _alpha(x: int) -> int:
        return x + 1

    assert "alpha" in reg and "A1" in reg and "first" in reg
    assert reg.get("a1")(41) == 42
    assert reg.resolve_key("FIRST") == "alpha"
    assert reg.get_metadata("a1")["tier"] == "core"
    assert "alpha" in reg.summary()

    # Duplicate name or colliding alias raises RegistryError (KeyError & GenEuclideanError)
    with pytest.raises(ge.RegistryError) as exc_info:
        reg.register("alpha", lambda x: x)
    assert isinstance(exc_info.value, (ge.GenEuclideanError, KeyError))

    with pytest.raises(ge.RegistryError):
        reg.register("beta", lambda x: x, aliases=("a1",))

    # Scoped temporary override restores exact prior entry even when an exception occurs
    with pytest.raises(RuntimeError, match="boom"), reg.temporary("alpha", lambda x: x + 100):
        assert reg.get("alpha")(1) == 101
        raise RuntimeError("boom")
    assert reg.get("alpha")(41) == 42
    assert reg.get("a1")(41) == 42

    # Scoped temporary new key removes itself on exit
    with reg.temporary("ephemeral", lambda x: x * 3, aliases=("eph",)):
        assert reg.get("eph")(7) == 21
    assert "ephemeral" not in reg and "eph" not in reg

    # Multi-threaded registration + lookup safety
    def _worker(i: int) -> int:
        name = f"worker_{i}"
        reg.register(name, lambda x, k=i: x + k)
        return int(reg.get(name)(10))

    with ThreadPoolExecutor(max_workers=8) as pool:
        outs = list(pool.map(_worker, range(24)))
    assert outs == [10 + i for i in range(24)]
    assert len(reg) == 25


def test_discover_plugins_from_mock_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify PEP 621 entry_points discovery loads plugins cleanly into global registries."""

    class _FakeEntryPoint:
        def __init__(self, name: str, value: str, obj: Any) -> None:
            self.name = name
            self.value = value
            self._obj = obj

        def load(self) -> Any:
            return self._obj

    fake_scorer = ge.FunctionScorer("ep_demo_scorer", lambda r: 0.77)

    def _fake_entry_points(*, group: str) -> list[Any]:
        if group == "geneuclidean.scorers":
            return [_FakeEntryPoint("ep_demo_scorer", "fake_pkg:scorer", fake_scorer)]
        return []

    monkeypatch.setattr(importlib_metadata, "entry_points", _fake_entry_points)
    try:
        loaded = ge.discover_plugins(kinds=("scorers",))
        assert loaded["scorers"] == ["ep_demo_scorer"]
        assert "ep_demo_scorer" in ge.SCORER_REGISTRY
        assert ge.SCORER_REGISTRY.get("ep_demo_scorer") is fake_scorer
        assert ge.SCORER_REGISTRY.get_metadata("ep_demo_scorer")["source"] == "entry_point"
    finally:
        if "ep_demo_scorer" in ge.SCORER_REGISTRY:
            ge.SCORER_REGISTRY.unregister("ep_demo_scorer")


# ---------------------------------------------------------------------------
# 3. Custom Encoder, Decoder, Sampling Strategy, Target & Backbone Plugins
# ---------------------------------------------------------------------------


class _DummyEquivariantEncoder(nn.Module):
    """Minimal custom SE(3)-invariant 3D pocket encoder satisfying PocketEncoderProtocol."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        coords: torch.Tensor,
        node_feats: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> PocketEncoderOutput:
        del coords
        node_emb = self.proj(node_feats) * atom_mask.unsqueeze(-1).to(node_feats.dtype)
        denom = atom_mask.sum(dim=1, keepdim=True).clamp(min=1).to(node_feats.dtype)
        pooled = node_emb.sum(dim=1) / denom
        return PocketEncoderOutput(node_emb=node_emb, pooled_emb=pooled, vec_emb=None)


class _DummyDecoder(nn.Module):
    """Minimal custom autoregressive decoder satisfying MolecularDecoderProtocol."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del memory, memory_mask, return_attn
        return self.head(self.emb(input_ids)), None


def test_custom_encoder_decoder_and_sampling_strategy_plugins() -> None:
    """Custom plugins pass Pydantic validation, plug into GenEuclideanModel, and reject typos."""
    # Unregistered names STILL raise ValidationError
    with pytest.raises(ValidationError):
        EncoderConfig(arch="nonexistent_encoder_xyz")
    with pytest.raises(ValidationError):
        DecoderConfig(arch="nonexistent_decoder_xyz")
    with pytest.raises(ValidationError):
        SamplingConfig(strategy="nonexistent_strategy_xyz")

    with (
        ge.ENCODER_REGISTRY.temporary(
            "custom_se3_demo",
            lambda cfg, max_neighbors=24: _DummyEquivariantEncoder(cfg.node_in_dim, cfg.out_dim),
        ),
        ge.DECODER_REGISTRY.temporary(
            "custom_dec_demo",
            lambda cfg, cross_attn_gate_init=0.1: _DummyDecoder(cfg.vocab_size, cfg.d_model),
        ),
        ge.SAMPLING_STRATEGY_REGISTRY.temporary(
            "uniform_top2",
            lambda logits, *, temperature=1.0, top_k=30, top_p=0.92, min_p=0.08: torch.softmax(
                logits / max(temperature, 1e-6), dim=-1
            ),
        ),
    ):
        assert "custom_se3_demo" in ge.list_encoders()
        assert "custom_dec_demo" in ge.list_decoders()
        assert "uniform_top2" in ge.list_sampling_strategies()

        cfg = ExperimentConfig(
            encoder=EncoderConfig(arch="custom_se3_demo", out_dim=32),
            cvae=CVAEConfig(enabled=False),
            decoder=DecoderConfig(arch="custom_dec_demo", vocab_size=16, d_model=32),
            multitask=MultiTaskConfig(use_pkd_head=False, use_clip_head=False, use_prop_head=False),
            sampling=SamplingConfig(strategy="uniform_top2"),
        )
        model = GenEuclideanModel(cfg)
        assert isinstance(model.encoder, PocketEncoderProtocol)

        batch = {
            "coords": torch.randn(2, 6, 3),
            "node_feats": torch.randn(2, 6, 52),
            "atom_mask": torch.ones(2, 6, dtype=torch.bool),
            "input_ids": torch.randint(0, 16, (2, 5)),
        }
        out = model(batch)
        assert out.logits.shape == (2, 5, 16)

        logits = torch.randn(3, 16)
        probs = compute_strategy_probs(logits, "uniform_top2", temperature=0.8)
        assert probs.shape == (3, 16)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)
        toks = select_next_tokens(logits, "uniform_top2", temperature=0.8)
        assert toks.shape == (3,)


def test_custom_target_registration_and_clean_unregister() -> None:
    """Custom protein targets resolve via SDK and restore exact atlas counts upon unregister."""
    base_v3_count = len(list_cns_targets())
    base_sdk_count = len(ge.list_targets())
    assert base_v3_count == 12
    assert base_sdk_count == 28

    custom = CNSTargetSpec(
        target_id="CUSTOM_CNS1",
        gene_symbol="CTST1",
        name="Synthetic Test CNS Receptor 1",
        target_class="Class C GPCR",
        uniprot_id="Q9TEST1",
        chembl_id="CHEMBL9999999",
        indications=("Custom Neurology Assay",),
        therapeutic_moa="Selective negative allosteric modulator",
        desired_state="Inactive allosteric",
        primary_pdb_id="9ZZZ",
        secondary_pdb_ids=(),
        primary_anchor_residue="Asp100",
        hotspot_residues=("Asp100", "Phe200"),
        hotspot_isoform_aliases=(),
        reference_drugs=(),
        disease_area="tourette",
        moa_class="NAM",
        keep_cofactors=("FAD",),
    )
    ge.register_target(custom, aliases=("TEST_ALIAS_1",))
    try:
        assert ge.get_target("CUSTOM_CNS1").primary_pdb_id == "9ZZZ"
        assert ge.get_target("TEST_ALIAS_1").target_id == "CUSTOM_CNS1"
        assert ge.get_target("CTST1").target_id == "CUSTOM_CNS1"
        assert ge.get_target("CHEMBL9999999").target_id == "CUSTOM_CNS1"
        assert len(ge.list_targets()) == base_sdk_count + 1
        p = ge.Pocket.from_target("TEST_ALIAS_1")
        assert p.effective_cofactors == ("FAD",)
    finally:
        removed = ge.unregister_target("CUSTOM_CNS1")
        assert removed.target_id == "CUSTOM_CNS1"

    assert len(list_cns_targets()) == 12
    assert len(ge.list_targets()) == 28


# ---------------------------------------------------------------------------
# 4. Bug-Fix Regression Guards & Structured Dual-Inheritance Exceptions
# ---------------------------------------------------------------------------


def test_pocket_featurize_preserves_target_cofactors_and_discovers_28_targets(
    tmp_path: Path,
) -> None:
    """Regression guard: Pocket.featurize forwards keep_cofactors (e.g., FAD in MAOB)."""
    maob_pdb = _write_maob_pocket_with_fad(tmp_path)
    p_with_target = ge.Pocket.from_pdb(maob_pdb, target_id="MAOB")
    p_without_cofactor = ge.Pocket.from_pdb(maob_pdb)

    assert p_with_target.effective_cofactors == ("FAD",)
    feat_with = p_with_target.featurize()
    feat_without = p_without_cofactor.featurize()
    # 7 protein heavy atoms + 3 FAD heavy atoms = 10 vs 7 without FAD retention
    assert len(feat_with.coords) == 10
    assert len(feat_without.coords) == 7

    # 28-target discovery including Alzheimer's & Tourette CSTC/GABA
    ad_targets = ge.list_targets("alzheimer")
    assert len(ad_targets) >= 6
    assert any(t.target_id == "ACHE" for t in ad_targets)
    assert any(t.target_id == "MAOB" for t in ad_targets)

    # Tourette CSTC circuit audit & mechanism-polarity scoring via top-level SDK
    audit = ge.audit_tourette_circuit("GRM5")
    assert audit["target_id"] == "GRM5"
    all_audit = ge.audit_tourette_circuit()
    assert len(all_audit) >= 12
    dcu_circuit = ge.score_tourette_circuit(DCU, "GRM5", vina_dg_kcal=-8.0)
    assert dcu_circuit.hygiene_gate == pytest.approx(0.0)
    assert dcu_circuit.circuit_readiness_score == pytest.approx(0.0)


def test_structured_exceptions_are_backward_compatible(tmp_path: Path) -> None:
    """Custom GenEuclideanError subclasses also match expected stdlib exception types."""
    with pytest.raises(ge.TargetNotFoundError) as e_target:
        ge.get_target("NO_SUCH_TARGET_999")
    assert isinstance(e_target.value, (ge.GenEuclideanError, KeyError))
    assert e_target.value.hint is not None

    with pytest.raises(ge.InvalidMoleculeError) as e_mol:
        ge.protonate_ph74("INVALID_SMILES!!!")
    assert isinstance(e_mol.value, (ge.GenEuclideanError, ValueError))

    with pytest.raises(ge.StructureNotFoundError) as e_struct:
        ge.Pocket.from_pdb(tmp_path / "does_not_exist.pdb")
    assert isinstance(e_struct.value, (ge.GenEuclideanError, FileNotFoundError))

    bad_ckpt = tmp_path / "bad_ckpt.pt"
    torch.save({"unexpected": 123}, bad_ckpt)
    with pytest.raises(ge.CheckpointError) as e_ckpt:
        ge.MoleculeDesigner.from_pretrained(bad_ckpt, device="cpu")
    assert isinstance(e_ckpt.value, (ge.GenEuclideanError, ValueError))


# ---------------------------------------------------------------------------
# 5. DesignPipeline, Custom Scorers/Filters, Callbacks & SDF/CSV/JSON Export
# ---------------------------------------------------------------------------


def test_filters_custom_scorers_and_immutability() -> None:
    """Verify composable filters, anti-reward-hacking custom scorer gating, and immutability."""
    bonus = ge.FunctionScorer("low_mw_bonus", lambda r: 1.0 if r.mw <= 350.0 else 0.5)
    assert isinstance(bonus, MoleculeScorerProtocol)

    reports = ge.score_molecules(
        [ECOPIPAM, DRD1_1, DONEPEZIL, DCU, "BAD_SMI"],
        target="DRD1",
        include_triage=True,
        include_admet=False,
        extra_scorers=[(bonus, 0.25)],
    )
    eco, lead, donep, dcu, bad = reports

    # Deep immutability of MoleculeReport.custom_scores
    with pytest.raises(TypeError):
        eco.custom_scores["mutated"] = 99.0  # type: ignore[index]

    assert lead.custom_scores["low_mw_bonus"] == pytest.approx(1.0)
    assert donep.custom_scores["low_mw_bonus"] == pytest.approx(0.5)
    # Critical pharma safety invariant: DCU (hygiene_gate=0) stays strictly 0.0 even with bonus!
    assert dcu.hygiene_gate == pytest.approx(0.0)
    assert dcu.composite_reward == pytest.approx(0.0)
    assert bad.valid is False and bad.composite_reward == pytest.approx(0.0)

    # Built-in composable filters satisfy MoleculeFilterProtocol and compose cleanly
    triage_f = ge.TriageFilter(max_lilly_demerits=100)
    mpo_f = ge.MPOFilter(min_mpo_v2=4.5)
    amine_f = ge.BasicAmineFilter(allowed_charges=(1,))
    qed_f = ge.QEDFilter(min_qed=0.6, max_sa_score=4.0)
    for flt in (triage_f, mpo_f, amine_f, qed_f):
        assert isinstance(flt, MoleculeFilterProtocol)

    passed = [r for r in reports if triage_f.keep(r) and mpo_f.keep(r) and amine_f.keep(r)]
    passed_smiles = {r.canonical_smiles for r in passed}
    assert DRD1_1 in passed_smiles
    assert DONEPEZIL in passed_smiles
    assert DCU not in passed_smiles


def test_design_pipeline_callbacks_batch_and_exports(tmp_path: Path) -> None:
    """End-to-end immutable DesignPipeline with callbacks, decorate(), and SDF/CSV/JSON export."""
    pytest.importorskip("transformers")
    pytest.importorskip("peft")

    pocket_pdb, ligand_sdf = _write_mini_drd1_pocket(tmp_path)
    pocket = ge.Pocket.from_pdb(pocket_pdb, ligand_sdf=ligand_sdf, target_id="DRD1")
    designer = ge.MoleculeDesigner.from_mock(device="cpu", seed=42)

    events: list[str] = []

    class _RecorderCallback(ge.BaseDesignCallback):
        def on_design_start(self, pocket: Any, config: Mapping[str, Any]) -> None:
            events.append(f"start:{config.get('mode')}")

        def on_candidates_generated(
            self,
            pocket: Any,
            raw_smiles: Sequence[str],
            reports: Sequence[ge.MoleculeReport],
        ) -> None:
            events.append(f"gen:{len(raw_smiles)}:{len(reports)}")

        def on_design_end(self, pocket: Any, result: ge.DesignResult) -> None:
            events.append(f"end:{result.pocket_id}")

    cb = _RecorderCallback()
    assert isinstance(cb, DesignCallbackProtocol)

    base_pipe = ge.DesignPipeline(designer, name="base_pipe").with_admet(enabled=False)
    derived_pipe = (
        base_pipe.add_filter(ge.TriageFilter())
        .add_scorer(ge.FunctionScorer("unit_bonus", lambda r: 0.8), weight=0.1)
        .add_callback(cb)
        .with_top_k(5)
    )
    # Immutability check: base_pipe unaffected by branching derived_pipe
    assert len(base_pipe.filters) == 0 and len(derived_pipe.filters) == 1
    desc = derived_pipe.describe()
    assert desc["name"] == "base_pipe"
    assert desc["filters"] == ["triage_filter"]
    assert desc["scorers"] == [{"name": "unit_bonus", "weight": 0.1}]

    res = derived_pipe.run(pocket, n=4, max_new_tokens=14, cfg_scale=1.2, seed=42)
    assert events == ["start:de_novo", "gen:4:4", "end:DRD1"]
    assert res.metadata["pipeline_name"] == "base_pipe"
    assert len(res) == len(res.reports)
    assert res.summary()["pocket_id"] == "DRD1"

    # Batch run across multiple pockets
    batch_res = base_pipe.run_batch([pocket, pocket], n=2, max_new_tokens=10, seed=42)
    assert len(batch_res) == 2

    # Export verification using known valid clinical reports
    known_reports = ge.score_molecules(
        [ECOPIPAM, DRD1_1],
        target="DRD1",
        include_triage=True,
        include_admet=False,
        extra_scorers=[ge.FunctionScorer("demo", lambda r: 0.9)],
    )
    export_res = ge.DesignResult(
        pocket_id="DRD1",
        backend="mock:test",
        raw_smiles=[ECOPIPAM, DRD1_1],
        reports=known_reports,
        valid_rate=1.0,
        unique_rate=1.0,
    )
    assert len(export_res) == 2
    assert export_res[0].target_id == "DRD1"
    assert len(export_res.top(1)) == 1
    filtered_res = export_res.filter(ge.MPOFilter(min_mpo_v2=5.0))
    assert len(filtered_res) == 1 and filtered_res[0].canonical_smiles == DRD1_1

    json_path = export_res.to_json(tmp_path / "hits.json")
    csv_path = export_res.to_csv(tmp_path / "hits.csv")
    sdf_path = export_res.to_sdf(tmp_path / "hits.sdf", embed_3d=True, protonate_ph74=True)
    assert json_path.is_file() and csv_path.is_file() and sdf_path.is_file()

    with csv_path.open("r", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 2
    assert "custom_demo" in csv_rows[0]

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mols = [m for m in suppl if m is not None]
    assert len(mols) == 2
    assert mols[0].GetNumConformers() == 1
    assert mols[0].GetProp("target_id") == "DRD1"
    assert float(mols[0].GetProp("custom_demo")) == pytest.approx(0.9, abs=1e-3)

    df = export_res.to_dataframe()
    assert len(df) == 2


# ---------------------------------------------------------------------------
# 6. Strict Metric Non-Regression Guard (Zero Degradation vs v4.0.0 Baselines)
# ---------------------------------------------------------------------------


def test_v400_clinical_scoring_metric_non_regression() -> None:
    """Guard exact Pfizer MPO, QED, SA, hygiene & reward values against v4.0.0 baselines."""
    reports = ge.score_molecules(
        [ECOPIPAM, DRD1_1, DONEPEZIL, DCU],
        target="DRD1",
        include_triage=True,
        include_admet=False,
    )
    expected = [
        (3.858698, 4.250067, 0.794966, 3.171443, 1.0, 0.762218, 1, True),
        (5.672000, 5.672000, 0.776711, 1.820116, 1.0, 0.911477, 1, True),
        (4.270060, 5.018664, 0.747461, 2.677222, 1.0, 0.830729, 1, True),
        (5.024500, 5.000000, 0.743623, 1.829867, 0.0, 0.000000, 0, False),
    ]
    for rep, exp in zip(reports, expected, strict=True):
        mpo1, mpo2, qed, sa, hyg, rew, q74, tri = exp
        assert rep.cns_mpo_v1 == pytest.approx(mpo1, abs=1e-5)
        assert rep.cns_mpo_v2 == pytest.approx(mpo2, abs=1e-5)
        assert rep.qed == pytest.approx(qed, abs=1e-5)
        assert rep.sa_score == pytest.approx(sa, abs=1e-5)
        assert rep.hygiene_gate == pytest.approx(hyg, abs=1e-6)
        assert rep.composite_reward == pytest.approx(rew, abs=1e-5)
        assert rep.net_charge_ph74 == q74
        assert rep.triage_passed is tri


def test_painn_se3_registry_dispatch_preserves_exact_se3_invariance() -> None:
    """Verify registry-dispatched PaiNN-SE3 preserves strict SE(3) rotation+translation parity."""
    seed_everything(42)
    enc = build_encoder(
        EncoderConfig(arch="painn", hidden_dim=32, vector_dim=16, out_dim=32, num_layers=2)
    ).eval()

    coords = torch.randn(1, 18, 3) * 2.5
    feats = torch.randn(1, 18, 52)
    mask = torch.ones(1, 18, dtype=torch.bool)

    # Proper SO(3) rotation (det = +1) + random 3D translation
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    shift = torch.tensor([[[4.2, -3.1, 7.5]]], dtype=torch.float32)
    coords_rot = coords @ q.T + shift

    with torch.no_grad():
        out_orig = enc(coords, feats, mask)
        out_rot = enc(coords_rot, feats, mask)

    max_node_diff = float((out_orig.node_emb - out_rot.node_emb).abs().max().item())
    max_pool_diff = float((out_orig.pooled_emb - out_rot.pooled_emb).abs().max().item())
    assert max_node_diff < 1e-4, f"SE(3) node invariance degraded: {max_node_diff}"
    assert max_pool_diff < 1e-4, f"SE(3) pooled invariance degraded: {max_pool_diff}"


def test_pocket_visualize_dock_error_decorate_and_register_backbone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Pocket.visualize, Pocket.dock errors, decorate(), and backbone registration."""
    pocket_pdb, ligand_sdf = _write_mini_drd1_pocket(tmp_path)
    pocket = ge.Pocket.from_pdb(pocket_pdb, ligand_sdf=ligand_sdf, target_id="DRD1")

    # Pocket.visualize pre-populates pdb_id, native ghost & disease automatically
    viz_out = pocket.visualize(ligand_sdf, out_html=tmp_path / "pocket_viz.html")
    assert viz_out["html_path"] == tmp_path / "pocket_viz.html"
    assert (tmp_path / "pocket_viz.html").is_file()

    # Pocket.dock surfaces OptionalDependencyError (also a RuntimeError & GenEuclideanError)
    monkeypatch.setattr("geneuclidean.evaluation.docking_pipeline._require_meeko", lambda: None)
    monkeypatch.setattr(
        "geneuclidean.evaluation.docking_pipeline._resolve_vina_binary", lambda _: None
    )
    with pytest.raises(ge.OptionalDependencyError, match="AutoDock Vina") as exc_info:
        pocket.dock(ECOPIPAM, vina_bin=tmp_path / "no_vina", require_vina=True)
    assert isinstance(exc_info.value, (ge.GenEuclideanError, RuntimeError))
    assert exc_info.value.hint is not None

    # MoleculeDesigner.decorate() works end-to-end on offline mock LLM
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    designer = ge.MoleculeDesigner.from_mock(device="cpu", seed=42)
    dec_res = designer.decorate(
        pocket,
        "[*]c1ccccc1",
        n=2,
        max_new_tokens=10,
        cfg_scale=1.2,
        include_admet=False,
        seed=42,
    )
    assert dec_res.metadata["mode"] == "scaffold_decoration"
    assert len(dec_res.raw_smiles) == 2

    # Custom backbone registration syncs both BACKBONE_REGISTRY and zero-PyTorch list_backbones()
    from geneuclidean.models.hf_backbones import BACKBONE_REGISTRY, BackboneSpec, register_backbone
    from geneuclidean.registry import BACKBONE_CATALOG

    custom_id = "org/custom-chem-llm-test"
    spec = BackboneSpec(
        backbone_id=custom_id,
        family="custom_chem",
        tokenizer_id=custom_id,
        lora_targets=("q_proj", "v_proj"),
        fan_in_fan_out=False,
        is_smiles_native=True,
        prompt_prefix="",
        representation="smiles",
    )
    try:
        register_backbone(spec, aliases=("custom-chem-alias",), params="42M", license_name="MIT")
        assert custom_id in BACKBONE_REGISTRY
        assert "custom-chem-alias" in BACKBONE_REGISTRY
        catalog = ge.list_backbones()
        assert custom_id in catalog and catalog[custom_id]["params"] == "42M"
    finally:
        BACKBONE_REGISTRY.pop(custom_id, None)
        BACKBONE_REGISTRY.pop("custom-chem-alias", None)
        BACKBONE_CATALOG.pop(custom_id, None)


def test_completed_candidate_triage_and_serialized_v4_gate() -> None:
    from dataclasses import replace

    from geneuclidean.cns.admet import PharmaAdmetPrediction

    smis = ["*CC", "CC.CC", "[Na+]", "[CH3]"]
    expected = ["unresolved_attachment", "multiple_components", "no_organic_parent", "radical"]
    for report, reason in zip(ge.score_molecules(smis, include_admet=False), expected, strict=True):
        assert report.valid and not ge.TriageFilter().keep(report)
        assert reason in report.triage_alerts
        assert report.hygiene_gate == report.composite_reward == 0
    panel = PharmaAdmetPrediction("CC", "CC", True, {"bbb": 0.9}, {}, {}, {}, True, False).to_dict()
    report = ge.MoleculeReport("CC", "CC", True, cns_mpo_v2=5, triage_passed=True, admet=panel)
    assert not report.is_cns_lead
    assert not ge.ADMETFilter().keep(report)
    no_herg = replace(report, admet={**panel, "admet_v4_pass": True})
    assert ge.ADMETFilter(max_herg_prob=0.3, pass_when_missing=True).keep(no_herg)
    assert not ge.ADMETFilter(max_herg_prob=0.3, pass_when_missing=False).keep(no_herg)
    invalid = replace(no_herg, admet={"admet_pass": True, "herg_prob": float("nan")})
    assert not ge.ADMETFilter(max_herg_prob=0.3).keep(invalid)


def test_pipeline_selects_diverse_reports_after_filtering(tmp_path, monkeypatch) -> None:
    from geneuclidean.sampling import llm_sampler
    from geneuclidean.sampling.rerank import maxmin_diversity_select

    candidates = ["CCO", "OCC", "CCC", "c1ccccc1", "Cc1ccccc1", "C1CCOCC1", "*CC"]
    monkeypatch.setattr(llm_sampler, "sample_llm_molecules", lambda **kwargs: candidates)
    designer = ge.MoleculeDesigner.from_mock(device="cpu")
    pdb, ligand = _write_mini_drd1_pocket(tmp_path)
    pocket = ge.Pocket.from_pdb(pdb, ligand_sdf=ligand)
    pipeline = ge.DesignPipeline(designer).with_admet(enabled=False).add_filter(ge.TriageFilter())
    opts = dict(n=len(candidates), seed=42)
    pool = pipeline.run(pocket, **opts)
    assert len(pool) >= 3
    scores = [r.composite_reward for r in pool]
    expected = maxmin_diversity_select(pool.smiles, scores, k=5, min_tanimoto_dist=1.0)
    selected = pipeline.with_top_k(5, min_tanimoto_dist=1.0).run(pocket, **opts)
    assert 0 < len(selected) < 5
    assert selected.smiles == [pool.smiles[i] for i in expected]
    assert selected.unique_rate == pool.unique_rate and selected.valid_rate == 1
    with pytest.raises(ValueError, match="nonnegative"):
        pipeline.with_top_k(-1).run(pocket, **opts)
