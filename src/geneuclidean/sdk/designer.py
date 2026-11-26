"""High-level 3D-conditioned de novo generation & SAFE scaffold decoration SDK."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from geneuclidean.exceptions import (
    CheckpointError,
    OptionalDependencyError,
    StructureNotFoundError,
)
from geneuclidean.protocols import (
    DesignCallbackProtocol,
    MoleculeFilterProtocol,
    MoleculeScorerProtocol,
)
from geneuclidean.sdk.pocket import Pocket
from geneuclidean.sdk.scoring import MoleculeReport, score_molecules

if TYPE_CHECKING:
    from geneuclidean.data.featurizer import PocketFeatures


@dataclass
class DesignResult:
    """Output of :meth:`MoleculeDesigner.design` or :meth:`MoleculeDesigner.decorate`.

    Supports ergonomic Python collection access (``len(res)``, ``for r in res:``,
    ``res[0]``), fluent filtering (:meth:`filter`), top-k slicing (:meth:`top`),
    JSON export (:meth:`to_json`), and optional ``pandas.DataFrame`` export
    (:meth:`to_dataframe`).
    """

    pocket_id: str
    backend: str
    raw_smiles: list[str]
    reports: list[MoleculeReport]
    valid_rate: float
    unique_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.reports)

    def __iter__(self) -> Iterator[MoleculeReport]:
        return iter(self.reports)

    def __getitem__(self, index: int) -> MoleculeReport:
        return self.reports[index]

    @property
    def smiles(self) -> list[str]:
        """Return canonical SMILES of ranked/filtered designed molecules."""
        return [r.canonical_smiles or r.smiles for r in self.reports]

    def top(self, k: int) -> list[MoleculeReport]:
        """Return the top ``k`` scored reports sorted by descending ``composite_reward``."""
        ranked = sorted(self.reports, key=lambda r: r.composite_reward, reverse=True)
        return ranked[: max(0, int(k))]

    def filter(self, predicate_or_filter: MoleculeFilterProtocol | Any) -> DesignResult:
        """Return a new :class:`DesignResult` keeping only reports matching a filter.

        Args:
            predicate_or_filter: Either a :class:`MoleculeFilterProtocol` instance
                (implementing ``.keep(report)``) or a plain callable ``fn(report) -> bool``.
        """
        if isinstance(predicate_or_filter, MoleculeFilterProtocol):
            kept = [r for r in self.reports if predicate_or_filter.keep(r)]
            filter_name = predicate_or_filter.name
        elif callable(predicate_or_filter):
            kept = [r for r in self.reports if bool(predicate_or_filter(r))]
            filter_name = getattr(predicate_or_filter, "__name__", "custom_predicate")
        else:
            raise TypeError(
                "filter() expects a MoleculeFilterProtocol or callable(MoleculeReport) -> bool, "
                f"got {type(predicate_or_filter).__name__}."
            )
        new_meta = dict(self.metadata)
        history = list(new_meta.get("applied_filters", []))
        history.append({"name": filter_name, "in": len(self.reports), "out": len(kept)})
        new_meta["applied_filters"] = history
        return DesignResult(
            pocket_id=self.pocket_id,
            backend=self.backend,
            raw_smiles=list(self.raw_smiles),
            reports=kept,
            valid_rate=self.valid_rate,
            unique_rate=self.unique_rate,
            metadata=new_meta,
        )

    def summary(self) -> dict[str, Any]:
        """Return compact summary statistics for logging and experiment tracking."""
        top_reward = max((r.composite_reward for r in self.reports), default=0.0)
        mean_mpo = (
            sum(r.cns_mpo_v2 for r in self.reports) / len(self.reports) if self.reports else 0.0
        )
        return {
            "pocket_id": self.pocket_id,
            "backend": self.backend,
            "n_raw": len(self.raw_smiles),
            "n_returned": len(self.reports),
            "valid_rate": round(self.valid_rate, 4),
            "unique_rate": round(self.unique_rate, 4),
            "mean_cns_mpo_v2": round(mean_mpo, 3),
            "top_composite_reward": round(top_reward, 4),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize design batch and scored molecule reports to a JSON-compatible dict."""
        return {
            "pocket_id": self.pocket_id,
            "backend": self.backend,
            "n_raw": len(self.raw_smiles),
            "n_returned": len(self.reports),
            "valid_rate": round(self.valid_rate, 4),
            "unique_rate": round(self.unique_rate, 4),
            "metadata": dict(self.metadata),
            "reports": [r.to_dict() for r in self.reports],
        }

    def to_json(self, out_path: str | Path, *, indent: int = 2) -> Path:
        """Persist this :class:`DesignResult` to a formatted UTF-8 JSON file."""
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=indent) + "\n", encoding="utf-8")
        return dest

    def to_csv(self, out_path: str | Path) -> Path:
        """Export scored candidate reports to a flat UTF-8 CSV file (zero pandas dependency)."""
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        fieldnames: list[str] = []
        seen_fields: set[str] = set()
        for rep in self.reports:
            flat = rep.to_dict()
            # Flatten nested custom_scores & serialize list/dict columns cleanly for spreadsheets.
            custom = flat.pop("custom_scores", None)
            if isinstance(custom, dict):
                for ck, cv in custom.items():
                    flat[f"custom_{ck}"] = cv
            if isinstance(flat.get("triage_alerts"), list):
                flat["triage_alerts"] = ";".join(str(a) for a in flat["triage_alerts"])
            if isinstance(flat.get("admet"), dict):
                flat["admet"] = json.dumps(flat["admet"], sort_keys=True)
            for key in flat:
                if key not in seen_fields:
                    seen_fields.add(key)
                    fieldnames.append(key)
            rows.append(flat)
        if not fieldnames:
            fieldnames = ["smiles", "canonical_smiles", "valid", "cns_mpo_v2", "composite_reward"]
        with dest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return dest

    def to_sdf(
        self,
        out_path: str | Path,
        *,
        embed_3d: bool = False,
        protonate_ph74: bool = False,
        seed: int = 42,
    ) -> Path:
        """Export valid designed candidates as an SDFile with full KPIs stored as SD properties.

        Compatible with PyMOL, Maestro, ChimeraX, DataWarrior, and RDKit downstream pipelines.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        writer = Chem.SDWriter(str(dest))
        try:
            for idx, rep in enumerate(self.reports):
                if not rep.valid:
                    continue
                smi_src = (
                    rep.smiles_ph74
                    if (protonate_ph74 and rep.smiles_ph74)
                    else (rep.canonical_smiles or rep.smiles)
                )
                mol = Chem.MolFromSmiles(smi_src)
                if mol is None:
                    continue
                if embed_3d:
                    mol_h = Chem.AddHs(mol)
                    params = AllChem.ETKDGv3()  # type: ignore[attr-defined]
                    params.randomSeed = int(seed) + idx
                    if AllChem.EmbedMolecule(mol_h, params) == 0:  # type: ignore[attr-defined]
                        mol = Chem.RemoveHs(mol_h)
                mol.SetProp("_Name", f"{self.pocket_id}_cand_{idx + 1}")
                mol.SetProp("target_id", str(rep.target_id or self.pocket_id))
                mol.SetProp("canonical_smiles", str(rep.canonical_smiles or rep.smiles))
                mol.SetProp("cns_mpo_v2", f"{rep.cns_mpo_v2:.3f}")
                mol.SetProp("composite_reward", f"{rep.composite_reward:.4f}")
                mol.SetProp("hygiene_gate", f"{rep.hygiene_gate:.4f}")
                mol.SetProp("triage_passed", str(rep.triage_passed))
                if rep.net_charge_ph74 is not None:
                    mol.SetProp("net_charge_ph74", str(rep.net_charge_ph74))
                if rep.murcko_scaffold:
                    mol.SetProp("murcko_scaffold", rep.murcko_scaffold)
                for ck, cv in rep.custom_scores.items():
                    mol.SetProp(f"custom_{ck}", f"{cv:.4f}")
                writer.write(mol)
        finally:
            writer.close()
        return dest

    def to_dataframe(self) -> Any:
        """Convert scored reports into a ``pandas.DataFrame`` (requires ``pandas``)."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise OptionalDependencyError(
                "DesignResult.to_dataframe() requires pandas.",
                hint="Install via: pip install 'geneuclidean[cns]'",
            ) from exc
        return pd.DataFrame([r.to_dict() for r in self.reports])


def _deduplicate_reports(reports: Sequence[MoleculeReport]) -> list[MoleculeReport]:
    """Deduplicate valid reports by canonical SMILES, keeping highest ``composite_reward``."""
    dedup_map: dict[str, MoleculeReport] = {}
    for rep in reports:
        if not rep.valid:
            continue
        key = rep.canonical_smiles or rep.smiles
        prev = dedup_map.get(key)
        if prev is None or rep.composite_reward > prev.composite_reward:
            dedup_map[key] = rep
    return list(dedup_map.values())


class MoleculeDesigner:
    """Unified 3D pocket-conditioned molecular generator wrapping v2 CVAE & v3/v4 LLMs.

    Lazily imports PyTorch and Hugging Face ``transformers``/``peft`` strictly when an
    instance is constructed via :meth:`from_pretrained` or :meth:`from_mock`.
    """

    def __init__(
        self,
        *,
        backend: Literal["v2", "llm"],
        model: Any,
        device: Any,
        cfg: Any,
        tokenizer: Any | None = None,
        backbone_id: str | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.device = device
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.backbone_id = backbone_id or (
            f"v2-{cfg.encoder.arch}-{cfg.decoder.arch}" if backend == "v2" else "llm"
        )

    def __repr__(self) -> str:
        return (
            f"MoleculeDesigner(backend={self.backend!r}, "
            f"backbone_id={self.backbone_id!r}, device={self.device!r})"
        )

    @classmethod
    def from_mock(
        cls,
        *,
        device: str = "cpu",
        seed: int = 42,
    ) -> MoleculeDesigner:
        """Instantiate an offline zero-download mock 3D-to-LLM designer for testing & demos."""
        from geneuclidean.config import ExperimentConfig, seed_everything, select_device

        try:
            from geneuclidean.models.hf_backbones import MOCK_BACKBONE_ID, load_hf_backbone
            from geneuclidean.models.llm_bridge import PocketConditionedLLM
        except ImportError as exc:
            raise OptionalDependencyError(
                "Mock LLM designer requires optional LLM dependencies.",
                hint="Install them via: pip install 'geneuclidean[llm]'",
            ) from exc

        seed_everything(seed)
        torch_device = select_device(device)
        v2_cfg = ExperimentConfig()
        backbone = load_hf_backbone(MOCK_BACKBONE_ID, use_lora=False, dtype="float32")
        model = PocketConditionedLLM(
            v2_cfg=v2_cfg,
            backbone=backbone,
            num_queries=8,
            perceiver_layers=1,
            perceiver_heads=4,
            num_global_tokens=2,
        )
        model.to(torch_device)
        model.eval()
        return cls(
            backend="llm",
            model=model,
            device=torch_device,
            cfg=v2_cfg,
            tokenizer=backbone.tokenizer,
            backbone_id=MOCK_BACKBONE_ID,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        mode: Literal["auto", "v2", "llm"] = "auto",
        v2_pocket_ckpt: str | Path | None = None,
        backbone_id: str | None = None,
        vocab_path: str | Path | None = None,
        cache_dir: str = ".hf_cache.nosync",
        device: str = "auto",
        dtype: str = "auto",
        local_files_only: bool = False,
    ) -> MoleculeDesigner:
        """Load a trained GenEuclidean v2 model or v3/v4 3D-to-LLM adapter checkpoint.

        Args:
            checkpoint: Path to a v2 ``best.pt`` (``model_state_dict``) or v3/v4 adapter
                ``best_adapter.pt`` (``bridge_state_dict`` + ``lora_state_dict``).
            mode: ``"auto"`` (detect from checkpoint keys), ``"v2"``, or ``"llm"``.
            v2_pocket_ckpt: Optional v2 pocket encoder checkpoint path for ``llm`` mode.
            backbone_id: Optional HF backbone override for ``llm`` checkpoints.
            vocab_path: Optional JSON vocabulary path for ``v2`` checkpoints.
            cache_dir: Hugging Face cache directory for ``llm`` backbones.
            device: Target device (``"auto"``, ``"mps"``, ``"cuda"``, ``"cpu"``).
            dtype: Backbone dtype (``"auto"``, ``"float32"``, ``"bfloat16"``).
            local_files_only: Prevent network downloads from Hugging Face Hub.

        Returns:
            Initialized :class:`MoleculeDesigner` ready for :meth:`design` and :meth:`decorate`.
        """
        import torch

        from geneuclidean.config import select_device

        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            raise StructureNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise CheckpointError(f"Unsupported checkpoint payload type: {type(payload).__name__}")

        if mode == "auto":
            if "bridge_state_dict" in payload:
                resolved_mode: Literal["v2", "llm"] = "llm"
            elif "model_state_dict" in payload:
                resolved_mode = "v2"
            else:
                raise CheckpointError(
                    f"Cannot infer checkpoint mode from keys {sorted(payload.keys())}; "
                    "expected 'bridge_state_dict' (llm) or 'model_state_dict' (v2)."
                )
        else:
            resolved_mode = mode

        torch_device = select_device(device)
        if resolved_mode == "v2":
            return cls._load_v2_designer(
                payload=payload,
                vocab_path=vocab_path,
                torch_device=torch_device,
            )
        return cls._load_llm_designer(
            ckpt_path=ckpt_path,
            payload=payload,
            v2_pocket_ckpt=v2_pocket_ckpt,
            backbone_id=backbone_id,
            cache_dir=cache_dir,
            dtype=dtype,
            local_files_only=local_files_only,
            torch_device=torch_device,
        )

    @classmethod
    def _load_v2_designer(
        cls,
        *,
        payload: dict[str, Any],
        vocab_path: str | Path | None,
        torch_device: Any,
    ) -> MoleculeDesigner:
        from geneuclidean.config import ExperimentConfig
        from geneuclidean.data.tokenizer import MolecularTokenizer
        from geneuclidean.models.geneuclidean import GenEuclideanModel

        raw_cfg = payload.get("config")
        cfg = (
            ExperimentConfig.model_validate(raw_cfg)
            if isinstance(raw_cfg, dict)
            else ExperimentConfig()
        )
        tok_path = (
            Path(vocab_path)
            if vocab_path is not None
            else Path(cfg.data.processed_dir) / f"vocab_{cfg.data.representation}.json"
        )
        if not tok_path.is_file():
            raise StructureNotFoundError(
                f"Tokenizer vocabulary file not found at {tok_path}.",
                hint="Pass vocab_path=/path/to/vocab_smiles.json explicitly.",
            )
        tokenizer = MolecularTokenizer.load(tok_path)
        cfg = cfg.model_copy(
            update={"decoder": cfg.decoder.model_copy(update={"vocab_size": tokenizer.vocab_size})}
        )
        model = GenEuclideanModel(cfg)
        model.load_state_dict(payload["model_state_dict"])
        model.to(torch_device)
        model.eval()
        return cls(
            backend="v2",
            model=model,
            device=torch_device,
            cfg=cfg,
            tokenizer=tokenizer,
        )

    @classmethod
    def _load_llm_designer(
        cls,
        *,
        ckpt_path: Path,
        payload: dict[str, Any],
        v2_pocket_ckpt: str | Path | None,
        backbone_id: str | None,
        cache_dir: str,
        dtype: str,
        local_files_only: bool,
        torch_device: Any,
    ) -> MoleculeDesigner:
        from geneuclidean.config import ExperimentConfig

        try:
            from geneuclidean.models.hf_backbones import load_hf_backbone
            from geneuclidean.models.llm_bridge import PocketConditionedLLM
        except ImportError as exc:
            raise OptionalDependencyError(
                "LLM checkpoint loading requires optional LLM dependencies.",
                hint="Install them via: pip install 'geneuclidean[llm]'",
            ) from exc

        meta: dict[str, Any] = dict(payload.get("metadata", {}))
        raw_v2_cfg = meta.get("v2_cfg")
        v2_cfg = (
            ExperimentConfig.model_validate(raw_v2_cfg)
            if isinstance(raw_v2_cfg, dict)
            else ExperimentConfig()
        )
        eff_backbone_id = str(
            backbone_id or meta.get("backbone_id", "ibm-research/GP-MoLFormer-Uniq")
        )
        has_lora = bool(payload.get("lora_state_dict"))
        backbone = load_hf_backbone(
            eff_backbone_id,
            cache_dir=cache_dir,
            dtype=dtype,
            use_lora=has_lora,
            local_files_only=local_files_only,
        )
        llm_model = PocketConditionedLLM(
            v2_cfg=v2_cfg,
            backbone=backbone,
            num_queries=int(meta.get("num_queries", 16)),
            perceiver_layers=int(meta.get("perceiver_layers", 2)),
            perceiver_heads=int(meta.get("perceiver_heads", 8)),
            num_global_tokens=int(meta.get("num_global_tokens", 4)),
            cfg_drop_prob=float(meta.get("cfg_drop_prob", 0.15)),
        )
        eff_v2_ckpt = v2_pocket_ckpt or meta.get("v2_ckpt_path")
        if eff_v2_ckpt is not None and Path(str(eff_v2_ckpt)).is_file():
            llm_model.load_v2_pocket_checkpoint(Path(str(eff_v2_ckpt)), map_location="cpu")
        llm_model.load_adapter_checkpoint(ckpt_path, map_location="cpu")
        llm_model.to(torch_device)
        llm_model.eval()
        return cls(
            backend="llm",
            model=llm_model,
            device=torch_device,
            cfg=v2_cfg,
            tokenizer=backbone.tokenizer,
            backbone_id=eff_backbone_id,
        )

    def _prepare_pocket(
        self, pocket: Pocket | PocketFeatures
    ) -> tuple[Any, Any, Any, str | None, str]:
        """Resolve pocket features into batched ``[1, N, ...]`` tensors on ``self.device``."""
        import torch

        if isinstance(pocket, Pocket):
            feat = pocket.featurize()
            target_id = pocket.target_id
            pocket_id = pocket.target_id or pocket.pdb_id
        else:
            feat = pocket
            target_id = None
            pocket_id = feat.pdb_id.upper()

        coords = (
            torch.from_numpy(feat.coords).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        )
        node_feats = (
            torch.from_numpy(feat.node_feats)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
        )
        atom_mask = torch.ones((1, len(feat.coords)), dtype=torch.bool, device=self.device)
        return coords, node_feats, atom_mask, target_id, pocket_id

    def _score_with_heads(
        self,
        pocket_out: Any,
        raw_smiles: Sequence[str],
    ) -> tuple[list[float] | None, list[float] | None]:
        """Evaluate optional warm-started v2 pKd and DrugCLIP heads on valid candidates."""
        if not hasattr(self.model, "score_candidates") or not raw_smiles:
            return None, None

        import numpy as np
        import torch

        from geneuclidean.data.featurizer import featurize_ligand_smiles

        fps: list[np.ndarray] = []
        valid_idx: list[int] = []
        fp_dim = int(self.cfg.multitask.ligand_fp_dim)
        for idx, smi in enumerate(raw_smiles):
            parsed = featurize_ligand_smiles(smi, fp_size=fp_dim)
            if parsed is not None:
                valid_idx.append(idx)
                fps.append(parsed[1])
        if not valid_idx:
            return None, None

        fp_tensor = torch.from_numpy(np.stack(fps, axis=0)).to(
            device=self.device, dtype=torch.float32
        )
        head_scores = self.model.score_candidates(pocket_out, fp_tensor)
        pred_pkd: list[float] | None = None
        clip_sim: list[float] | None = None
        if "pkd" in head_scores:
            pred_pkd = [float("nan")] * len(raw_smiles)
            for idx, val in zip(valid_idx, head_scores["pkd"].detach().cpu().tolist(), strict=True):
                pred_pkd[idx] = float(val)
        if "clip_sim" in head_scores:
            clip_sim = [float("nan")] * len(raw_smiles)
            for idx, val in zip(
                valid_idx, head_scores["clip_sim"].detach().cpu().tolist(), strict=True
            ):
                clip_sim[idx] = float(val)
        return pred_pkd, clip_sim

    def _finalize_result(
        self,
        *,
        pocket_id: str,
        target_id: str | None,
        raw_smiles: list[str],
        pred_pkd: list[float] | None,
        clip_sim: list[float] | None,
        cns_safe_only: bool,
        include_triage: bool,
        include_admet: bool,
        admet_profile: Literal["v3", "v4", "auto"],
        oracle_path: str | Path | None,
        extra_scorers: Sequence[MoleculeScorerProtocol | tuple[MoleculeScorerProtocol, float]]
        | None,
        filters: Sequence[MoleculeFilterProtocol] | None,
        rerank: bool,
        top_k_hits: int | None,
        metadata: dict[str, Any],
        callbacks: Sequence[DesignCallbackProtocol] | None = None,
        pocket: Pocket | PocketFeatures | None = None,
    ) -> DesignResult:
        all_reports = score_molecules(
            raw_smiles,
            target=target_id,
            include_triage=include_triage,
            include_admet=include_admet,
            admet_profile=admet_profile,
            oracle_path=oracle_path,
            pred_pkd=pred_pkd,
            clip_sim=clip_sim,
            extra_scorers=extra_scorers,
        )
        for cb in callbacks or ():
            cb.on_candidates_generated(pocket, raw_smiles, all_reports)

        valid_reports = [r for r in all_reports if r.valid]
        valid_rate = len(valid_reports) / max(len(raw_smiles), 1)
        unique_canon = {r.canonical_smiles for r in valid_reports if r.canonical_smiles}
        unique_rate = len(unique_canon) / max(len(valid_reports), 1) if valid_reports else 0.0

        filtered = _deduplicate_reports(valid_reports)
        if cns_safe_only:
            filtered = [r for r in filtered if r.is_cns_lead]

        stage_log: list[dict[str, Any]] = []
        for flt in filters or ():
            before = len(filtered)
            filtered = [r for r in filtered if flt.keep(r)]
            stage_log.append({"filter": flt.name, "in": before, "out": len(filtered)})
        if stage_log:
            metadata["filter_stages"] = stage_log

        if rerank:
            filtered.sort(key=lambda r: r.composite_reward, reverse=True)
        if top_k_hits is not None:
            filtered = filtered[: int(top_k_hits)]

        result = DesignResult(
            pocket_id=pocket_id,
            backend=f"{self.backend}:{self.backbone_id}",
            raw_smiles=raw_smiles,
            reports=filtered,
            valid_rate=valid_rate,
            unique_rate=unique_rate,
            metadata=metadata,
        )
        for cb in callbacks or ():
            cb.on_design_end(pocket, result)
        return result

    def design(
        self,
        pocket: Pocket | PocketFeatures,
        *,
        n: int = 32,
        strategy: str = "min_p",
        temperature: float = 1.0,
        min_p: float = 0.08,
        top_p: float = 0.92,
        top_k: int = 30,
        cfg_scale: float = 1.6,
        latent_temp: float = 1.0,
        max_new_tokens: int = 96,
        cns_safe_only: bool = False,
        include_triage: bool = True,
        include_admet: bool = True,
        admet_profile: Literal["v3", "v4", "auto"] = "v3",
        oracle_path: str | Path | None = None,
        extra_scorers: Sequence[MoleculeScorerProtocol | tuple[MoleculeScorerProtocol, float]]
        | None = None,
        filters: Sequence[MoleculeFilterProtocol] | None = None,
        callbacks: Sequence[DesignCallbackProtocol] | None = None,
        rerank: bool = True,
        top_k_hits: int | None = None,
        seed: int = 42,
    ) -> DesignResult:
        """Generate and score *de novo* candidate ligands conditioned on a 3D binding pocket."""
        import torch

        run_meta: dict[str, Any] = {
            "mode": "de_novo",
            "n_requested": int(n),
            "strategy": strategy,
            "cfg_scale": float(cfg_scale),
            "temperature": float(temperature),
            "cns_safe_only": bool(cns_safe_only),
            "admet_profile": admet_profile,
            "seed": int(seed),
        }
        for cb in callbacks or ():
            cb.on_design_start(pocket, run_meta)

        coords, node_feats, atom_mask, target_id, pocket_id = self._prepare_pocket(pocket)

        with torch.no_grad():
            pocket_out = self.model.encode_pocket(coords, node_feats, atom_mask)

            if self.backend == "llm":
                from geneuclidean.sampling.llm_sampler import sample_llm_molecules

                raw_smiles = sample_llm_molecules(
                    model=self.model,
                    pocket_out=pocket_out,
                    atom_mask=atom_mask,
                    num_samples=int(n),
                    max_new_tokens=int(max_new_tokens),
                    strategy=strategy,
                    temperature=float(temperature),
                    min_p=float(min_p),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    cfg_scale=float(cfg_scale),
                    latent_temp=float(latent_temp),
                    seed=int(seed),
                )
            else:
                from geneuclidean.sampling.decoder_strategies import sample_molecules_detailed

                sampling_cfg = self.cfg.sampling.model_copy(
                    update={
                        "strategy": strategy,
                        "num_samples": int(n),
                        "max_len": int(max_new_tokens),
                        "temperature": float(temperature),
                        "min_p": float(min_p),
                        "top_p": float(top_p),
                        "top_k": int(top_k),
                        "cfg_scale": float(cfg_scale),
                        "latent_temp": float(latent_temp),
                    }
                )
                gen = torch.Generator(device=self.device).manual_seed(int(seed))
                batch = {"coords": coords, "node_feats": node_feats, "atom_mask": atom_mask}
                samp_res = sample_molecules_detailed(
                    model=self.model,
                    pocket_batch=batch,
                    tokenizer=self.tokenizer,
                    sampling_cfg=sampling_cfg,
                    generator=gen,
                    device=self.device,
                    return_attn=False,
                )
                raw_smiles = list(samp_res.smiles)

            pred_pkd, clip_sim = self._score_with_heads(pocket_out, raw_smiles)

        return self._finalize_result(
            pocket_id=pocket_id,
            target_id=target_id,
            raw_smiles=raw_smiles,
            pred_pkd=pred_pkd,
            clip_sim=clip_sim,
            cns_safe_only=cns_safe_only,
            include_triage=include_triage,
            include_admet=include_admet,
            admet_profile=admet_profile,
            oracle_path=oracle_path,
            extra_scorers=extra_scorers,
            filters=filters,
            rerank=rerank,
            top_k_hits=top_k_hits,
            metadata=run_meta,
            callbacks=callbacks,
            pocket=pocket,
        )

    def decorate(
        self,
        pocket: Pocket | PocketFeatures,
        core_smiles_with_dummy: str,
        *,
        n: int = 16,
        strategy: str = "min_p",
        temperature: float = 0.9,
        min_p: float = 0.08,
        top_p: float = 0.92,
        top_k: int = 30,
        cfg_scale: float = 1.6,
        latent_temp: float = 1.0,
        max_new_tokens: int = 64,
        attachment_start: int | None = None,
        cns_safe_only: bool = False,
        include_triage: bool = True,
        include_admet: bool = True,
        admet_profile: Literal["v3", "v4", "auto"] = "v3",
        oracle_path: str | Path | None = None,
        extra_scorers: Sequence[MoleculeScorerProtocol | tuple[MoleculeScorerProtocol, float]]
        | None = None,
        filters: Sequence[MoleculeFilterProtocol] | None = None,
        callbacks: Sequence[DesignCallbackProtocol] | None = None,
        rerank: bool = True,
        top_k_hits: int | None = None,
        seed: int = 42,
    ) -> DesignResult:
        """Run 3D-pocket-conditioned SAFE/LLM scaffold decoration & R-group optimization.

        Requires an ``llm`` backend whose underlying ``PocketConditionedLLM`` supports
        :meth:`~geneuclidean.models.llm_bridge.PocketConditionedLLM.decorate_scaffold`.

        Args:
            pocket: Target 3D :class:`Pocket` or precomputed :class:`PocketFeatures`.
            core_smiles_with_dummy: Scaffold core containing attachment point(s) such as
                ``"[*]c1ccc2cc1"`` or ``"COc1cc2c(cc1OC)C(=O)C([*])C2"``.
        """
        if self.backend != "llm" or not hasattr(self.model, "decorate_scaffold"):
            raise CheckpointError(
                "Scaffold decoration requires a PocketConditionedLLM ('llm') designer.",
                hint=(
                    "Load a SAFE-GPT / GP-MoLFormer / NovoMolGen adapter via "
                    "MoleculeDesigner.from_pretrained(...)."
                ),
            )

        import torch

        run_meta: dict[str, Any] = {
            "mode": "scaffold_decoration",
            "core_smiles": core_smiles_with_dummy,
            "n_requested": int(n),
            "strategy": strategy,
            "cfg_scale": float(cfg_scale),
            "temperature": float(temperature),
            "cns_safe_only": bool(cns_safe_only),
            "admet_profile": admet_profile,
            "seed": int(seed),
        }
        for cb in callbacks or ():
            cb.on_design_start(pocket, run_meta)

        coords, node_feats, atom_mask, target_id, pocket_id = self._prepare_pocket(pocket)
        with torch.no_grad():
            pocket_out = self.model.encode_pocket(coords, node_feats, atom_mask)
            dec_rows = self.model.decorate_scaffold(
                core_smiles_with_dummy,
                pocket_out=pocket_out,
                atom_mask=atom_mask,
                num_samples=int(n),
                max_new_tokens=int(max_new_tokens),
                strategy=strategy,
                temperature=float(temperature),
                min_p=float(min_p),
                top_p=float(top_p),
                top_k=int(top_k),
                cfg_scale=float(cfg_scale),
                latent_temp=float(latent_temp),
                attachment_start=attachment_start,
                seed=int(seed),
            )
            raw_smiles = [str(row.get("smiles") or row.get("raw_text") or "") for row in dec_rows]
            pred_pkd, clip_sim = self._score_with_heads(pocket_out, raw_smiles)

        return self._finalize_result(
            pocket_id=pocket_id,
            target_id=target_id,
            raw_smiles=raw_smiles,
            pred_pkd=pred_pkd,
            clip_sim=clip_sim,
            cns_safe_only=cns_safe_only,
            include_triage=include_triage,
            include_admet=include_admet,
            admet_profile=admet_profile,
            oracle_path=oracle_path,
            extra_scorers=extra_scorers,
            filters=filters,
            rerank=rerank,
            top_k_hits=top_k_hits,
            metadata=run_meta,
            callbacks=callbacks,
            pocket=pocket,
        )


__all__ = ["DesignResult", "MoleculeDesigner"]
