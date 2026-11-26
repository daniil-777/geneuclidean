"""Pluggable ligand embedding encoders for 3D contrastive virtual screening.

Provides :data:`LIGAND_ENCODER_REGISTRY` with three interchangeable backends:
1. ``"drugclip_v2"``: GenEuclidean's trained ``PaiNN-SE3``-aligned ``DrugCLIP`` ligand
   projection head (zero extra downloads beyond the ``v2`` checkpoint).
2. ``"ecfp4_binary"``: Pure-RDKit L2-normalized Morgan ECFP4 baseline for clean ablations.
3. ``"ibm_mmelon"``: Lazy optional adapter for IBM Research's Apache-2.0 multi-view
   molecular foundation model (``ibm-research/biomed.sm.mv-te-84m``) requiring ``.[llm]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from geneuclidean.exceptions import OptionalDependencyError
from geneuclidean.registry import Registry

if TYPE_CHECKING:
    from geneuclidean.models.geneuclidean import GenEuclideanModel

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]
RDLogger.DisableLog("rdApp.warning")  # type: ignore[attr-defined]


def featurize_ecfp4_batch(
    smiles_list: Sequence[str],
    *,
    n_bits: int = 1024,
    radius: int = 2,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Compute dense ``[N, n_bits]`` float32 Morgan ECFP4 fingerprints and validity mask."""
    n = len(smiles_list)
    mat = np.zeros((n, int(n_bits)), dtype=np.float32)
    valid = np.zeros(n, dtype=np.bool_)
    buf = np.zeros(int(n_bits), dtype=np.float32)
    for idx, smi in enumerate(smiles_list):
        if not isinstance(smi, str) or not smi.strip():
            continue
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(n_bits))  # type: ignore[attr-defined]
        buf.fill(0.0)
        DataStructs.ConvertToNumpyArray(bv, buf)
        mat[idx] = buf
        valid[idx] = True
    return mat, valid


class ECFP4BinaryEncoder:
    """Zero-PyTorch L2-normalized Morgan ECFP4 ligand encoder."""

    def __init__(self, *, n_bits: int = 1024, radius: int = 2) -> None:
        self.n_bits = int(n_bits)
        self.radius = int(radius)

    @property
    def name(self) -> str:
        return "ecfp4_binary"

    @property
    def dim(self) -> int:
        return self.n_bits

    def encode(
        self, smiles_list: Sequence[str], *, batch_size: int = 2048
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        _ = batch_size
        mat, valid = featurize_ecfp4_batch(smiles_list, n_bits=self.n_bits, radius=self.radius)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        return (mat / norms).astype(np.float32, copy=False), valid


class DrugCLIPLigandEncoder:
    """Encode ligands into GenEuclidean's 3D-pocket-aligned ``DrugCLIP`` unit hypersphere."""

    def __init__(self, model: GenEuclideanModel, *, device: str = "cpu") -> None:
        import torch

        self.model = model
        self.device = torch.device(device)
        self.fp_dim = int(model.cfg.multitask.ligand_fp_dim)
        self._clip_dim = int(model.cfg.multitask.clip_dim)

    @property
    def name(self) -> str:
        return "drugclip_v2"

    @property
    def dim(self) -> int:
        return self._clip_dim

    def encode(
        self, smiles_list: Sequence[str], *, batch_size: int = 2048
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        import torch
        import torch.nn.functional as F

        if self.model.heads.clip_ligand_proj is None:
            raise ValueError("Supplied GenEuclideanModel does not have an enabled DrugCLIP head.")

        n = len(smiles_list)
        out = np.zeros((n, self._clip_dim), dtype=np.float32)
        valid_all = np.zeros(n, dtype=np.bool_)
        self.model.eval()
        bs = max(1, int(batch_size))
        with torch.no_grad():
            for start in range(0, n, bs):
                chunk = smiles_list[start : start + bs]
                fp_np, valid = featurize_ecfp4_batch(chunk, n_bits=self.fp_dim)
                valid_all[start : start + len(chunk)] = valid
                if not bool(valid.any()):
                    continue
                fp_t = torch.from_numpy(fp_np).to(self.device)
                lig_emb = self.model.heads.encode_ligand(fp_t)
                clip_vec = F.normalize(
                    self.model.heads.clip_ligand_proj(lig_emb), p=2, dim=-1, eps=1e-8
                )
                arr = clip_vec.detach().cpu().numpy().astype(np.float32, copy=False)
                arr[~valid] = 0.0
                out[start : start + len(chunk)] = arr
        return out, valid_all


class IBMMMelonLigandEncoder:
    """Optional lazy adapter for IBM Research's multi-view ``MMELON`` foundation model.

    Loads ``ibm-research/biomed.sm.mv-te-84m`` (Apache-2.0) lazily on first :meth:`encode`
    call when ``transformers`` is installed, enabling multi-view 2D/SMILES foundation
    representations to be fused or benchmarked alongside chiral 3D ``PaiNN-SE3`` pockets.
    """

    def __init__(
        self,
        model_id: str = "ibm-research/biomed.sm.mv-te-84m",
        *,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        self.model_id = str(model_id)
        self.device = str(device)
        self.local_files_only = bool(local_files_only)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._dim: int = 768

    @property
    def name(self) -> str:
        return "ibm_mmelon"

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise OptionalDependencyError(
                "IBMMMelonLigandEncoder requires optional `transformers` dependency.",
                hint="Install with `pip install -e '.[llm]'`.",
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        ).to(torch.device(self.device))
        self._model.eval()
        cfg = getattr(self._model, "config", None)
        self._dim = int(getattr(cfg, "hidden_size", 768))

    def encode(
        self, smiles_list: Sequence[str], *, batch_size: int = 64
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        import torch
        import torch.nn.functional as F

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        valid = np.array(
            [
                bool(isinstance(s, str) and s.strip() and Chem.MolFromSmiles(s.strip()))
                for s in smiles_list
            ],
            dtype=np.bool_,
        )
        n = len(smiles_list)
        out = np.zeros((n, self._dim), dtype=np.float32)
        bs = max(1, int(batch_size))
        with torch.no_grad():
            for start in range(0, n, bs):
                chunk = [
                    s.strip() if isinstance(s, str) else "" for s in smiles_list[start : start + bs]
                ]
                enc = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                enc = {k: v.to(torch.device(self.device)) for k, v in enc.items()}
                res = self._model(**enc)
                hidden = res.last_hidden_state
                mask = enc.get("attention_mask")
                if mask is not None:
                    w = mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
                else:
                    pooled = hidden.mean(dim=1)
                normed = F.normalize(pooled.float(), p=2, dim=-1, eps=1e-8)
                arr = normed.detach().cpu().numpy().astype(np.float32, copy=False)
                arr[~valid[start : start + len(chunk)]] = 0.0
                out[start : start + len(chunk)] = arr
        return out, valid


LIGAND_ENCODER_REGISTRY: Registry[Any] = Registry("ligand_encoders")
LIGAND_ENCODER_REGISTRY.register("ecfp4_binary", ECFP4BinaryEncoder)
LIGAND_ENCODER_REGISTRY.register("drugclip_v2", DrugCLIPLigandEncoder)
LIGAND_ENCODER_REGISTRY.register("ibm_mmelon", IBMMMelonLigandEncoder, aliases=("mmelon",))

__all__ = [
    "DrugCLIPLigandEncoder",
    "ECFP4BinaryEncoder",
    "IBMMMelonLigandEncoder",
    "LIGAND_ENCODER_REGISTRY",
    "featurize_ecfp4_batch",
]
