#!/usr/bin/env python3
"""Real 3-Tier Apple Silicon MPS Benchmark & Symmetry Scorecard for GenEuclidean 3D Encoders.

Evaluates all 8 canonical 3D SE(3)/E(3) pocket encoders (``schnet``, ``egnn``, ``painn``,
``e3nn``, ``dense_painn_se3``, ``cartformer_se3``, ``atomgs_se3``, ``subpocket_vn_jepa``)
strictly under ``PYTORCH_ENABLE_MPS_FALLBACK=0`` on Apple M3 Pro MPS (or CUDA/CPU fallback
if MPS is unavailable) and writes a structured JSON report to
``results/tables/encoders_v52_mps_benchmark.json``:

- **Micro-benchmark:** Row-vectorized ``dense_knn.gather_neighbors`` (``torch.index_select``)
  vs. legacy elementwise ``torch.gather`` on MPS.
- **Tier 1 (MPS Throughput & Memory):** Side-by-side forward and forward+backward latency (ms),
  memory footprint (MB), and parameter counts across all 8 encoders at ``(B=16, N=384, K=24)``
  and ``(B=4, N=1024, K=24)``, plus ``AtomGS-SE3`` analytical 3D ROCS overlap (``384 x 40``).
- **Tier 2 (Exact Symmetry & Chirality Scorecard):** ``SO(3) + 100 A`` max error, chiral
  ``(R)/(S)`` enantiomer & 3D pocket mirror discrimination, ``E(3)`` toggle parity, asymmetric
  planar benzene reflection consistency, 26-neighbor cube tie-swap error, and dummy corruption.
- **Tier 3 (Real ``lp_pdbbind`` / ``CASF-2016`` pKd Head-to-Head + JEPA SSL Ablation):**
  Strictly leak-free train (``lp_pdbbind.train``) / validation (``lp_pdbbind.val``) / held-out
  evaluation (``CASF-2016 Core`` + ``lp_pdbbind.test``) comparing ``PaiNN-SE3``,
  ``DensePaiNN-SE3``, ``CartFormer-SE3``, ``AtomGS-SE3``, ``SubPocketVN-JEPA`` (scratch),
  and ``SubPocketVN-JEPA`` pretrained with Sonata/Concerto 3D JEPA SSL on real training pockets.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch import Tensor

from geneuclidean.config import EncoderConfig, seed_everything
from geneuclidean.models.encoders import (
    AtomGSSE3Encoder,
    PocketEncoderOutput,
    build_encoder,
    gather_neighbors,
)
from geneuclidean.models.encoders.gaussian_ops import rocs_self_overlap
from geneuclidean.models.heads import LigandFingerprintEncoder
from geneuclidean.training.jepa_ssl import SubPocketJEPAPretrainer
from geneuclidean.training.losses import compute_drugclip_loss

ALL_ARCHS: tuple[str, ...] = (
    "schnet",
    "egnn",
    "painn",
    "e3nn",
    "dense_painn_se3",
    "cartformer_se3",
    "atomgs_se3",
    "subpocket_vn_jepa",
)
SE3_CHIRAL_ARCHS: frozenset[str] = frozenset(
    {
        "painn",
        "e3nn",
        "dense_painn_se3",
        "cartformer_se3",
        "atomgs_se3",
        "subpocket_vn_jepa",
    }
)
TIER3_ARCHS: tuple[str, ...] = (
    "painn",
    "dense_painn_se3",
    "cartformer_se3",
    "atomgs_se3",
    "subpocket_vn_jepa",
)


def _sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _empty_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _device_memory_mb(device: torch.device) -> tuple[float, float]:
    """Return ``(current_allocated_mb, driver_allocated_mb)`` for MPS/CUDA (0.0 on CPU)."""
    if device.type == "mps":
        cur = float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
        drv = float(torch.mps.driver_allocated_memory()) / (1024.0 * 1024.0)
        return round(cur, 2), round(drv, 2)
    if device.type == "cuda":
        cur = float(torch.cuda.memory_allocated(device)) / (1024.0 * 1024.0)
        drv = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        return round(cur, 2), round(drv, 2)
    return 0.0, 0.0


def _random_so3_matrix(gen: torch.Generator, device: torch.device) -> Tensor:
    """Sample a Haar-uniform proper rotation matrix R in SO(3) (det(R) = +1)."""
    mat = torch.randn((3, 3), generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    q = q * diag_sign.unsqueeze(0)
    if float(torch.det(q).item()) < 0.0:
        q[:, 0] = -q[:, 0]
    out: Tensor = q.to(device=device, dtype=torch.float32)
    return out


def _measure_cold_import() -> dict[str, Any]:
    """Measure cold ``import geneuclidean`` latency and verify zero eager heavy imports."""
    probe = (
        "import json, sys, time; "
        "t0 = time.perf_counter(); "
        "import geneuclidean as ge; "
        "dt = (time.perf_counter() - t0) * 1000.0; "
        "heavy_mods = ('torch','rdkit','transformers','peft',"
        "'sklearn','scipy','matplotlib','e3nn'); "
        "heavy = [m for m in heavy_mods if m in sys.modules]; "
        "out = {'cold_import_ms': round(dt, 2), 'heavy': heavy, "
        "'n_encoders': len(ge.list_encoders())}; "
        "print(json.dumps(out))"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    result: dict[str, Any] = json.loads(proc.stdout.strip())
    return result


def _legacy_torch_gather(x: Tensor, nbr_idx: Tensor) -> Tensor:
    """Reference pre-optimization ``torch.gather`` implementation for direct MPS comparison."""
    batch_size, num_atoms, k = nbr_idx.shape
    feat_shape = x.shape[2:]
    flat_feat = x.reshape(batch_size, num_atoms, -1)
    feat_dim = flat_feat.shape[-1]
    flat_idx = nbr_idx.reshape(batch_size, num_atoms * k, 1).expand(-1, -1, feat_dim)
    return torch.gather(flat_feat, dim=1, index=flat_idx).reshape(
        batch_size, num_atoms, k, *feat_shape
    )


def run_gather_neighbors_microbench(device: torch.device) -> dict[str, Any]:
    """Benchmark row-vectorized ``gather_neighbors`` vs. legacy ``torch.gather``."""
    results: dict[str, Any] = {}
    configs = [
        ("B16_N384_C192_K24", 16, 384, 192, 24),
        ("B4_N1024_C192_K24", 4, 1024, 192, 24),
    ]
    gen = torch.Generator(device="cpu").manual_seed(2026)
    for label, bsz, num_atoms, channels, k in configs:
        x = torch.randn((bsz, num_atoms, channels), generator=gen, dtype=torch.float32).to(device)
        idx = torch.randint(0, num_atoms, (bsz, num_atoms, k), generator=gen, dtype=torch.long).to(
            device
        )
        with torch.no_grad():
            y_new = gather_neighbors(x, idx)
            y_old = _legacy_torch_gather(x, idx)
            max_diff = float((y_new - y_old).abs().max().item())

            for _ in range(5):
                _ = gather_neighbors(x, idx)
                _ = _legacy_torch_gather(x, idx)
            _sync_device(device)

            new_times: list[float] = []
            for _ in range(25):
                t0 = time.perf_counter()
                _ = gather_neighbors(x, idx)
                _sync_device(device)
                new_times.append((time.perf_counter() - t0) * 1000.0)

            old_times: list[float] = []
            for _ in range(25):
                t0 = time.perf_counter()
                _ = _legacy_torch_gather(x, idx)
                _sync_device(device)
                old_times.append((time.perf_counter() - t0) * 1000.0)

        new_med = float(statistics.median(new_times))
        old_med = float(statistics.median(old_times))
        results[label] = {
            "legacy_torch_gather_ms": round(old_med, 3),
            "optimized_index_select_ms": round(new_med, 3),
            "speedup_x": round(old_med / max(new_med, 1e-6), 2),
            "max_abs_diff": max_diff,
        }
        del x, idx, y_new, y_old
        _empty_device_cache(device)
    return results


def _make_protein_density_batch(
    batch_size: int,
    num_atoms: int,
    device: torch.device,
    *,
    seed: int = 42,
    records: dict[str, dict[str, Any]] | None = None,
    real_ids: list[str] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create a realistic protein-density 3D pocket batch (real PDBbind pockets when N=384)."""
    if records is not None and real_ids is not None and num_atoms == 384:
        full_pockets = [pid for pid in real_ids if len(records[pid]["coords"]) >= 360]
        if len(full_pockets) >= batch_size:
            coords = torch.zeros((batch_size, num_atoms, 3), dtype=torch.float32)
            feats = torch.zeros((batch_size, num_atoms, 52), dtype=torch.float32)
            mask = torch.zeros((batch_size, num_atoms), dtype=torch.bool)
            for b, pid in enumerate(full_pockets[:batch_size]):
                rec = records[pid]
                n = min(num_atoms, len(rec["coords"]))
                coords[b, :n] = torch.from_numpy(np.asarray(rec["coords"][:n], dtype=np.float32))
                feats[b, :n] = torch.from_numpy(np.asarray(rec["node_feats"][:n], dtype=np.float32))
                mask[b, :n] = True
            return coords.to(device), feats.to(device), mask.to(device)

    # Match real PDBbind heavy-atom packing density (~0.107 atoms / A^3 -> mean degree ~25 < 5.5 A).
    gen = torch.Generator(device="cpu").manual_seed(seed)
    radius = (float(num_atoms) / (0.107 * (4.0 / 3.0) * math.pi)) ** (1.0 / 3.0)
    dirs = torch.randn((batch_size, num_atoms, 3), generator=gen, dtype=torch.float32)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    u = torch.rand((batch_size, num_atoms, 1), generator=gen, dtype=torch.float32).pow(1.0 / 3.0)
    coords = dirs * (u * radius)

    feats = torch.randn((batch_size, num_atoms, 52), generator=gen, dtype=torch.float32) * 0.4
    feats[..., 45] = 0.72 + 0.25 * torch.rand(
        (batch_size, num_atoms), generator=gen, dtype=torch.float32
    )
    feats[..., 46] = torch.tanh(feats[..., 46])
    mask = torch.ones((batch_size, num_atoms), dtype=torch.bool)
    return coords.to(device), feats.to(device), mask.to(device)


def run_tier1_throughput_benchmark(
    device: torch.device,
    records: dict[str, dict[str, Any]] | None = None,
    real_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Tier 1: Side-by-side MPS fwd & fwd+bwd latency, memory, and params across all 8 encoders."""
    base_cfg_kwargs: dict[str, Any] = {
        "node_in_dim": 52,
        "hidden_dim": 192,
        "vector_dim": 64,
        "out_dim": 256,
        "num_layers": 4,
        "num_rbf": 24,
        "cutoff": 5.5,
        "e3nn_lmax": 1,
        "tensor_dim": 24,
        "num_heads": 8,
        "num_landmarks": 8,
        "num_subpockets": 24,
        "use_se3_chirality": True,
        "use_coulomb_bias": True,
        "enable_3d_overlap": True,
        "dropout": 0.0,
    }
    max_neighbors = 24

    regimes = [
        ("B16_N384_K24", 16, 384, 8, 5),
        ("B4_N1024_K24", 4, 1024, 6, 4),
    ]
    regime_results: dict[str, dict[str, Any]] = {}

    for regime_name, bsz, num_atoms, fwd_iters, bwd_iters in regimes:
        coords, feats, mask = _make_protein_density_batch(
            bsz,
            num_atoms,
            device,
            seed=100 + num_atoms,
            records=records,
            real_ids=real_ids,
        )
        per_arch: dict[str, dict[str, Any]] = {}
        print(f"\n[Tier 1] Regime {regime_name} (device={device})")

        for arch in ALL_ARCHS:
            torch.manual_seed(42)
            cfg = EncoderConfig(arch=arch, **base_cfg_kwargs)
            enc = build_encoder(cfg, max_neighbors=max_neighbors).to(device)
            n_params = sum(p.numel() for p in enc.parameters())

            # Forward timing
            enc.eval()
            with torch.no_grad():
                for _ in range(3):
                    _ = enc(coords, feats, mask)
                _sync_device(device)
                fwd_times: list[float] = []
                for _ in range(fwd_iters):
                    t0 = time.perf_counter()
                    _ = enc(coords, feats, mask)
                    _sync_device(device)
                    fwd_times.append((time.perf_counter() - t0) * 1000.0)

            # Forward + Backward timing
            enc.train()
            for _ in range(2):
                enc.zero_grad(set_to_none=True)
                out: PocketEncoderOutput = enc(coords, feats, mask)
                loss = out.pooled_emb.pow(2).mean() + 0.1 * out.node_emb.pow(2).mean()
                torch.autograd.backward(loss)
            _sync_device(device)

            bwd_times: list[float] = []
            peak_cur_mb = 0.0
            peak_drv_mb = 0.0
            for _ in range(bwd_iters):
                t0 = time.perf_counter()
                enc.zero_grad(set_to_none=True)
                out = enc(coords, feats, mask)
                loss = out.pooled_emb.pow(2).mean() + 0.1 * out.node_emb.pow(2).mean()
                torch.autograd.backward(loss)
                _sync_device(device)
                bwd_times.append((time.perf_counter() - t0) * 1000.0)
                cur_mb, drv_mb = _device_memory_mb(device)
                peak_cur_mb = max(peak_cur_mb, cur_mb)
                peak_drv_mb = max(peak_drv_mb, drv_mb)

            fwd_med = float(statistics.median(fwd_times))
            fwd_mean = float(statistics.mean(fwd_times))
            bwd_med = float(statistics.median(bwd_times))
            bwd_mean = float(statistics.mean(bwd_times))

            per_arch[arch] = {
                "params": n_params,
                "fwd_ms_median": round(fwd_med, 2),
                "fwd_ms_mean": round(fwd_mean, 2),
                "fwdbwd_ms_median": round(bwd_med, 2),
                "fwdbwd_ms_mean": round(bwd_mean, 2),
                "peak_alloc_mb": round(peak_cur_mb, 2),
                "driver_alloc_mb": round(peak_drv_mb, 2),
            }
            print(
                f"  {arch:18s} | params={n_params:>10,d} | "
                f"fwd={fwd_med:7.2f} ms | fwd+bwd={bwd_med:7.2f} ms | "
                f"alloc={peak_cur_mb:6.1f} MB"
            )
            del enc, out, loss
            _empty_device_cache(device)

        painn_fwd = per_arch["painn"]["fwd_ms_median"]
        painn_bwd = per_arch["painn"]["fwdbwd_ms_median"]
        for entry in per_arch.values():
            entry["fwd_speedup_vs_painn"] = round(painn_fwd / max(entry["fwd_ms_median"], 1e-6), 2)
            entry["fwdbwd_speedup_vs_painn"] = round(
                painn_bwd / max(entry["fwdbwd_ms_median"], 1e-6), 2
            )

        regime_results[regime_name] = per_arch
        del coords, feats, mask
        _empty_device_cache(device)

    # AtomGS-SE3 3D ROCS complex overlap benchmark (B=16, N_P=384, N_L=40)
    p_coords, p_feats, p_mask = _make_protein_density_batch(
        16, 384, device, seed=555, records=records, real_ids=real_ids
    )
    l_coords, l_feats, l_mask = _make_protein_density_batch(16, 40, device, seed=556)
    l_coords = l_coords * 0.45  # Place ligand cleanly inside the pocket cavity
    atomgs_cfg = EncoderConfig(arch="atomgs_se3", **base_cfg_kwargs)
    atomgs = build_encoder(atomgs_cfg, max_neighbors=max_neighbors).to(device).eval()
    assert isinstance(atomgs, AtomGSSE3Encoder)

    with torch.no_grad():
        p_field = atomgs.encode_gaussian_field(p_coords, p_feats, p_mask)
        l_field = atomgs.encode_gaussian_field(l_coords, l_feats, l_mask)
        o_pp = rocs_self_overlap(p_field, mask=p_mask, max_neighbors=32, cutoff=12.0)
        o_ll = rocs_self_overlap(l_field, mask=l_mask)

        for _ in range(4):
            _ = atomgs.score_pocket_ligand_overlap(
                p_field,
                p_mask,
                l_field,
                l_mask,
                pocket_self_overlap=o_pp,
                ligand_self_overlap=o_ll,
            )
        _sync_device(device)

        cached_times: list[float] = []
        last_scores: dict[str, Tensor] = {}
        for _ in range(25):
            t0 = time.perf_counter()
            last_scores = atomgs.score_pocket_ligand_overlap(
                p_field,
                p_mask,
                l_field,
                l_mask,
                pocket_self_overlap=o_pp,
                ligand_self_overlap=o_ll,
            )
            _sync_device(device)
            cached_times.append((time.perf_counter() - t0) * 1000.0)

        uncached_times: list[float] = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = atomgs.score_pocket_ligand_overlap(
                p_field,
                p_mask,
                l_field,
                l_mask,
                max_self_neighbors=32,
            )
            _sync_device(device)
            uncached_times.append((time.perf_counter() - t0) * 1000.0)

    rocs_summary = {
        "batch_size": 16,
        "pocket_atoms": 384,
        "ligand_atoms": 40,
        "cached_self_overlap_ms_median": round(float(statistics.median(cached_times)), 3),
        "cached_self_overlap_ms_mean": round(float(statistics.mean(cached_times)), 3),
        "uncached_full_ms_median": round(float(statistics.median(uncached_times)), 3),
        "mean_rocs_tanimoto": round(float(last_scores["rocs_shape_tanimoto"].mean().item()), 4),
        "all_finite": all(bool(torch.isfinite(v).all().item()) for v in last_scores.values()),
    }
    print(
        f"  [AtomGS 3D ROCS overlap B=16, 384x40] "
        f"cached={rocs_summary['cached_self_overlap_ms_median']:.3f} ms | "
        f"uncached={rocs_summary['uncached_full_ms_median']:.3f} ms"
    )
    del atomgs, p_coords, p_feats, p_mask, l_coords, l_feats, l_mask, p_field, l_field
    _empty_device_cache(device)

    return {
        "config": {**base_cfg_kwargs, "max_neighbors": max_neighbors},
        "regimes": regime_results,
        "atomgs_3d_rocs_overlap_B16_384x40": rocs_summary,
    }


def _make_tier2_geometries(device: torch.device) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
    """Construct exact geometric fixtures for Tier 2 symmetry, chirality, and tie-swap tests."""
    gen = torch.Generator(device="cpu").manual_seed(2026)

    # 1. Realistic 32-atom chiral pocket batch (B=2, variable valid atom masks)
    p_coords = torch.randn((2, 32, 3), generator=gen, dtype=torch.float32) * 2.2
    p_feats = torch.randn((2, 32, 52), generator=gen, dtype=torch.float32)
    p_feats[..., 45] = 0.72 + 0.25 * torch.sigmoid(p_feats[..., 45])
    p_feats[..., 46] = torch.tanh(p_feats[..., 46])
    p_mask = torch.zeros((2, 32), dtype=torch.bool)
    p_mask[0, :27] = True
    p_mask[1, :24] = True

    # 2. Asymmetric 5-atom chiral (R)/(S) stereocenter (4 chemically & sterically distinct ligands)
    rs_coords = torch.tensor(
        [
            [
                [0.00, 0.00, 0.00],
                [1.54, 0.00, 0.00],
                [-0.48, 1.38, 0.00],
                [-0.52, -0.74, 1.25],
                [-0.60, -0.82, -1.52],
            ]
        ],
        dtype=torch.float32,
    )
    rs_feats = torch.randn((1, 5, 52), generator=gen, dtype=torch.float32)
    rs_feats[..., 45] = torch.tensor([[0.85, 0.75, 0.78, 0.90, 0.96]], dtype=torch.float32)
    rs_feats[..., 46] = torch.tensor([[0.15, -0.35, 0.45, -0.55, 0.30]], dtype=torch.float32)
    rs_mask = torch.ones((1, 5), dtype=torch.bool)

    # 3. Planar benzene ring (z=0) with chemically asymmetric atom features
    angles = torch.linspace(0.0, 2.0 * math.pi, 7, dtype=torch.float32)[:-1]
    bz_coords = torch.stack(
        (1.40 * torch.cos(angles), 1.40 * torch.sin(angles), torch.zeros_like(angles)),
        dim=-1,
    ).unsqueeze(0)
    bz_feats = torch.randn((1, 6, 52), generator=gen, dtype=torch.float32)
    bz_feats[..., 45] = 0.82 + 0.10 * torch.sigmoid(bz_feats[..., 45])
    bz_feats[..., 46] = torch.tanh(bz_feats[..., 46])
    bz_mask = torch.ones((1, 6), dtype=torch.bool)

    # 4. 26-neighbor equidistant cubic tie shell (4.8 A) + 6 inner neighbors (2.2 A) around origin
    offsets = [
        [float(dx), float(dy), float(dz)]
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    shell = torch.tensor(offsets, dtype=torch.float32)
    shell = shell / shell.norm(dim=-1, keepdim=True) * 4.8
    inner = torch.tensor(
        [
            [2.2, 0.0, 0.0],
            [-2.2, 0.0, 0.0],
            [0.0, 2.2, 0.0],
            [0.0, -2.2, 0.0],
            [0.0, 0.0, 2.2],
            [0.0, 0.0, -2.2],
        ],
        dtype=torch.float32,
    )
    center = torch.zeros((1, 3), dtype=torch.float32)
    cube_coords = torch.cat((center, inner, shell), dim=0).unsqueeze(0)  # [1, 33, 3]
    cube_feats = torch.randn((1, 33, 52), generator=gen, dtype=torch.float32)
    cube_feats[..., 45] = 0.85
    cube_feats[..., 46] = torch.tanh(cube_feats[..., 46])
    cube_mask = torch.ones((1, 33), dtype=torch.bool)

    return {
        "pocket": (p_coords.to(device), p_feats.to(device), p_mask.to(device)),
        "enantiomer": (rs_coords.to(device), rs_feats.to(device), rs_mask.to(device)),
        "benzene": (bz_coords.to(device), bz_feats.to(device), bz_mask.to(device)),
        "cube_tieswap": (cube_coords.to(device), cube_feats.to(device), cube_mask.to(device)),
    }


def run_tier2_symmetry_scorecard(device: torch.device) -> dict[str, Any]:
    """Tier 2: Exact SO(3)+100A, chirality, planar benzene, and 26-nbr tie-swap scorecard."""
    fixtures = _make_tier2_geometries(device)
    p_coords, p_feats, p_mask = fixtures["pocket"]
    rs_coords, rs_feats, rs_mask = fixtures["enantiomer"]
    bz_coords, bz_feats, bz_mask = fixtures["benzene"]
    cube_coords, cube_feats, cube_mask = fixtures["cube_tieswap"]

    reflect = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=device, dtype=torch.float32))
    gen = torch.Generator(device="cpu").manual_seed(9001)
    rots = [_random_so3_matrix(gen, device) for _ in range(5)]
    trans_list = [
        (torch.rand((1, 1, 3), generator=gen, dtype=torch.float32) * 200.0 - 100.0).to(device)
        for _ in range(5)
    ]

    # Deterministic corrupted dummy-atom inputs
    p_coords_corrupt = p_coords.clone()
    p_feats_corrupt = p_feats.clone()
    p_coords_corrupt[~p_mask] = 999.0
    p_feats_corrupt[~p_mask] = 1e4

    scorecard: dict[str, dict[str, Any]] = {}
    print(f"\n[Tier 2] Symmetry & Chirality Scorecard (device={device})")

    for arch in ALL_ARCHS:
        torch.manual_seed(42)
        cfg = EncoderConfig(
            arch=arch,
            node_in_dim=52,
            hidden_dim=64,
            vector_dim=16,
            out_dim=64,
            num_layers=3,
            num_rbf=16,
            cutoff=5.5,
            e3nn_lmax=1,
            tensor_dim=12,
            num_heads=4,
            num_landmarks=8,
            num_subpockets=12,
            use_se3_chirality=True,
            use_coulomb_bias=True,
            dropout=0.0,
        )
        enc = build_encoder(cfg, max_neighbors=24).to(device).eval()

        with torch.no_grad():
            ref = enc(p_coords, p_feats, p_mask)

            pure_rot_err = 0.0
            node_rt_err = 0.0
            pool_rt_err = 0.0
            vec_rt_err: float | None = None if arch == "schnet" else 0.0

            for rot, trans in zip(rots, trans_list, strict=True):
                out_rot = enc(p_coords @ rot.T, p_feats, p_mask)
                out_rt = enc(p_coords @ rot.T + trans, p_feats, p_mask)

                pure_rot_err = max(
                    pure_rot_err,
                    float((out_rot.node_emb - ref.node_emb).abs().max().item()),
                    float((out_rot.pooled_emb - ref.pooled_emb).abs().max().item()),
                )
                node_rt_err = max(
                    node_rt_err, float((out_rt.node_emb - ref.node_emb).abs().max().item())
                )
                pool_rt_err = max(
                    pool_rt_err, float((out_rt.pooled_emb - ref.pooled_emb).abs().max().item())
                )
                if (
                    ref.vec_emb is not None
                    and out_rot.vec_emb is not None
                    and out_rt.vec_emb is not None
                ):
                    exp_vec = ref.vec_emb @ rot.T
                    pure_rot_err = max(
                        pure_rot_err, float((out_rot.vec_emb - exp_vec).abs().max().item())
                    )
                    assert vec_rt_err is not None
                    vec_rt_err = max(
                        vec_rt_err, float((out_rt.vec_emb - exp_vec).abs().max().item())
                    )

            overall_rt_err = max(node_rt_err, pool_rt_err, vec_rt_err or 0.0)

            # Chiral (R)/(S) enantiomer & 32-atom pocket reflection diff
            out_r = enc(rs_coords, rs_feats, rs_mask)
            out_s = enc(rs_coords @ reflect.T, rs_feats, rs_mask)
            rs_diff = float((out_r.node_emb - out_s.node_emb).abs().max().item())

            out_pl = enc(p_coords, p_feats, p_mask)
            out_pd = enc(p_coords @ reflect.T, p_feats, p_mask)
            pocket_mirror_diff = float((out_pl.pooled_emb - out_pd.pooled_emb).abs().max().item())

            # Planar benzene asymmetric reflection diff (must be ~0 achiral consistency)
            out_bz_a = enc(bz_coords, bz_feats, bz_mask)
            out_bz_b = enc(bz_coords @ reflect.T, bz_feats, bz_mask)
            benzene_diff = float((out_bz_a.node_emb - out_bz_b.node_emb).abs().max().item())

            # 26-neighbor cube tie-swap invariance on central atom (index 0)
            out_cube_ref = enc(cube_coords, cube_feats, cube_mask)
            tieswap_err = 0.0
            for rot in rots:
                out_cube_rot = enc(cube_coords @ rot.T, cube_feats, cube_mask)
                err_c = float(
                    (out_cube_rot.node_emb[0, 0] - out_cube_ref.node_emb[0, 0]).abs().max().item()
                )
                tieswap_err = max(tieswap_err, err_c)

            # Dummy-atom corruption sensitivity
            out_corrupt = enc(p_coords_corrupt, p_feats_corrupt, p_mask)
            dummy_err = max(
                float((out_corrupt.node_emb - ref.node_emb).abs().max().item()),
                float((out_corrupt.pooled_emb - ref.pooled_emb).abs().max().item()),
                float(out_corrupt.node_emb[~p_mask].abs().max().item()),
            )

        # Verify use_se3_chirality=False E(3) toggle for SE(3) encoders
        e3_toggle_diff: float | None = None
        if arch in SE3_CHIRAL_ARCHS:
            torch.manual_seed(42)
            cfg_e3 = EncoderConfig(
                arch=arch,
                node_in_dim=52,
                hidden_dim=64,
                vector_dim=16,
                out_dim=64,
                num_layers=3,
                num_rbf=16,
                cutoff=5.5,
                e3nn_lmax=1,
                tensor_dim=12,
                num_heads=4,
                num_landmarks=8,
                num_subpockets=12,
                use_se3_chirality=False,
                use_coulomb_bias=True,
                dropout=0.0,
            )
            enc_e3 = build_encoder(cfg_e3, max_neighbors=24).to(device).eval()
            with torch.no_grad():
                e3_l = enc_e3(p_coords, p_feats, p_mask)
                e3_d = enc_e3(p_coords @ reflect.T, p_feats, p_mask)
                e3_toggle_diff = float((e3_l.pooled_emb - e3_d.pooled_emb).abs().max().item())
            del enc_e3

        is_chiral_arch = arch in SE3_CHIRAL_ARCHS
        is_tiesafe_arch = arch in {
            "dense_painn_se3",
            "cartformer_se3",
            "atomgs_se3",
            "subpocket_vn_jepa",
        }
        passed = (
            overall_rt_err < 2e-4
            and benzene_diff < 1e-4
            and dummy_err < 1e-5
            and (
                (pocket_mirror_diff > 1e-4 and (e3_toggle_diff or 0.0) < 1e-4)
                if is_chiral_arch
                else pocket_mirror_diff < 1e-4
            )
            and (tieswap_err < 1e-4 if is_tiesafe_arch else True)
        )

        scorecard[arch] = {
            "symmetry_class": "SE(3)" if is_chiral_arch else "E(3)",
            "so3_pure_rot_max_err": pure_rot_err,
            "so3_rot_trans100_node_max_err": node_rt_err,
            "so3_rot_trans100_pooled_max_err": pool_rt_err,
            "so3_rot_trans100_vec_max_err": vec_rt_err,
            "so3_rot_trans100_max_err": overall_rt_err,
            "chiral_enantiomer_rs_diff": rs_diff,
            "chiral_pocket_mirror_diff": pocket_mirror_diff,
            "e3_toggle_mirror_diff": e3_toggle_diff,
            "planar_benzene_asym_reflection_diff": benzene_diff,
            "cube_26nbr_tieswap_max_err": tieswap_err,
            "dummy_atom_corrupt_max_err": dummy_err,
            "pass_all_invariants": passed,
        }
        print(
            f"  {arch:18s} | SO3+100A={overall_rt_err:.2e} | "
            f"(R)/(S)={rs_diff:.2e} | pocket_mir={pocket_mirror_diff:.2e} | "
            f"benzene={benzene_diff:.2e} | 26nbr_tie={tieswap_err:.2e} | pass={passed}"
        )
        del enc
        _empty_device_cache(device)

    return scorecard


# ---------------------------------------------------------------------------
# Tier 3: Real lp_pdbbind / CASF-2016 pKd Head-to-Head + JEPA SSL Ablation
# ---------------------------------------------------------------------------


class PocketLigandAffinityModel(nn.Module):
    """Shared pocket-ligand pKd + DrugCLIP evaluation wrapper over any 3D pocket encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        out_dim: int = 96,
        ligand_fp_dim: int = 1024,
        prop_dim: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.out_dim = int(out_dim)
        self.ligand_fp_enc = LigandFingerprintEncoder(
            fp_dim=ligand_fp_dim,
            out_dim=self.out_dim,
            dropout=dropout,
        )
        self.prop_enc = nn.Sequential(
            nn.Linear(prop_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.SiLU(),
        )
        self.ligand_fuse = nn.Sequential(
            nn.Linear(2 * self.out_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.SiLU(),
        )
        self.pocket_norm = nn.LayerNorm(self.out_dim)
        self.pkd_head = nn.Sequential(
            nn.Linear(4 * self.out_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.out_dim, self.out_dim // 2),
            nn.SiLU(),
            nn.Linear(self.out_dim // 2, 1),
        )
        self.clip_pocket_proj = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.clip_ligand_proj = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))

    def forward(
        self,
        coords: Tensor,
        node_feats: Tensor,
        atom_mask: Tensor,
        ligand_fp: Tensor,
        ligand_props: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        enc_out: PocketEncoderOutput = self.encoder(coords, node_feats, atom_mask)
        p_emb = self.pocket_norm(enc_out.pooled_emb)
        l_fp = self.ligand_fp_enc(ligand_fp)
        l_pr = self.prop_enc(ligand_props)
        l_emb = self.ligand_fuse(torch.cat((l_fp, l_pr), dim=-1))

        inter = torch.cat((p_emb, l_emb, p_emb * l_emb, (p_emb - l_emb).abs()), dim=-1)
        pred_pkd = self.pkd_head(inter).squeeze(-1)

        clip_p = F.normalize(self.clip_pocket_proj(p_emb), p=2, dim=-1, eps=1e-8)
        clip_l = F.normalize(self.clip_ligand_proj(l_emb), p=2, dim=-1, eps=1e-8)
        scale = self.logit_scale.clamp(min=-math.log(100.0), max=math.log(100.0))
        return pred_pkd, clip_p, clip_l, scale


def _prepare_split_items(
    ids: list[str],
    records: dict[str, dict[str, Any]],
    prop_mean: np.ndarray,
    prop_std: np.ndarray,
) -> list[dict[str, Any]]:
    """Pre-unpack finite-pKd records into lightweight CPU tensors for fast MPS batching."""
    items: list[dict[str, Any]] = []
    for pid in ids:
        rec = records.get(pid)
        if rec is None:
            continue
        pkd_val = float(rec["pkd"])
        if not math.isfinite(pkd_val):
            continue
        coords = np.asarray(rec["coords"], dtype=np.float32)
        node_feats = np.asarray(rec["node_feats"], dtype=np.float32)
        fp = np.unpackbits(rec["ligand_fp_packed"])[:1024].astype(np.float32)
        props = (np.asarray(rec["ligand_props"], dtype=np.float32) - prop_mean) / prop_std
        items.append(
            {
                "pdb_id": pid,
                "n_atoms": int(coords.shape[0]),
                "coords": torch.from_numpy(coords),
                "node_feats": torch.from_numpy(node_feats),
                "ligand_fp": torch.from_numpy(fp),
                "ligand_props": torch.from_numpy(props.astype(np.float32)),
                "pkd": pkd_val,
            }
        )
    return items


def _build_bucketed_batches(
    items: list[dict[str, Any]],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> list[dict[str, Tensor]]:
    """Collate length-bucketed batches to minimize padding waste without truncating any atom."""
    rng = np.random.default_rng(seed)
    indices = list(range(len(items)))
    if shuffle:
        # Sort by atom count with slight random noise so batches have tight padding & fresh mixes.
        noisy_keys = [(items[i]["n_atoms"] + float(rng.normal(0.0, 20.0)), i) for i in indices]
        noisy_keys.sort(key=lambda pair: pair[0])
        ordered = [i for _, i in noisy_keys]
        chunks = [ordered[k : k + batch_size] for k in range(0, len(ordered), batch_size)]
        rng.shuffle(chunks)
    else:
        ordered = sorted(indices, key=lambda i: items[i]["n_atoms"])
        chunks = [ordered[k : k + batch_size] for k in range(0, len(ordered), batch_size)]

    batches: list[dict[str, Tensor]] = []
    for chunk in chunks:
        bsz = len(chunk)
        max_n = max(items[i]["n_atoms"] for i in chunk)
        coords = torch.zeros((bsz, max_n, 3), dtype=torch.float32)
        feats = torch.zeros((bsz, max_n, 52), dtype=torch.float32)
        mask = torch.zeros((bsz, max_n), dtype=torch.bool)
        fps = torch.zeros((bsz, 1024), dtype=torch.float32)
        props = torch.zeros((bsz, 6), dtype=torch.float32)
        pkds = torch.zeros((bsz,), dtype=torch.float32)
        for bi, idx in enumerate(chunk):
            it = items[idx]
            n = it["n_atoms"]
            coords[bi, :n] = it["coords"]
            feats[bi, :n] = it["node_feats"]
            mask[bi, :n] = True
            fps[bi] = it["ligand_fp"]
            props[bi] = it["ligand_props"]
            pkds[bi] = it["pkd"]
        batches.append(
            {
                "coords": coords,
                "node_feats": feats,
                "atom_mask": mask,
                "ligand_fp": fps,
                "ligand_props": props,
                "pkd": pkds,
            }
        )
    return batches


def _evaluate_pkd_split(
    model: PocketLigandAffinityModel,
    batches: list[dict[str, Tensor]],
    device: torch.device,
    pkd_mean: float,
    pkd_std: float,
) -> dict[str, float]:
    """Evaluate pKd Pearson R, Spearman rho, RMSE, MAE, and DrugCLIP Recall@1/@5."""
    model.eval()
    preds_all: list[np.ndarray] = []
    trues_all: list[np.ndarray] = []
    clip_p_all: list[Tensor] = []
    clip_l_all: list[Tensor] = []

    with torch.no_grad():
        for batch in batches:
            pred_norm, cp, cl, _ = model(
                batch["coords"].to(device),
                batch["node_feats"].to(device),
                batch["atom_mask"].to(device),
                batch["ligand_fp"].to(device),
                batch["ligand_props"].to(device),
            )
            pred_pkd = pred_norm.detach().cpu().numpy() * pkd_std + pkd_mean
            preds_all.append(pred_pkd)
            trues_all.append(batch["pkd"].numpy())
            clip_p_all.append(cp.detach().cpu())
            clip_l_all.append(cl.detach().cpu())

    y_pred = np.concatenate(preds_all, axis=0).astype(np.float64)
    y_true = np.concatenate(trues_all, axis=0).astype(np.float64)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    r_val = float(pearsonr(y_true, y_pred).statistic) if np.std(y_pred) > 1e-12 else 0.0
    rho_val = float(spearmanr(y_true, y_pred).statistic) if np.std(y_pred) > 1e-12 else 0.0

    # Symmetric DrugCLIP Recall@1 and Recall@5
    p_mat = F.normalize(torch.cat(clip_p_all, dim=0), p=2, dim=-1, eps=1e-8)
    l_mat = F.normalize(torch.cat(clip_l_all, dim=0), p=2, dim=-1, eps=1e-8)
    n = p_mat.shape[0]
    sim = p_mat @ l_mat.T
    targets = torch.arange(n, dtype=torch.long)

    def _recall(mat: Tensor, k: int) -> float:
        eff_k = min(k, mat.shape[1])
        topk = torch.topk(mat, k=eff_k, dim=-1).indices
        return float((topk == targets.unsqueeze(1)).any(dim=-1).float().mean().item())

    clip_r1 = 0.5 * (_recall(sim, 1) + _recall(sim.T, 1))
    clip_r5 = 0.5 * (_recall(sim, 5) + _recall(sim.T, 5))

    return {
        "pearson_r": round(r_val, 4),
        "spearman_rho": round(rho_val, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "clip_r1": round(clip_r1, 4),
        "clip_r5": round(clip_r5, 4),
    }


def _train_single_pkd_model(
    arch: str,
    tier3_cfg_kwargs: dict[str, Any],
    *,
    train_items: list[dict[str, Any]],
    val_batches: list[dict[str, Tensor]],
    casf_batches: list[dict[str, Tensor]],
    test_batches: list[dict[str, Tensor]],
    device: torch.device,
    pkd_mean: float,
    pkd_std: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    jepa_ssl_ckpt: str | None = None,
) -> dict[str, Any]:
    """Train one 3D encoder + shared pKd/DrugCLIP head strictly on ``lp_pdbbind.train``."""
    seed_everything(seed)
    cfg = EncoderConfig(arch=arch, jepa_ssl_ckpt=jepa_ssl_ckpt, **tier3_cfg_kwargs)
    encoder = build_encoder(cfg, max_neighbors=24)
    model = PocketLigandAffinityModel(
        encoder,
        out_dim=int(tier3_cfg_kwargs["out_dim"]),
        ligand_fp_dim=1024,
        prop_dim=6,
        dropout=0.2,
    ).to(device)

    params_enc = sum(p.numel() for p in model.encoder.parameters())
    params_total = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = max(1, epochs * math.ceil(len(train_items) / batch_size))
    warmup_steps = max(1, int(0.10 * total_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    best_val_rmse = float("inf")
    best_epoch = 1
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []

    train_t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        train_batches = _build_bucketed_batches(
            train_items, batch_size=batch_size, shuffle=True, seed=seed + epoch * 31
        )
        model.train()
        ep_losses: list[float] = []
        noise_gen = torch.Generator(device="cpu").manual_seed(seed + epoch * 97)

        for batch in train_batches:
            coords = batch["coords"]
            mask = batch["atom_mask"]
            # Mild isotropic 0.10 A coordinate jitter on valid atoms only
            jitter = torch.randn(coords.shape, generator=noise_gen, dtype=torch.float32) * 0.10
            coords_aug = torch.where(mask.unsqueeze(-1), coords + jitter, coords).to(device)
            feats = batch["node_feats"].to(device)
            mask_d = mask.to(device)
            fp = batch["ligand_fp"].to(device)
            props = batch["ligand_props"].to(device)
            pkd_norm = ((batch["pkd"] - pkd_mean) / pkd_std).to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_norm, clip_p, clip_l, scale = model(coords_aug, feats, mask_d, fp, props)
            loss_pkd = F.smooth_l1_loss(pred_norm, pkd_norm, beta=0.5)
            loss_clip = compute_drugclip_loss(clip_p, clip_l, scale)
            loss = loss_pkd + 0.15 * loss_clip
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ep_losses.append(float(loss.detach().item()))

        _sync_device(device)
        ep_s = time.perf_counter() - ep_t0
        val_metrics = _evaluate_pkd_split(model, val_batches, device, pkd_mean, pkd_std)
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(np.mean(ep_losses)), 4),
                "val_pearson_r": val_metrics["pearson_r"],
                "val_rmse": val_metrics["rmse"],
                "epoch_s": round(ep_s, 2),
            }
        )

    _sync_device(device)
    total_wall_s = time.perf_counter() - train_t0

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    val_final = _evaluate_pkd_split(model, val_batches, device, pkd_mean, pkd_std)
    casf_final = _evaluate_pkd_split(model, casf_batches, device, pkd_mean, pkd_std)
    test_final = _evaluate_pkd_split(model, test_batches, device, pkd_mean, pkd_std)

    del model, optimizer, scheduler, best_state
    _empty_device_cache(device)

    return {
        "params_encoder": params_enc,
        "params_total": params_total,
        "train_wall_time_s": round(total_wall_s, 2),
        "sec_per_epoch": round(total_wall_s / float(epochs), 2),
        "best_epoch": best_epoch,
        "val_lp_pdbbind": val_final,
        "casf2016_core": casf_final,
        "test_lp_pdbbind": test_final,
        "epoch_history": history,
    }


def run_tier3_pdbbind_benchmark(
    device: torch.device,
    payload: dict[str, Any],
    splits: dict[str, Any],
    *,
    epochs: int = 5,
    ssl_epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1.5e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Tier 3: Real lp_pdbbind / CASF-2016 pKd head-to-head + Sonata/Concerto JEPA SSL ablation."""
    records: dict[str, dict[str, Any]] = payload["records"]
    prop_mean = np.asarray(payload["prop_mean"], dtype=np.float32)
    prop_std = np.maximum(np.asarray(payload["prop_std"], dtype=np.float32), 1e-6)

    train_ids = [pid for pid in splits["lp_pdbbind"]["train"] if pid in records]
    val_ids = [pid for pid in splits["lp_pdbbind"]["val"] if pid in records]
    test_ids = [pid for pid in splits["lp_pdbbind"]["test"] if pid in records]
    casf_ids = [pid for pid in splits["casf2016_core_ids"] if pid in records]

    # Strict leak-free assertions
    assert len(set(train_ids) & set(val_ids)) == 0, "Leakage: train & val overlap"
    assert len(set(train_ids) & set(test_ids)) == 0, "Leakage: train & test overlap"
    assert len(set(train_ids) & set(casf_ids)) == 0, "Leakage: train & CASF-2016 overlap"
    assert len(set(val_ids) & set(casf_ids)) == 0, "Leakage: val & CASF-2016 overlap"

    train_items = _prepare_split_items(train_ids, records, prop_mean, prop_std)
    val_items = _prepare_split_items(val_ids, records, prop_mean, prop_std)
    test_items = _prepare_split_items(test_ids, records, prop_mean, prop_std)
    casf_items = _prepare_split_items(casf_ids, records, prop_mean, prop_std)

    train_pkds = np.asarray([it["pkd"] for it in train_items], dtype=np.float64)
    pkd_mean = float(np.mean(train_pkds))
    pkd_std = max(float(np.std(train_pkds)), 1e-6)

    val_batches = _build_bucketed_batches(val_items, batch_size=batch_size, shuffle=False, seed=0)
    casf_batches = _build_bucketed_batches(casf_items, batch_size=batch_size, shuffle=False, seed=0)
    test_batches = _build_bucketed_batches(test_items, batch_size=batch_size, shuffle=False, seed=0)

    tier3_cfg_kwargs: dict[str, Any] = {
        "node_in_dim": 52,
        "hidden_dim": 64,
        "vector_dim": 16,
        "out_dim": 96,
        "num_layers": 2,
        "num_rbf": 16,
        "cutoff": 5.5,
        "tensor_dim": 8,
        "num_heads": 4,
        "num_landmarks": 6,
        "num_subpockets": 16,
        "use_se3_chirality": True,
        "use_coulomb_bias": True,
        "enable_3d_overlap": True,
        "dropout": 0.1,
    }

    print(
        f"\n[Tier 3] Real lp_pdbbind / CASF-2016 pKd Benchmark "
        f"(train={len(train_items)}, val={len(val_items)}, "
        f"casf2016={len(casf_items)}, test={len(test_items)}, epochs={epochs})"
    )

    models_out: dict[str, dict[str, Any]] = {}
    for arch in TIER3_ARCHS:
        res = _train_single_pkd_model(
            arch,
            tier3_cfg_kwargs,
            train_items=train_items,
            val_batches=val_batches,
            casf_batches=casf_batches,
            test_batches=test_batches,
            device=device,
            pkd_mean=pkd_mean,
            pkd_std=pkd_std,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
        )
        models_out[arch] = res
        c = res["casf2016_core"]
        v = res["val_lp_pdbbind"]
        print(
            f"  {arch:22s} | wall={res['train_wall_time_s']:5.1f}s "
            f"({res['sec_per_epoch']:4.1f}s/ep) | "
            f"val R={v['pearson_r']:.3f} RMSE={v['rmse']:.3f} | "
            f"CASF-16 R={c['pearson_r']:.3f} rho={c['spearman_rho']:.3f} "
            f"RMSE={c['rmse']:.3f} CLIP@5={c['clip_r5']:.3f}"
        )

    # Sonata / Concerto 3D JEPA SSL Pretraining on real lp_pdbbind.train ONLY (zero leakage!)
    print(
        f"  [SSL] Running Sonata/Concerto 3D JEPA pretraining "
        f"({ssl_epochs} epochs on lp_pdbbind.train)..."
    )
    seed_everything(seed + 1000)
    ssl_enc = build_encoder(
        EncoderConfig(arch="subpocket_vn_jepa", **tier3_cfg_kwargs),
        max_neighbors=24,
    )
    ssl_steps_total = max(1, ssl_epochs * math.ceil(len(train_items) / batch_size))
    pretrainer = SubPocketJEPAPretrainer(
        ssl_enc,
        emb_dim=int(tier3_cfg_kwargs["out_dim"]),
        vector_dim=int(tier3_cfg_kwargs["vector_dim"]),
        predictor_layers=1,
        predictor_heads=4,
        radial_drop_prob=0.80,
        coord_jitter_std=0.20,
        vicreg_weight=0.10,
        cross_modal_dim=1024,
        cross_modal_proj_dim=96,
        cross_modal_weight=0.25,
        ema_decay_start=0.96,
        ema_decay_end=0.995,
        total_steps=ssl_steps_total,
    ).to(device)
    ssl_opt = torch.optim.AdamW(
        [p for p in pretrainer.parameters() if p.requires_grad],
        lr=1.5e-3,
        weight_decay=1e-2,
    )

    ssl_t0 = time.perf_counter()
    ssl_metrics_log: list[dict[str, float]] = []
    step_idx = 0
    for ep in range(1, ssl_epochs + 1):
        ssl_batches = _build_bucketed_batches(
            train_items, batch_size=batch_size, shuffle=True, seed=seed + 500 + ep * 17
        )
        for batch in ssl_batches:
            step_idx += 1
            step_gen = torch.Generator(device="cpu").manual_seed(seed + 10000 + step_idx)
            m = pretrainer.pretrain_step(
                {
                    "coords": batch["coords"].to(device),
                    "node_feats": batch["node_feats"].to(device),
                    "atom_mask": batch["atom_mask"].to(device),
                    "ligand_emb": batch["ligand_fp"].to(device),
                },
                ssl_opt,
                grad_clip=1.0,
                generator=step_gen,
            )
            ssl_metrics_log.append(m)
    _sync_device(device)
    ssl_wall_s = time.perf_counter() - ssl_t0

    with tempfile.TemporaryDirectory(prefix="geneuc_jepa_ssl_") as tmpdir:
        ckpt_path = Path(tmpdir) / "subpocket_vn_jepa_ssl.pt"
        pretrainer.save_checkpoint(
            ckpt_path,
            use_teacher=True,
            metadata={"epochs": ssl_epochs, "steps": step_idx, "split": "lp_pdbbind.train"},
        )
        # Verify jepa_ssl_ckpt loading roundtrip cleanly through build_encoder(...)
        res_ssl = _train_single_pkd_model(
            "subpocket_vn_jepa",
            tier3_cfg_kwargs,
            train_items=train_items,
            val_batches=val_batches,
            casf_batches=casf_batches,
            test_batches=test_batches,
            device=device,
            pkd_mean=pkd_mean,
            pkd_std=pkd_std,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            jepa_ssl_ckpt=str(ckpt_path),
        )

    del pretrainer, ssl_enc, ssl_opt
    _empty_device_cache(device)

    models_out["subpocket_vn_jepa_ssl"] = res_ssl
    c_ssl = res_ssl["casf2016_core"]
    v_ssl = res_ssl["val_lp_pdbbind"]
    print(
        f"  {'subpocket_vn_jepa_ssl':22s} | wall={res_ssl['train_wall_time_s']:5.1f}s "
        f"(+{ssl_wall_s:4.1f}s SSL) | "
        f"val R={v_ssl['pearson_r']:.3f} RMSE={v_ssl['rmse']:.3f} | "
        f"CASF-16 R={c_ssl['pearson_r']:.3f} rho={c_ssl['spearman_rho']:.3f} "
        f"RMSE={c_ssl['rmse']:.3f} CLIP@5={c_ssl['clip_r5']:.3f}"
    )

    painn_wall = models_out["painn"]["train_wall_time_s"]
    for m_entry in models_out.values():
        m_entry["train_speedup_vs_painn"] = round(
            painn_wall / max(m_entry["train_wall_time_s"], 1e-6), 2
        )

    scratch_casf = models_out["subpocket_vn_jepa"]["casf2016_core"]
    scratch_val = models_out["subpocket_vn_jepa"]["val_lp_pdbbind"]
    ssl_summary = {
        "wall_time_s": round(ssl_wall_s, 2),
        "epochs": ssl_epochs,
        "steps": step_idx,
        "initial_loss": round(ssl_metrics_log[0]["loss"], 4),
        "final_loss": round(ssl_metrics_log[-1]["loss"], 4),
        "final_jepa_loss": round(ssl_metrics_log[-1]["jepa_loss"], 4),
        "final_vicreg_loss": round(ssl_metrics_log[-1]["vicreg_loss"], 4),
        "final_concerto_loss": round(ssl_metrics_log[-1]["cross_modal_loss"], 4),
        "final_effective_rank": round(ssl_metrics_log[-1]["effective_rank"], 2),
        "mean_radial_obscured_frac": round(
            float(np.mean([x["radial_obscured_frac"] for x in ssl_metrics_log])), 3
        ),
        "checkpoint_roundtrip_verified": True,
    }
    ablation_delta = {
        "casf2016_delta_pearson_r": round(c_ssl["pearson_r"] - scratch_casf["pearson_r"], 4),
        "casf2016_delta_spearman_rho": round(
            c_ssl["spearman_rho"] - scratch_casf["spearman_rho"], 4
        ),
        "casf2016_delta_rmse": round(c_ssl["rmse"] - scratch_casf["rmse"], 4),
        "casf2016_delta_clip_r5": round(c_ssl["clip_r5"] - scratch_casf["clip_r5"], 4),
        "val_delta_pearson_r": round(v_ssl["pearson_r"] - scratch_val["pearson_r"], 4),
        "val_delta_rmse": round(v_ssl["rmse"] - scratch_val["rmse"], 4),
    }

    return {
        "dataset_summary": {
            "source": "data/processed/pdbbind_pockets.pt",
            "split_type": "lp_pdbbind + casf2016_core",
            "n_train": len(train_items),
            "n_val": len(val_items),
            "n_test_lp_pdbbind": len(test_items),
            "n_casf2016_core": len(casf_items),
            "train_val_overlap": 0,
            "train_casf2016_overlap": 0,
            "val_casf2016_overlap": 0,
            "train_test_overlap": 0,
            "pkd_train_mean": round(pkd_mean, 4),
            "pkd_train_std": round(pkd_std, 4),
        },
        "training_config": {
            **tier3_cfg_kwargs,
            "max_neighbors": 24,
            "epochs": epochs,
            "ssl_epochs": ssl_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "aux_drugclip_weight": 0.15,
        },
        "jepa_ssl_pretraining": ssl_summary,
        "models": models_out,
        "jepa_ssl_ablation_delta": ablation_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 3-tier Apple M3 Pro MPS benchmark for GenEuclidean 3D encoders."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/encoders_v52_mps_benchmark.json"),
        help="Destination JSON path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Target accelerator device.",
    )
    parser.add_argument("--tier3-epochs", type=int, default=5)
    parser.add_argument("--ssl-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    script_t0 = time.perf_counter()
    cold_import = _measure_cold_import()
    print(
        f"[Meta] device={device} | torch={torch.__version__} | "
        f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'unset')} | "
        f"cold_import={cold_import['cold_import_ms']} ms (heavy={cold_import['heavy']})"
    )

    cache_path = Path("data/processed/pdbbind_pockets.pt")
    splits_path = Path("data/processed/splits.json")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = payload["records"]
    real_train_ids = [pid for pid in splits["lp_pdbbind"]["train"] if pid in records]

    gather_bench = run_gather_neighbors_microbench(device)
    print("\n[Micro-bench] dense_knn.gather_neighbors (index_select vs legacy torch.gather):")
    for k, v in gather_bench.items():
        print(
            f"  {k}: old={v['legacy_torch_gather_ms']:.3f} ms -> "
            f"new={v['optimized_index_select_ms']:.3f} ms "
            f"({v['speedup_x']:.1f}x, diff={v['max_abs_diff']})"
        )

    tier1 = run_tier1_throughput_benchmark(device, records=records, real_ids=real_train_ids)
    tier2 = run_tier2_symmetry_scorecard(device)
    tier3 = run_tier3_pdbbind_benchmark(
        device,
        payload,
        splits,
        epochs=int(args.tier3_epochs),
        ssl_epochs=int(args.ssl_epochs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )

    total_wall_s = round(time.perf_counter() - script_t0, 2)
    report: dict[str, Any] = {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "device": str(device),
            "mps_available": bool(torch.backends.mps.is_available()),
            "pytorch_version": torch.__version__,
            "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
            "seed": int(args.seed),
            "total_wall_time_s": total_wall_s,
            **cold_import,
        },
        "gather_neighbors_microbench": gather_bench,
        "tier1_mps_throughput": tier1,
        "tier2_symmetry_chirality_scorecard": tier2,
        "tier3_pdbbind_pkd_and_jepa_ablation": tier3,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[Done] Wrote complete benchmark report to {args.output} in {total_wall_s:.1f}s")


if __name__ == "__main__":
    main()
