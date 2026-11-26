"""Strict mathematical E(3)/SE(3) equivariance, invariance, chirality, and registry tests.

Verifies all 8 pluggable 3D pocket encoders (PaiNN-SE3, EGNN, E3NN v0.6, SchNet,
DensePaiNN-SE3, CartFormer-SE3, AtomGS-SE3, SubPocketVN-JEPA) and the complete multi-task
GenEuclideanModel on both CPU and Apple Silicon MPS across:
- Canonical registry & alias resolution (`BUILTIN_ENCODER_ALIASES`)
- 5 Haar-uniform SO(3) rotations (det(R) = +1) + large rigid translations t in [-100, 100]^3
- Random atom permutations
- Extreme dummy-atom coordinate & feature corruption (exact 0.0 sensitivity)
- Improper reflections (det(R) = -1) proving SE(3) chiral sensitivity vs E(3) invariance
"""

from __future__ import annotations

from typing import Literal

import pytest
import torch
from torch import Tensor

from geneuclidean.config import (
    CVAEConfig,
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    ExperimentConfig,
    MultiTaskConfig,
)
from geneuclidean.models import (
    E3NNEncoder,
    GenEuclideanModel,
    PaiNNEncoder,
    build_encoder,
    build_radius_knn_graph,
)
from geneuclidean.models.encoders import (
    ENCODER_REGISTRY,
    AtomGSSE3Encoder,
    CartFormerSE3Encoder,
    DensePaiNNSE3Encoder,
    EGNNEncoder,
    SchNetEncoder,
    SubPocketVNJEPAEncoder,
)
from geneuclidean.registry import BUILTIN_ENCODER_ALIASES, BUILTIN_ENCODER_NAMES

EncoderArch = Literal[
    "painn",
    "egnn",
    "e3nn",
    "schnet",
    "dense_painn_se3",
    "cartformer_se3",
    "atomgs_se3",
    "subpocket_vn_jepa",
]
ALL_ARCHS: list[EncoderArch] = [
    "painn",
    "egnn",
    "e3nn",
    "schnet",
    "dense_painn_se3",
    "cartformer_se3",
    "atomgs_se3",
    "subpocket_vn_jepa",
]
SE3_CHIRAL_ARCHS: list[EncoderArch] = [
    "painn",
    "e3nn",
    "dense_painn_se3",
    "cartformer_se3",
    "atomgs_se3",
    "subpocket_vn_jepa",
]
EXPECTED_ENCODER_CLASSES: dict[EncoderArch, type[torch.nn.Module]] = {
    "painn": PaiNNEncoder,
    "egnn": EGNNEncoder,
    "e3nn": E3NNEncoder,
    "schnet": SchNetEncoder,
    "dense_painn_se3": DensePaiNNSE3Encoder,
    "cartformer_se3": CartFormerSE3Encoder,
    "atomgs_se3": AtomGSSE3Encoder,
    "subpocket_vn_jepa": SubPocketVNJEPAEncoder,
}
DEVICES: list[str] = ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]


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


def _make_pocket_batch(
    batch_size: int = 2,
    num_atoms: int = 32,
    node_in_dim: int = 52,
    device: torch.device | str = "cpu",
    seed: int = 123,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create a realistic chiral 3D pocket batch with variable valid/dummy atom masks."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Realistic 3D coordinates spread over ~6 Angstroms so radius 5.5 A is well connected.
    coords = torch.randn((batch_size, num_atoms, 3), generator=gen, dtype=torch.float32) * 2.2
    node_feats = torch.randn(
        (batch_size, num_atoms, node_in_dim), generator=gen, dtype=torch.float32
    )
    if node_in_dim > 46:
        # Realistic vdW radius prior (col 45) & bounded partial charge prior (col 46).
        node_feats[..., 45] = 0.72 + 0.25 * torch.sigmoid(node_feats[..., 45])
        node_feats[..., 46] = torch.tanh(node_feats[..., 46])
    atom_mask = torch.zeros((batch_size, num_atoms), dtype=torch.bool)
    for b in range(batch_size):
        valid_count = max(8, num_atoms - 5 - 3 * b)
        atom_mask[b, :valid_count] = True

    dev = torch.device(device)
    return coords.to(dev), node_feats.to(dev), atom_mask.to(dev)


def _small_encoder_cfg(arch: EncoderArch, use_se3_chirality: bool = True) -> EncoderConfig:
    return EncoderConfig(
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
        use_se3_chirality=use_se3_chirality,
        dropout=0.0,
    )


def _small_experiment_cfg(
    arch: EncoderArch,
    decoder_arch: Literal["transformer", "lstm"] = "transformer",
) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(max_neighbors=24),
        encoder=_small_encoder_cfg(arch=arch, use_se3_chirality=True),
        cvae=CVAEConfig(enabled=True, latent_dim=16, num_latent_tokens=3),
        decoder=DecoderConfig(
            arch=decoder_arch,
            vocab_size=48,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            max_seq_len=32,
            dropout=0.0,
            use_cross_attn=True,
            gated_cross_attn=True,
            cfg_drop_prob=0.15,
        ),
        multitask=MultiTaskConfig(
            use_pkd_head=True,
            use_clip_head=True,
            use_prop_head=True,
            clip_dim=32,
            ligand_fp_dim=128,
        ),
    )


@pytest.mark.equivariance
def test_encoder_registry_covers_all_canonical_archs_and_aliases() -> None:
    """Every canonical arch and alias resolves via EncoderConfig, ENCODER_REGISTRY, and factory."""
    assert set(ALL_ARCHS) == set(BUILTIN_ENCODER_NAMES)
    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=2, num_atoms=20, device="cpu")

    for arch in ALL_ARCHS:
        assert arch in ENCODER_REGISTRY
        cfg = _small_encoder_cfg(arch)
        assert cfg.arch == arch
        enc = build_encoder(cfg, max_neighbors=24).eval()
        assert isinstance(enc, EXPECTED_ENCODER_CLASSES[arch])
        with torch.no_grad():
            out = enc(coords, node_feats, atom_mask)
        assert out.node_emb.shape == (2, 20, cfg.out_dim)
        assert out.pooled_emb.shape == (2, cfg.out_dim)

    for alias, canonical in BUILTIN_ENCODER_ALIASES.items():
        assert alias in ENCODER_REGISTRY
        assert ENCODER_REGISTRY.resolve_key(alias) == canonical
        cfg_alias = EncoderConfig(
            arch=alias,
            hidden_dim=64,
            vector_dim=16,
            out_dim=64,
            num_layers=2,
            num_rbf=16,
            tensor_dim=12,
            num_heads=4,
            num_landmarks=4,
            num_subpockets=8,
            dropout=0.0,
        )
        assert cfg_alias.arch == canonical
        enc_alias = build_encoder(cfg_alias, max_neighbors=24)
        assert isinstance(enc_alias, EXPECTED_ENCODER_CLASSES[canonical])  # type: ignore[index]


@pytest.mark.equivariance
@pytest.mark.parametrize("arch", ALL_ARCHS)
@pytest.mark.parametrize("device", DEVICES)
def test_all_encoders_so3_rotation_and_translation_equivariance(
    arch: EncoderArch,
    device: str,
) -> None:
    """Verify exact SO(3) invariance (scalars) and equivariance (vectors) over 5 transforms."""
    torch.manual_seed(42)
    dev = torch.device(device)
    encoder = build_encoder(_small_encoder_cfg(arch)).to(dev).eval()
    coords, node_feats, atom_mask = _make_pocket_batch(device=dev, seed=7)

    with torch.no_grad():
        ref = encoder(coords, node_feats, atom_mask)

    # Pure rotation (no t~100 fp32 quantization) is < 1e-5 on CPU; with t in [-100, 100]^3
    # fp32 translation quantization (~1.2e-5 A) * Bessel frequency yields <= 2e-4 on vectors.
    tol_rot_trans = 1e-4
    tol_vec_rot_trans = 2e-4
    tol_pure_rot = 1e-5 if device == "cpu" else 5e-5
    gen = torch.Generator(device="cpu").manual_seed(2026)

    for _ in range(5):
        rot = _random_so3_matrix(gen, dev)
        trans = (torch.rand((1, 1, 3), generator=gen, dtype=torch.float32) * 200.0 - 100.0).to(dev)
        coords_rot = coords @ rot.T
        coords_t = coords_rot + trans

        with torch.no_grad():
            out_rot = encoder(coords_rot, node_feats, atom_mask)
            out = encoder(coords_t, node_feats, atom_mask)

        assert (out_rot.node_emb - ref.node_emb).abs().max().item() < tol_pure_rot
        assert (out_rot.pooled_emb - ref.pooled_emb).abs().max().item() < tol_pure_rot
        node_diff = (out.node_emb - ref.node_emb).abs().max().item()
        pool_diff = (out.pooled_emb - ref.pooled_emb).abs().max().item()
        assert node_diff < tol_rot_trans, f"{arch} ({device}) node_emb SO(3) diff={node_diff:.2e}"
        assert pool_diff < tol_rot_trans, f"{arch} ({device}) pooled_emb SO(3) diff={pool_diff:.2e}"

        if arch == "schnet":
            assert ref.vec_emb is None and out.vec_emb is None
        else:
            assert ref.vec_emb is not None and out.vec_emb is not None
            assert ref.vec_emb.abs().max().item() > 1e-3, f"{arch} vec_emb is trivially zero"
            expected_vec = ref.vec_emb @ rot.T
            assert (out_rot.vec_emb - expected_vec).abs().max().item() < tol_pure_rot
            vec_diff = (out.vec_emb - expected_vec).abs().max().item()
            assert vec_diff < tol_vec_rot_trans, (
                f"{arch} ({device}) vec_emb SO(3) diff={vec_diff:.2e}"
            )


@pytest.mark.equivariance
@pytest.mark.parametrize("lmax", [0, 1, 2])
@pytest.mark.parametrize("device", DEVICES)
def test_e3nn_lmax_variants_and_zero_float64_buffers(lmax: int, device: str) -> None:
    """Verify E3NNEncoder across lmax in {0, 1, 2} has no float64 buffers and is SO(3)-exact."""
    torch.manual_seed(11)
    dev = torch.device(device)
    enc = (
        E3NNEncoder(
            node_in_dim=52,
            hidden_dim=32,
            vector_dim=16,
            out_dim=48,
            num_layers=2,
            num_rbf=12,
            cutoff=5.5,
            e3nn_lmax=lmax,
            use_se3_chirality=True,
            dropout=0.0,
        )
        .to(dev)
        .eval()
    )

    for name, buf in enc.named_buffers():
        assert buf.dtype != torch.float64, f"float64 buffer found in E3NNEncoder: {name}"
    for name, param in enc.named_parameters():
        assert param.dtype != torch.float64, f"float64 param found in E3NNEncoder: {name}"

    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=1, num_atoms=20, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(99)
    rot = _random_so3_matrix(gen, dev)
    trans = torch.tensor([[[42.5, -71.2, 88.9]]], device=dev, dtype=torch.float32)

    with torch.no_grad():
        ref = enc(coords, node_feats, atom_mask)
        out = enc(coords @ rot.T + trans, node_feats, atom_mask)

    tol = 3e-5 if device == "cpu" else 1e-4
    assert (out.node_emb - ref.node_emb).abs().max().item() < tol
    assert (out.pooled_emb - ref.pooled_emb).abs().max().item() < tol
    if lmax == 0:
        assert ref.vec_emb is None and out.vec_emb is None
    else:
        assert ref.vec_emb is not None and out.vec_emb is not None
        assert (out.vec_emb - ref.vec_emb @ rot.T).abs().max().item() < tol


@pytest.mark.equivariance
@pytest.mark.parametrize("arch", ALL_ARCHS)
@pytest.mark.parametrize("device", DEVICES)
def test_all_encoders_permutation_invariance(arch: EncoderArch, device: str) -> None:
    """Verify exact atom permutation invariance/equivariance across all 8 encoders."""
    torch.manual_seed(17)
    dev = torch.device(device)
    encoder = build_encoder(_small_encoder_cfg(arch)).to(dev).eval()
    coords, node_feats, atom_mask = _make_pocket_batch(device=dev, seed=19)

    gen = torch.Generator(device="cpu").manual_seed(303)
    perm = torch.randperm(coords.shape[1], generator=gen).to(dev)

    with torch.no_grad():
        ref = encoder(coords, node_feats, atom_mask)
        out = encoder(coords[:, perm], node_feats[:, perm], atom_mask[:, perm])

    tol = 1e-5 if device == "cpu" else 2e-5
    assert (out.pooled_emb - ref.pooled_emb).abs().max().item() < tol
    assert (out.node_emb - ref.node_emb[:, perm]).abs().max().item() < tol
    if ref.vec_emb is not None and out.vec_emb is not None:
        assert (out.vec_emb - ref.vec_emb[:, perm]).abs().max().item() < tol


@pytest.mark.equivariance
@pytest.mark.parametrize("arch", ALL_ARCHS)
@pytest.mark.parametrize("device", DEVICES)
def test_all_encoders_exact_zero_dummy_atom_sensitivity(arch: EncoderArch, device: str) -> None:
    """Corrupting dummy atom coords to (999,999,999) and features to huge noise gives 0.0 diff."""
    torch.manual_seed(23)
    dev = torch.device(device)
    encoder = build_encoder(_small_encoder_cfg(arch)).to(dev).eval()
    coords, node_feats, atom_mask = _make_pocket_batch(device=dev, seed=29)

    # Verify radius-kNN graph has zero edges touching dummy atoms.
    graph = build_radius_knn_graph(coords, atom_mask, cutoff=5.5, max_neighbors=24)
    mask_flat = atom_mask.reshape(-1)
    assert bool(mask_flat[graph.src].all().item())
    assert bool(mask_flat[graph.dst].all().item())
    assert bool((graph.src != graph.dst).all().item())

    coords_corrupt = coords.clone()
    feats_corrupt = node_feats.clone()
    coords_corrupt[~atom_mask] = 999.0
    feats_corrupt[~atom_mask] = torch.randn_like(feats_corrupt[~atom_mask]) * 1e4

    with torch.no_grad():
        ref = encoder(coords, node_feats, atom_mask)
        out = encoder(coords_corrupt, feats_corrupt, atom_mask)

    # CPU is bit-exact 0.0; MPS atomic index_add_ has ~1e-6 float32 repeat jitter on real atoms.
    assert out.node_emb[~atom_mask].abs().max().item() == 0.0
    if device == "cpu":
        assert (out.node_emb - ref.node_emb).abs().max().item() == 0.0
        assert (out.pooled_emb - ref.pooled_emb).abs().max().item() == 0.0
    else:
        assert (out.node_emb - ref.node_emb).abs().max().item() < 5e-6
        assert (out.pooled_emb - ref.pooled_emb).abs().max().item() < 5e-6
    if ref.vec_emb is not None and out.vec_emb is not None:
        assert out.vec_emb[~atom_mask].abs().max().item() == 0.0
        if device == "cpu":
            assert (out.vec_emb - ref.vec_emb).abs().max().item() == 0.0
        else:
            assert (out.vec_emb - ref.vec_emb).abs().max().item() < 5e-6


@pytest.mark.equivariance
@pytest.mark.parametrize("arch", SE3_CHIRAL_ARCHS)
@pytest.mark.parametrize("device", DEVICES)
def test_all_se3_encoders_chirality_reflection_sensitivity(
    arch: EncoderArch,
    device: str,
) -> None:
    """Every SE(3) encoder distinguishes mirror-image pockets while E(3) toggle stays invariant."""
    torch.manual_seed(31)
    dev = torch.device(device)
    coords, node_feats, atom_mask = _make_pocket_batch(device=dev, seed=37)
    reflect = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=dev, dtype=torch.float32))
    coords_mirror = coords @ reflect.T

    enc_se3 = build_encoder(_small_encoder_cfg(arch, use_se3_chirality=True)).to(dev).eval()
    with torch.no_grad():
        out_l = enc_se3(coords, node_feats, atom_mask)
        out_d = enc_se3(coords_mirror, node_feats, atom_mask)
    min_chiral_diff = 1e-4 if arch == "e3nn" else 1e-3
    assert (out_l.pooled_emb - out_d.pooled_emb).abs().max().item() > min_chiral_diff
    assert (out_l.node_emb - out_d.node_emb).abs().max().item() > min_chiral_diff

    torch.manual_seed(31)
    enc_e3 = build_encoder(_small_encoder_cfg(arch, use_se3_chirality=False)).to(dev).eval()
    with torch.no_grad():
        out_l_e3 = enc_e3(coords, node_feats, atom_mask)
        out_d_e3 = enc_e3(coords_mirror, node_feats, atom_mask)
    tol = 2e-5 if device == "cpu" else 1e-4
    assert (out_l_e3.pooled_emb - out_d_e3.pooled_emb).abs().max().item() < tol
    assert (out_l_e3.node_emb - out_d_e3.node_emb).abs().max().item() < tol


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_painn_and_e3nn_se3_chirality_reflection_sensitivity(device: str) -> None:
    """Verify SE(3) chirality distinguishes mirror-image pockets (det(P)=-1) while E(3) does not."""
    torch.manual_seed(31)
    dev = torch.device(device)
    coords, node_feats, atom_mask = _make_pocket_batch(device=dev, seed=37)
    reflect = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=dev, dtype=torch.float32))
    coords_mirror = coords @ reflect.T

    # 1. PaiNN with use_se3_chirality=True -> reflection-sensitive!
    painn_se3 = (
        PaiNNEncoder(
            node_in_dim=52,
            hidden_dim=64,
            vector_dim=16,
            out_dim=64,
            num_layers=3,
            num_rbf=16,
            cutoff=5.5,
            use_se3_chirality=True,
            dropout=0.0,
        )
        .to(dev)
        .eval()
    )
    with torch.no_grad():
        out_l = painn_se3(coords, node_feats, atom_mask)
        out_d = painn_se3(coords_mirror, node_feats, atom_mask)
    assert (out_l.pooled_emb - out_d.pooled_emb).abs().max().item() > 1e-3
    assert (out_l.node_emb - out_d.node_emb).abs().max().item() > 1e-3

    # 2. PaiNN with use_se3_chirality=False -> strictly E(3)-invariant under reflection!
    painn_e3 = (
        PaiNNEncoder(
            node_in_dim=52,
            hidden_dim=64,
            vector_dim=16,
            out_dim=64,
            num_layers=3,
            num_rbf=16,
            cutoff=5.5,
            use_se3_chirality=False,
            dropout=0.0,
        )
        .to(dev)
        .eval()
    )
    with torch.no_grad():
        out_l_e3 = painn_e3(coords, node_feats, atom_mask)
        out_d_e3 = painn_e3(coords_mirror, node_feats, atom_mask)
    tol = 2e-5 if device == "cpu" else 1e-4
    assert (out_l_e3.pooled_emb - out_d_e3.pooled_emb).abs().max().item() < tol
    assert (out_l_e3.node_emb - out_d_e3.node_emb).abs().max().item() < tol

    # 3. E3NN with use_se3_chirality=True and 3 layers -> 0o pseudoscalars detect chirality!
    e3nn_se3 = (
        E3NNEncoder(
            node_in_dim=52,
            hidden_dim=32,
            vector_dim=16,
            out_dim=48,
            num_layers=3,
            num_rbf=12,
            cutoff=5.5,
            e3nn_lmax=1,
            use_se3_chirality=True,
            dropout=0.0,
        )
        .to(dev)
        .eval()
    )
    with torch.no_grad():
        e3_l = e3nn_se3(coords, node_feats, atom_mask)
        e3_d = e3nn_se3(coords_mirror, node_feats, atom_mask)
    assert (e3_l.pooled_emb - e3_d.pooled_emb).abs().max().item() > 1e-4


@pytest.mark.equivariance
@pytest.mark.parametrize("arch", ALL_ARCHS)
@pytest.mark.parametrize("device", DEVICES)
def test_full_geneuclidean_model_so3_invariance(arch: EncoderArch, device: str) -> None:
    """Verify full GenEuclideanModel (logits, pKd, DrugCLIP, props, CVAE) is SO(3)-invariant."""
    torch.manual_seed(41)
    dev = torch.device(device)
    cfg = _small_experiment_cfg(arch=arch, decoder_arch="transformer")
    model = GenEuclideanModel(cfg).to(dev).eval()

    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=2, num_atoms=28, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(55)
    input_ids = torch.randint(1, 48, (2, 14), generator=gen).to(dev)
    ligand_fp = (torch.rand((2, 128), generator=gen) > 0.7).to(device=dev, dtype=torch.float32)

    base_batch = {
        "coords": coords,
        "node_feats": node_feats,
        "atom_mask": atom_mask,
        "input_ids": input_ids,
        "ligand_fp": ligand_fp,
    }
    with torch.no_grad():
        ref = model(base_batch, cfg_drop=False, return_attn=True)
        # Sanity check: perturbing pocket geometry non-rigidly changes outputs (non-vacuous test).
        perturbed_batch = dict(base_batch)
        pert_noise = torch.randn(coords.shape, generator=gen, dtype=torch.float32).to(dev) * 2.0
        perturbed_batch["coords"] = coords + pert_noise
        pert_out = model(perturbed_batch, cfg_drop=False)
        assert (pert_out.logits - ref.logits).abs().max().item() > 5e-5
        assert ref.pred_pkd is not None and pert_out.pred_pkd is not None
        assert (pert_out.pred_pkd - ref.pred_pkd).abs().max().item() > 2e-4
        assert ref.clip_pocket is not None and pert_out.clip_pocket is not None
        assert (pert_out.clip_pocket - ref.clip_pocket).abs().max().item() > 1e-3

    tol = 1e-4
    for _ in range(5):
        rot = _random_so3_matrix(gen, dev)
        trans = (torch.rand((1, 1, 3), generator=gen, dtype=torch.float32) * 200.0 - 100.0).to(dev)
        rot_batch = dict(base_batch)
        rot_batch["coords"] = coords @ rot.T + trans

        with torch.no_grad():
            out = model(rot_batch, cfg_drop=False, return_attn=True)

        assert (out.logits - ref.logits).abs().max().item() < tol
        assert ref.pred_pkd is not None and out.pred_pkd is not None
        assert (out.pred_pkd - ref.pred_pkd).abs().max().item() < tol
        assert ref.pred_props is not None and out.pred_props is not None
        assert (out.pred_props - ref.pred_props).abs().max().item() < tol
        assert ref.clip_pocket is not None and out.clip_pocket is not None
        assert (out.clip_pocket - ref.clip_pocket).abs().max().item() < tol
        assert ref.mu_p is not None and out.mu_p is not None
        assert (out.mu_p - ref.mu_p).abs().max().item() < tol


@pytest.mark.equivariance
@pytest.mark.parametrize("device", DEVICES)
def test_e3nn_lstm_v15_so3_invariance(device: str) -> None:
    """Verify GenEuclidean-v1.5 (E3NN + Attention-LSTM) is strictly SO(3)-invariant."""
    torch.manual_seed(77)
    dev = torch.device(device)
    cfg = _small_experiment_cfg(arch="e3nn", decoder_arch="lstm")
    cfg.cvae.enabled = False
    model = GenEuclideanModel(cfg).to(dev).eval()

    coords, node_feats, atom_mask = _make_pocket_batch(batch_size=2, num_atoms=24, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(88)
    input_ids = torch.randint(1, 48, (2, 12), generator=gen).to(dev)
    batch = {
        "coords": coords,
        "node_feats": node_feats,
        "atom_mask": atom_mask,
        "input_ids": input_ids,
    }
    rot = _random_so3_matrix(gen, dev)
    trans = torch.tensor([[[-65.0, 91.5, -18.2]]], device=dev, dtype=torch.float32)

    with torch.no_grad():
        ref = model(batch, return_attn=True)
        rot_batch = dict(batch)
        rot_batch["coords"] = coords @ rot.T + trans
        out = model(rot_batch, return_attn=True)

    tol = 5e-5 if device == "cpu" else 1e-4
    assert (out.logits - ref.logits).abs().max().item() < tol
