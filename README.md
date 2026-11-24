# GenEuclidean v2.0

**SE(3)/E(3)-Equivariant 3D Protein Pocket-Conditioned De Novo Molecular Generation & Binding Affinity Prediction**

GenEuclidean v2.0 is a clean-room modernization of the 2021 ETH Zürich semester thesis (*De novo drug design using protein structural information and Generative neural network models*), elevating the original E(3)NN + LSTM prototype into a mathematically verified, Apple Silicon MPS / CUDA / CPU-native structure-based drug design (SBDD) framework.

## Key Capabilities

- **True SE(3)/E(3)-Equivariant 3D Pocket Encoders:** PaiNN-SE3 (default, chirality-sensitive via cross-product updates), EGNN, modern `e3nn >= 0.5` tensor-product encoders, and SchNet baselines — strictly `float32`, centered sparse radius graphs, and unit-tested to numerical precision.
- **Two-Stage Pocket-Conditioned Chemical Transformer + CVAE:** Pretrained RoPE + SwiGLU decoder on ZINC-250k with zero-initialized gated cross-attention, TamGen-style pocket-conditioned CVAE (cyclical KL annealing + free-bits), and Classifier-Free Guidance (CFG).
- **Multi-Task Biophysical Supervision:** Joint conditional generation, binding affinity (`pKd`) regression, DrugCLIP-style pocket-ligand contrastive alignment, and auxiliary physicochemical property prediction.
- **Leak-Proof Benchmarking & Historical Continuity:** Preserves the original thesis label tables under `data/metadata/` and archives the 2020/2021 baselines at git tags `v0.1.0-legacy-binding` and `v1.0.0-thesis-2021`. See [docs/legacy_audit.md](docs/legacy_audit.md) for the complete forensic audit and v2.0 mathematical fixes.

## Quickstart

```bash
make install
make lint
make typecheck
make test
```
