| model | validity | uniqueness | novelty | int_div_1 | QED | SA_norm | True Tanimoto | Count Dice | cliffs_delta | attn_enrichment | pKd Pearson r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 Thesis Baseline (e3nn+LSTM) | 0.650 | 0.700 | 0.650 | — | 0.450 | 0.667 | 0.170 | 0.290 | — | — | 0.502 |
| PaiNN-PocketFormer (Pretrained) | 0.921 | 0.985 | 0.988 | 0.896 | 0.487 | 0.738 | 0.098 | 0.195 | -0.007 | 1.160 | — |
| GenEuclidean v2 (SchNet) | 0.900 | 0.974 | 0.985 | 0.904 | 0.481 | 0.720 | 0.125 | 0.253 | 0.248 | 1.499 | 0.722 |
| GenEuclidean v2 (EGNN) | 0.898 | 0.981 | 0.980 | 0.900 | 0.485 | 0.726 | 0.125 | 0.256 | 0.212 | 0.973 | 0.729 |
| GenEuclidean v2 (SELFIES) | 1.000 | 0.983 | 0.992 | 0.911 | 0.422 | 0.670 | 0.107 | 0.232 | 0.184 | 0.622 | 0.731 |
| GenEuclidean v2 (PaiNN-SE3) | 0.893 | 0.966 | 0.983 | 0.906 | 0.464 | 0.705 | 0.159 | 0.319 | 0.486 | 1.597 | 0.746 |
| GenEuclidean v2 (PaiNN-SE3+DPO) | 0.916 | 0.970 | 0.982 | 0.904 | 0.521 | 0.737 | 0.157 | 0.317 | 0.457 | 1.657 | 0.746 |
| GenEuclidean v3 (PaiNN-SE3 + Qwen2.5-0.5B) | 0.432 | 0.810 | 0.943 | 0.880 | 0.433 | 0.685 | 0.189 | 0.332 | 0.582 | 1.079 | 0.746 |
| GenEuclidean v3 (PaiNN-SE3 + GPT2-ZINC-87M) | 0.882 | 0.735 | 0.961 | 0.881 | 0.464 | 0.714 | 0.156 | 0.280 | 0.338 | 3.984 | 0.746 |
| GenEuclidean v3 (GP-MoLFormer Iter1 Rand-SMILES) | 0.709 | 0.953 | 0.992 | 0.919 | 0.479 | 0.712 | 0.145 | 0.303 | 0.496 | 6.661 | 0.746 |
| GenEuclidean v3 (PaiNN-SE3 + GP-MoLFormer-Uniq) | 0.904 | 0.853 | 0.978 | 0.899 | 0.471 | 0.739 | 0.162 | 0.302 | 0.432 | 6.313 | 0.746 |
| GenEuclidean v3 (PaiNN-SE3 + GP-MoLFormer + RPO-DPO) | 0.949 | 0.853 | 0.975 | 0.888 | 0.560 | 0.801 | 0.163 | 0.309 | 0.407 | 6.313 | 0.746 |
