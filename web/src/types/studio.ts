export type StudioTab =
  | 'overview'
  | 'studio3d'
  | 'generator'
  | 'experiments'
  | 'disorders'
  | 'figures';

export type ThemeMode = 'dark' | 'light';

export interface HeadlineKPIs {
  cns_validity_pct: number;
  cns_strict_novelty_pct: number;
  cns_int_div_1: number;
  cns_mpo_v2: number;
  joint_cns_safe_pct: number;
  posebusters_pass_pct: number;
  joint_phys_safe_core10_pct: number;
  joint_phys_safe_all12_pct: number;
  gate1_core10_rmsd_median: number;
  gate1_anchor_recovery_pct: number;
  cross_target_col_z_p: number;
  cross_target_diag_above_mean: number;
  drd1_lead_vina: number;
  drd1_lead_ddg: number | null;
  drd1_lead_mpo: number;
  drd1_lead_herg: number;
  attn_contact_enrichment: number;
  casf_pkd_r: number;
  casf_drugclip_r5_pct: number;
  total_vina_dockings: number;
  n_targets: number;
}

export interface EvolutionStage {
  version: string;
  tag: string;
  year: string;
  title: string;
  architecture: string;
  highlights: string[];
  status: 'archived' | 'verified' | 'active' | 'ready' | 'live';
}

export interface ModelComparisonRow {
  model: string;
  run_id: string;
  num_params: number | null;
  validity: number | null;
  uniqueness: number | null;
  novelty: number | null;
  int_div_1: number | null;
  murcko_div: number | null;
  qed: number | null;
  sa: number | null;
  sa_norm: number | null;
  lipinski: number | null;
  pains_free: number | null;
  true_tanimoto: number | null;
  count_dice: number | null;
  shuffled_tanimoto: number | null;
  cliffs_delta: number | null;
  attn_enrichment: number | null;
  pkd_r: number | null;
  pkd_rmse: number | null;
  drugclip_r1: number | null;
  drugclip_auc: number | null;
}

export interface CasfDockingRow {
  label: string;
  file: string;
  n_pockets: number;
  n_ligands_docked: number;
  top1_vina_mean: number | null;
  top1_vina_median: number | null;
  vina_mean: number | null;
  posebusters_pass_rate: number | null;
  zero_clash_rate: number | null;
  strain_mean: number | null;
  high_affinity_pocket_rate: number | null;
}

export interface FigureItem {
  id: string;
  file: string;
  category: string;
  title: string;
  caption: string;
  size_kb: number;
}

export interface V4Discovery {
  detected: boolean;
  files: string[];
  active_version: string;
  roadmap_targets: {
    target: string;
    disorder: string;
    mechanism: string;
    status: string;
  }[];
}

export interface OverviewData {
  generated_at: string;
  version: string;
  repo_url: string;
  disclaimer: string;
  v4_discovery: V4Discovery;
  headline_kpis: HeadlineKPIs;
  evolution: EvolutionStage[];
  model_comparison_casf: ModelComparisonRow[];
  casf_docking: CasfDockingRow[];
  figures: FigureItem[];
}

export interface MoleculeIndexEntry {
  target: string;
  lead_id: string;
  is_curated_rank2: boolean;
  title: string;
  family: string;
  pdb_id: string;
  disease_tag: string;
  disorders: string[];
  canonical_smiles: string;
  formula: string;
  mw: number;
  vina_dg: number;
  native_vina_dg: number | null;
  le: number;
  ddg_vs_drd2_mean: number | null;
  cns_mpo_v2: number;
  bbb_prob: number;
  herg_prob: number;
  pgp_inhib_prob: number;
  posebusters_passed: boolean;
  anchor_label: string;
  anchor_min_dist: number | null;
  max_chembl_tan: number;
  n_contacts: number;
  svg_mini_dark: string;
  svg_mini_light: string;
}

export interface MoleculesIndexData {
  generated_at: string;
  version: string;
  count: number;
  disclaimer: string;
  molecules: MoleculeIndexEntry[];
}

export type InteractionKind =
  | 'salt_bridge'
  | 'hbond'
  | 'pi_stacking'
  | 'cation_pi'
  | 'hydrophobic'
  | 'halogen_bond';

export interface InteractionRecord {
  kind: InteractionKind;
  protein_residue: string;
  protein_atom: string;
  ligand_element: string;
  distance_angstrom: number;
  angle_deg: number | null;
  protein_xyz: [number, number, number];
  ligand_xyz: [number, number, number];
  subtype?: string;
}

export interface MPOComponent {
  key: string;
  label: string;
  value: number;
  unit: string;
  t0: number;
  weight: number;
  ideal: string;
}

export interface NearestChembl {
  chembl_id: string;
  target: string;
  best_pchembl: number;
  canonical_smiles: string;
}

export interface ComplexDetail {
  target: string;
  lead_id: string;
  is_curated_rank2: boolean;
  title: string;
  family: string;
  pdb_id: string;
  disease_tag: string;
  disorders: string[];
  mechanism: string;
  clinical_precedent: string;
  triage_note: string;
  canonical_smiles: string;
  smiles_ph74: string;
  murcko_scaffold: string;
  svg_2d_dark: string;
  svg_2d_light: string;
  descriptors: {
    formula: string;
    mw: number;
    clogp: number;
    clogd_74: number;
    tpsa: number;
    hbd: number;
    hba: number;
    rot_bonds: number;
    heavy_atoms: number;
    rings: number;
    aromatic_rings: number;
    most_basic_pka: number;
    qed: number;
    sa_score: number;
    pains_free: boolean;
    brenk_free: boolean;
    lipinski_pass: boolean;
  };
  cns_mpo_components: MPOComponent[];
  admet: {
    cns_mpo_v1: number;
    cns_mpo_v2: number;
    bbb_prob: number;
    pgp_inhib_prob: number;
    pgp_minus_prob: number;
    herg_prob: number;
    min_ad_sim: number;
    in_domain_all: boolean;
    joint_cns_safe: boolean;
    pred_pkd_painn: number | null;
    qsar_p_on_target: number | null;
    qsar_delta_sel: number | null;
  };
  physics_3d: {
    top1_vina_dg: number;
    native_vina_dg: number | null;
    delta_vs_native: number | null;
    le: number;
    lle_vina: number;
    anchor_label: string;
    anchor_min_dist: number | null;
    anchor_mode_rank: number | null;
    anchor_engaged_top1: boolean;
    posebusters_passed: boolean;
    protein_clashes: number;
    mmff_strain_kcal: number;
    plif_tanimoto: number;
    residue_recovery: number;
    ddg_vs_drd2_6cm4: number | null;
    ddg_vs_drd2_6luq: number | null;
    ddg_vs_drd2_mean: number | null;
  };
  novelty: {
    is_strict_novel: boolean;
    max_tanimoto_all_chembl: number;
    nearest_all_chembl: NearestChembl | null;
    max_tanimoto_same_target: number;
    nearest_same_target: NearestChembl | null;
  };
  attention_stats: {
    contact_enrichment: number | null;
    hotspot_enrichment: number | null;
    recall_at_10: number | null;
    top5_residues: string[];
  };
  attention: Record<string, number>;
  profile: {
    interactions: InteractionRecord[];
    counts_by_type: Record<string, number>;
    key_residues: string[];
  };
  pdb: string;
  ligand_sdf: string;
  native_sdf: string;
  notes: string;
}

export interface Gate1Complex {
  complex_id: string;
  target: string;
  pdb_id: string;
  drug_name: string;
  tier: string;
  resolution_angstrom: number | null;
  anchor_label: string;
  top1_rmsd_median: number;
  top1_rmsd_min: number;
  top1_rmsd_max: number;
  best3_rmsd_median: number;
  best3_anchor_pct: number;
  top1_vina_dg: number;
  posebusters_pct: number;
}

export interface ClinicalDrugPoint {
  name: string;
  group: 'CNS' | 'QT_CNS' | 'Peripheral';
  target_id: string;
  smiles: string;
  bbb_prob: number;
  herg_prob: number;
  pgp_prob: number;
  cns_mpo_v2: number;
  mw: number;
  clogp: number;
  tpsa: number;
  hbd: number;
}

export interface ClosedLoopArm {
  key: string;
  label: string;
  short_label: string;
  validity_pct: number;
  strict_novelty_pct: number;
  int_div_1: number;
  basic_amine_pct: number | null;
  cns_mpo_v2: number;
  bbb_pct: number;
  herg_pct: number;
  pgp_pct: number;
  joint_cns_safe_pct: number;
  min_ad_sim: number;
  test_scaffold_snn: number | null;
  vina_dg_core10: number | null;
  le_core10: number | null;
  lle_vina_core10: number | null;
  posebusters_core10_pct: number | null;
  anchor_top1_core10_pct: number | null;
  joint_phys_safe_core10_pct: number | null;
  joint_phys_safe_all12_pct: number | null;
}

export interface Drd1ScatterPoint {
  arm: string;
  label: string;
  raw_label: string;
  featured: boolean;
  smiles: string;
  dg_drd1: number;
  dg_drd2_6cm4: number;
  dg_drd2_6luq: number;
  ddg_d1_vs_d2_mean: number;
  le_drd1: number;
}

export interface Drd1ClinicalControl {
  label: string;
  short_name: string;
  smiles: string;
  dg_drd1: number;
  dg_drd2_6cm4: number;
  dg_drd2_6luq: number;
  ddg_d1_vs_d2_mean: number;
  le_drd1: number;
  is_d1_selective: boolean;
}

export interface ExperimentsData {
  generated_at: string;
  gate1: {
    cohort_summaries: Record<string, Record<string, number | string>>;
    acceptance_checks: Record<string, boolean>;
    complexes: Gate1Complex[];
  };
  gate2: {
    admet: {
      key: string;
      label: string;
      n_samples: number;
      roc_auc: number;
      roc_auc_std: number;
      pr_auc: number;
      ece_raw: number;
      ece_cal: number;
      brier: number;
    }[];
    qsar: {
      target: string;
      n_samples: number;
      roc_auc: number;
      roc_auc_std: number;
      ece: number;
    }[];
    qsar_mean_roc_auc: number;
    herg_external_wang_auc: number | null;
  };
  clinical_drugs: ClinicalDrugPoint[];
  closed_loop: {
    arms: ClosedLoopArm[];
    ref_clinical: {
      key: string;
      label: string;
      short_label: string;
      basic_amine_pct: number | null;
      cns_mpo_v2: number;
      bbb_pct: number;
      herg_pct: number;
      pgp_pct: number;
      joint_cns_safe_pct: number;
    };
    statistics_all12: Record<
      string,
      {
        joint_cns_safe_diff_pp: number;
        joint_cns_safe_wilcoxon_p: number;
        joint_cns_safe_cliffs_delta: number;
        joint_cns_safe_holm_n: number;
        herg_wilcoxon_p: number;
        herg_cliffs_delta: number;
      }
    >;
    casf2016_retention: {
      n_core_complexes: number;
      pkd_pearson_r: number;
      pkd_rmse: number;
      drugclip_recall_at_5: number;
      drugclip_roc_auc: number;
      gen_validity: number;
    };
  };
  ablations: {
    a3_ad_damping: {
      safe_drop_pp: number;
      metrics: {
        metric: string;
        iter2: number;
        a3_noad: number;
        wilcoxon_p: number;
        display_iter2: string;
        display_noad: string;
      }[];
    };
    a5_cfg_pareto: {
      cfg_scale: number;
      drd1_delta_sel: number;
      joint_cns_safe_pct: number;
      cns_mpo_v2: number;
      int_div_1: number;
      validity_pct: number;
    }[];
  };
  per_target_specificity: {
    targets: {
      target: string;
      fold_enrichment: number;
      matched_p_on_target: number;
      shuffled_p_on_target: number;
      contact_enrichment: number;
      hotspot_enrichment: number;
      recall_at_10_pct: number;
      top_residue: string;
    }[];
    mean_contact_enrichment: number;
    mean_hotspot_enrichment: number;
    mean_anchor_enrichment: number;
    mean_recall_at_10_pct: number;
    matched_vs_shuffled_p: number;
  };
  iter3_physics: {
    protocol: Record<string, unknown>;
    core10: Record<string, Record<string, number>>;
    all12: Record<string, Record<string, number>>;
    stats_core10: Record<string, unknown>;
  };
  cross_target_matrix: {
    targets: string[];
    top_k_per_target: number;
    n_permutations: number;
    mean_vina_dg_matrix: number[][];
    col_zscore_selectivity_matrix: number[][];
    double_centered_z_matrix: number[][];
    mean_le_matrix: number[][];
    mean_plif_tanimoto_matrix: number[][];
    permutation_tests: Record<
      string,
      {
        diagonal_mean: number;
        off_diagonal_mean: number;
        enrichment_margin: number;
        permutation_p_10k: number;
        n_targets_above_pocket_mean: number;
      }
    >;
    per_target_diagonal: {
      target: string;
      diag_vina_dg: number;
      col_mean_vina_dg: number;
      col_zscore: number;
      double_centered_z: number;
      double_centered_ddg_kcal: number;
      diag_plif_tanimoto: number;
      col_rank_of_10: number;
    }[];
  };
  drd1_vs_drd2: {
    native_baselines: Record<string, number>;
    arm_summaries: Record<string, Record<string, number>>;
    clinical_controls: Drd1ClinicalControl[];
    points: Drd1ScatterPoint[];
  };
}
