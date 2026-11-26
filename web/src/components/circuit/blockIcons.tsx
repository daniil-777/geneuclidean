import {
  Activity,
  Atom,
  BarChart3,
  Beaker,
  Bot,
  Box,
  Compass,
  Cpu,
  Database,
  Dna,
  Eye,
  FileCheck2,
  Filter,
  FlaskConical,
  Gauge,
  GitBranch,
  GitCompareArrows,
  Layers,
  Magnet,
  Network,
  Orbit,
  Radar,
  Repeat,
  Scale,
  Search,
  ShieldAlert,
  ShieldCheck,
  Shuffle,
  Sliders,
  Sparkles,
  Target,
  Waypoints,
  Workflow,
  Zap,
  type LucideIcon,
} from 'lucide-react';
import type { BlockCategoryId, SignalType } from '../../types/circuit';
import type { ThemeMode } from '../../types/studio';
import { CATEGORY_META_MAP, SIGNAL_TYPE_META } from '../../lib/circuitCatalog';

export const CATEGORY_ICON_MAP: Record<BlockCategoryId, LucideIcon> = {
  inputs: Database,
  encoders: Orbit,
  generators: Sparkles,
  screening: Search,
  active_learning: Repeat,
  physics_audit: ShieldCheck,
  leadopt_admet: FlaskConical,
  outputs: Network,
};

export const BLOCK_ICON_MAP: Record<string, LucideIcon> = {
  // Inputs
  protein_pocket: Atom,
  multiconformer_ensemble: Layers,
  reagent_library: Beaker,
  boltz2_cofold: Dna,
  chembl_pool: Database,
  // Encoders
  painn_se3: Orbit,
  egnn_3d: Box,
  e3nn_irreps: Compass,
  drugclip_encoder: Magnet,
  ibm_mmelon: Cpu,
  // Generators
  safe_gpt: Sparkles,
  chemical_llm: Zap,
  pocket_ensemble_cfg: Sliders,
  selectivity_mtl_aligner: GitCompareArrows,
  // Screening
  ligand_vector_index: Radar,
  pocket_screener: Search,
  neural_pkd_reranker: BarChart3,
  catalog_grounding: Target,
  // Active Learning
  tanimoto_gp: Activity,
  ensemble_surrogate: GitBranch,
  conformal_calibrator: Gauge,
  batch_acquisition: Filter,
  multifidelity_router: Waypoints,
  combinatorial_ts: Shuffle,
  // Physics Audit
  protonation_ph74: Zap,
  autodock_vina: Atom,
  ring_torsion_strain: ShieldAlert,
  pose_reality_auditor: ShieldCheck,
  // Lead-Opt & ADMET
  reaction_enumerator: FlaskConical,
  leadopt_mmp_sar: Workflow,
  cns_admet_suite: ShieldCheck,
  conformal_dmpk: Scale,
  // Outputs
  openfe_rbfe_exporter: Network,
  studio_3d_inspector: Eye,
  mcp_tool_export: Bot,
  dmta_ledger: FileCheck2,
};

export function getBlockIcon(blockType: string): LucideIcon {
  return BLOCK_ICON_MAP[blockType] ?? Cpu;
}

export function getSignalColor(signalType: SignalType, theme: ThemeMode): string {
  const meta = SIGNAL_TYPE_META[signalType];
  if (!meta) return theme === 'dark' ? '#38BDF8' : '#0284C7';
  return theme === 'dark' ? meta.darkHex : meta.lightHex;
}

export function getCategoryColor(category: BlockCategoryId, theme: ThemeMode): string {
  const meta = CATEGORY_META_MAP[category];
  if (!meta) return theme === 'dark' ? '#38BDF8' : '#0284C7';
  return theme === 'dark' ? meta.darkHex : meta.lightHex;
}
