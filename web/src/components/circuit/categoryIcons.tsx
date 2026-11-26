import {
  Database,
  FlaskConical,
  Network,
  Orbit,
  Repeat,
  Search,
  ShieldCheck,
  Sparkles,
  type LucideIcon,
} from 'lucide-react';
import type { BlockCategoryId } from '../../types/circuit';

export const CATEGORY_ICONS: Record<BlockCategoryId, LucideIcon> = {
  inputs: Database,
  encoders: Orbit,
  generators: Sparkles,
  screening: Search,
  active_learning: Repeat,
  physics_audit: ShieldCheck,
  leadopt_admet: FlaskConical,
  outputs: Network,
};
