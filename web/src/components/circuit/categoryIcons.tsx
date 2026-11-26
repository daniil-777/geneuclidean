import {
  Cpu,
  Database,
  FlaskConical,
  RefreshCw,
  Search,
  Share2,
  ShieldCheck,
  Sparkles,
  type LucideIcon,
} from 'lucide-react';
import type { BlockCategoryId } from '../../types/circuit';

export const CATEGORY_ICONS: Record<BlockCategoryId, LucideIcon> = {
  inputs: Database,
  encoders: Cpu,
  generators: Sparkles,
  screening: Search,
  active_learning: RefreshCw,
  physics_audit: ShieldCheck,
  leadopt_admet: FlaskConical,
  outputs: Share2,
};
