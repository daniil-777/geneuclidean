// Native rigid-body Langevin binding dynamics for geneuclidean.biophysics.
//
// WHY THIS EXISTS
// ---------------
// The PyTorch Hamiltonian costs 77.6 ms per energy+gradient call on a 40-atom ligand in a
// 962-atom pocket (measured, 6 threads). Almost none of that is arithmetic: the
// ligand-pocket cross block is only 38,480 pairs, and even with Generalized Born disabled
// an energy-only call still takes 20.8 ms. The cost is kernel-launch and allocation
// overhead across dozens of elementwise ops on small tensors, plus a Generalized Born
// stage that recomputes 1,004,004 pairs every step.
//
// Two facts make a native port dramatically cheaper rather than merely somewhat cheaper:
//
//   1. The receptor is FROZEN and the ligand is RIGID, so everything that depends only on
//      internal geometry is constant. The descreening integral splits exactly as
//      I_i = I_i^intra + I_i^cross, and only the cross block moves. Likewise the apo Born
//      radii, the apo Still f_GB matrix, the apo SASA, the donor axes, the halogen
//      sigma-hole axes and the ring frames are all constants (the ligand's merely rotate).
//   2. The apolar term collapses algebraically. Since
//      sasa_comp_i = max_area_i * w_i * exp(-(occ_intra_i + occ_cross_i))
//                  = sasa_apo_i * exp(-occ_cross_i),
//      the buried area is sasa_apo_i * (1 - exp(-occ_cross_i)), which is non-negative, so
//      the reference implementation's clamp(min=0) is never active and never needs a
//      branch here.
//
// So one energy evaluation is the 38,480-pair cross block plus O(N) closings, plus a
// restricted receptor desolvation sum over the pairs whose Born radius actually changed.
//
// FORCES BY FINITE DIFFERENCE, DELIBERATELY
// -----------------------------------------
// A rigid body has six degrees of freedom, so the generalized forces are six numbers and
// central differences need twelve energy evaluations. At double precision with h ~ 1e-5
// the truncation error is ~1e-10 relative -- far below the float32 noise of the reference
// implementation. The alternative, hand-differentiating the two-stage Generalized Born
// chain rule, the coordinate-derived donor axes, the Newell ring normals and the Gaussian
// SASA, is thousands of lines of derivative code in which a single wrong term would
// silently destroy the Boltzmann stationary distribution while still looking like a
// plausible animation. Finite differences instead *guarantee* that the force is the
// gradient of the energy actually being reported, which is the one property a sampler
// cannot do without.
//
// Caveat, stated rather than hidden: the reference potential has C^0 kinks (the H-bond
// 12-10 well is clamped to [-1, 2], the angular gate is a relu, the switching envelopes
// clamp at the endpoints). Differencing across a kink returns an averaged slope. Autograd
// returns an arbitrary one-sided slope at the same points, so this is not a regression --
// but it is why the acceptance tests check energy conservation rather than trusting either.
//
// UNITS
// -----
// Angstrom, picosecond, amu, kcal/mol, Kelvin, radians.
// a[A/ps^2] = 418.4 * F[kcal/mol/A] / m[amu].

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#define GEX_API extern "C" __attribute__((visibility("default")))

namespace {

constexpr double kCoulomb = 332.06371;         // kcal * A / (mol * e^2)
constexpr double kProbeRadiusWater = 1.40;     // A
constexpr double kMdAccel = 418.4;             // amu*A^2/(ps^2 * kcal/mol)
constexpr double kBoltzmann = 0.001987204259;  // kcal/(mol*K)

// ---------------------------------------------------------------------------------------
// Reference-exact scalar kernels. Each mirrors one expression in forcefield.py; the epsilon
// guards are reproduced verbatim because they are inside the function being matched, not
// decoration -- dropping them changes the energy in the last digits and the acceptance
// test compares against the Python result.
// ---------------------------------------------------------------------------------------

// _smooth_switch_c2: CHARMM/OpenMM C^2 potential switch.
inline double SmoothSwitchC2(double dist, double r_on, double r_off) {
  if (dist <= r_on) return 1.0;
  const double r2 = dist * dist;
  const double on2 = r_on * r_on;
  const double off2 = r_off * r_off;
  double denom = (off2 - on2);
  denom = denom * denom * denom;
  if (denom < 1e-8) denom = 1e-8;
  double base = off2 - r2;
  if (base < 0.0) base = 0.0;
  double num = base * base * (off2 + 2.0 * r2 - 3.0 * on2);
  double s = num / denom;
  if (s < 0.0) s = 0.0;
  if (s > 1.0) s = 1.0;
  return s;
}

// mehler_solmajer_dielectric with the softplus(beta=2.5) floor.
inline double MehlerSolmajer(double dist, double min_eps) {
  constexpr double a_const = -8.5525;
  constexpr double b_const = 78.5 - a_const;  // 87.0525
  constexpr double kappa = 7.7839;
  constexpr double lam = 0.003627;
  const double raw = a_const + b_const / (1.0 + kappa * std::exp(-lam * b_const * dist));
  const double x = raw - min_eps;
  // softplus(x, beta) = log1p(exp(beta*x))/beta, with the same large-argument shortcut
  // torch takes (threshold 20 on beta*x).
  const double bx = 2.5 * x;
  const double sp = (bx > 20.0) ? x : std::log1p(std::exp(bx)) / 2.5;
  return min_eps + sp;
}

// One ordered-pair contribution to the HCT descreening integral, as in
// compute_effective_born_radii. `rho_i` is the offset radius of the atom being descreened
// and `sr_j` the scaled offset radius of the descreener.
inline double HctPairTerm(double dist, double rho_i, double sr_j) {
  const double upper = dist + sr_j;
  if (!(rho_i < upper)) return 0.0;  // j too far to descreen i at all
  double lower = std::fabs(dist - sr_j);
  if (rho_i > lower) lower = rho_i;
  const double inv_l = 1.0 / (lower < 1e-6 ? 1e-6 : lower);
  const double inv_u = 1.0 / (upper < 1e-6 ? 1e-6 : upper);
  double bracket = inv_l - inv_u +
                   0.25 * (dist - sr_j * sr_j / dist) * (inv_u * inv_u - inv_l * inv_l) +
                   0.5 * std::log(lower / upper) / dist;
  // Engulfment: atom i lies entirely inside j's descreening sphere. The 0.5 prefactor
  // multiplies this addend too -- see the comment in forcefield.py, where putting the
  // 2.0*(...) outside the 0.5 was a latent factor-2 bug.
  if (rho_i < (sr_j - dist)) bracket += 2.0 * (1.0 / rho_i - inv_l);
  return 0.5 * bracket;
}

// OBC2 closing: integral -> effective Born radius.
inline double Obc2Close(double integral, double rho, double rho_t, double r_max) {
  if (integral < 0.0) integral = 0.0;
  const double psi = integral * rho_t;
  const double arg = psi * (1.0 - 0.8 * psi + 4.85 * psi * psi);
  double inv = (1.0 / rho_t) - std::tanh(arg) / rho;
  const double floor_inv = 1.0 / r_max;
  if (inv < floor_inv) inv = floor_inv;
  return 1.0 / inv;
}

// Still's f_GB. `sq_d` already includes any soft_r2.
inline double StillFgb(double sq_d, double ri, double rj) {
  const double ri_rj = ri * rj;
  return std::sqrt(sq_d + ri_rj * std::exp(-sq_d / (4.0 * ri_rj + 1e-8)) + 1e-8);
}

struct Vec3 {
  double x, y, z;
};

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline double Dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// ---------------------------------------------------------------------------------------
// Engine: owns the rigid constants and the caches that rigidity makes valid.
// ---------------------------------------------------------------------------------------

// Per-thread working storage. Energy() must be reentrant because the twelve
// finite-difference evaluations of one force call run concurrently, so none of this can
// live on the Engine as shared mutable state.
struct Scratch {
  std::vector<double> pose, integ_l, integ_r, occ_l, occ_r, born_l, born_r;
  std::vector<double> daxes, haxes, lringc, lringn;
  std::vector<uint8_t> changed;
  std::vector<int> changed_list, unchanged_list;
  std::vector<double> cross_integ;
  int n_changed_last = 0;

  void Resize(int nl, int np, int n_lring) {
    changed_list.resize(np);
    unchanged_list.resize(np);
    cross_integ.resize(np);
    pose.resize(3u * nl);
    integ_l.resize(nl);
    occ_l.resize(nl);
    born_l.resize(nl);
    integ_r.resize(np);
    occ_r.resize(np);
    born_r.resize(np);
    changed.resize(np);
    daxes.resize(3u * nl);
    haxes.resize(3u * nl);
    lringc.resize(3u * static_cast<size_t>(n_lring > 0 ? n_lring : 1));
    lringn.resize(3u * static_cast<size_t>(n_lring > 0 ? n_lring : 1));
  }
};

struct Engine {
  int nl = 0, np = 0, n_lring = 0, n_rring = 0;

  // Ligand, body frame (relative to the ligand centre of mass), constant.
  std::vector<double> lig_body;        // [3*nl]
  std::vector<double> lig_daxes_body;  // [3*nl]
  std::vector<double> lig_haxes_body;  // [3*nl]
  std::vector<double> lig_ringc_body;  // [3*n_lring]
  std::vector<double> lig_ringn_body;  // [3*n_lring]
  std::vector<double> lig_q, lig_sig, lig_eps, lig_rho, lig_obc, lig_hstr, lig_r0x;
  std::vector<uint8_t> lig_don, lig_acc, lig_cat, lig_ani, lig_hyd, lig_hal;

  // Receptor, lab frame, constant.
  std::vector<double> rec_xyz, rec_daxes, rec_ringc, rec_ringn;
  std::vector<double> rec_q, rec_sig, rec_eps, rec_rho, rec_obc;
  std::vector<uint8_t> rec_don, rec_acc, rec_cat, rec_ani, rec_hyd;

  // Caches valid only because the receptor is frozen and the ligand rigid.
  std::vector<double> lig_intra_integral, rec_intra_integral;  // HCT intra sums
  std::vector<double> lig_born_apo, rec_born_apo;
  std::vector<double> lig_sasa_apo, rec_sasa_apo;  // already weighted
  std::vector<double> rec_d2;                      // [np*np] receptor pair squared distances
  std::vector<double> rec_f_apo;                   // [np*np] apo Still f_GB
  std::vector<double> lig_d2;                      // [nl*nl]

  // Parameters mirrored from BiophysicalHamiltonian.
  double interior_eps = 4.0, solvent_eps = 78.5, gb_weight = 0.30;
  double switch_on = 20.0, cutoff = 28.0;
  double hbond_well = 2.60, salt_boost = 1.55;
  double hal_crown = 2.80, hal_belt = 1.15;
  double cation_pi = 3.20, pi_pd = 2.20, pi_t = 1.85;
  double sasa_tension = -0.0072;
  double r_max_born = 30.0;
  double gb_changed_tol = 0.0;
  int use_gb = 1;

  // One Scratch per worker thread, sized once at create time.
  mutable std::vector<Scratch> scratch;
};

// Rotate the body frame into the lab and translate to `com`, writing [3*nl] into `out`.
void BuildPose(const Engine& e, const double* rot, const double* com, double* out) {
  for (int i = 0; i < e.nl; ++i) {
    const double bx = e.lig_body[3 * i], by = e.lig_body[3 * i + 1], bz = e.lig_body[3 * i + 2];
    out[3 * i + 0] = rot[0] * bx + rot[1] * by + rot[2] * bz + com[0];
    out[3 * i + 1] = rot[3] * bx + rot[4] * by + rot[5] * bz + com[1];
    out[3 * i + 2] = rot[6] * bx + rot[7] * by + rot[8] * bz + com[2];
  }
}

inline void RotateVec(const double* rot, const double* v, double* out) {
  out[0] = rot[0] * v[0] + rot[1] * v[1] + rot[2] * v[2];
  out[1] = rot[3] * v[0] + rot[4] * v[1] + rot[5] * v[2];
  out[2] = rot[6] * v[0] + rot[7] * v[1] + rot[8] * v[2];
}

struct Terms {
  double coulomb = 0, gb = 0, vdw = 0, hbond = 0, halogen = 0, cation_pi = 0, pi_stack = 0,
         hydrophobic = 0;
  double Total() const {
    return coulomb + gb + vdw + hbond + halogen + cation_pi + pi_stack + hydrophobic;
  }
};

// The full intermolecular energy at softcore_lambda = 1.
Terms Energy(const Engine& e, const double* rot, const double* com, Scratch& s) {
  const int nl = e.nl, np = e.np;
  Terms t;
  if (nl == 0 || np == 0) return t;

  double* pose = s.pose.data();
  BuildPose(e, rot, com, pose);

  // Rotated ligand frames. These are constants in the body frame; only the rotation moves.
  double* daxes = s.daxes.data();
  double* haxes = s.haxes.data();
  for (int i = 0; i < nl; ++i) {
    RotateVec(rot, &e.lig_daxes_body[3 * i], &daxes[3 * i]);
    RotateVec(rot, &e.lig_haxes_body[3 * i], &haxes[3 * i]);
  }
  double* lringc = s.lringc.data();
  double* lringn = s.lringn.data();
  for (int r = 0; r < e.n_lring; ++r) {
    RotateVec(rot, &e.lig_ringn_body[3 * r], &lringn[3 * r]);
    double c[3];
    RotateVec(rot, &e.lig_ringc_body[3 * r], c);
    lringc[3 * r + 0] = c[0] + com[0];
    lringc[3 * r + 1] = c[1] + com[1];
    lringc[3 * r + 2] = c[2] + com[2];
  }

  double* integ_l = s.integ_l.data();
  double* integ_r = s.integ_r.data();
  double* occ_l = s.occ_l.data();
  double* occ_r = s.occ_r.data();
  for (int i = 0; i < nl; ++i) {
    integ_l[i] = e.lig_intra_integral[i];
    occ_l[i] = 0.0;
  }
  for (int j = 0; j < np; ++j) {
    integ_r[j] = e.rec_intra_integral[j];
    occ_r[j] = 0.0;
  }

  double e_coul = 0, e_vdw = 0, e_hb = 0, e_hal = 0;

  // ---------------------------------------------------------------------------------
  // The one hot loop: the ligand-pocket cross block. Every per-step quantity lives here.
  // Single-threaded on purpose -- the parallelism is one level up, across the twelve
  // independent finite-difference evaluations of a force call, which is a far better
  // granularity than splitting a 38k-pair loop twelve times over.
  // ---------------------------------------------------------------------------------
  {
    double* my_integ = integ_l;
    double* my_occ = occ_l;

    for (int j = 0; j < np; ++j) {
      const Vec3 rj{e.rec_xyz[3 * j], e.rec_xyz[3 * j + 1], e.rec_xyz[3 * j + 2]};
      const double qj = e.rec_q[j];
      const double rho_j = e.rec_rho[j] < 0.15 ? 0.15 : e.rec_rho[j];
      const double rho_tj = (rho_j - 0.09) < 0.10 ? 0.10 : (rho_j - 0.09);
      const double sr_j = e.rec_obc[j] * rho_tj;
      const double r_solv_j = 0.561231 * e.rec_sig[j] + kProbeRadiusWater;
      double acc_integ_j = 0.0, acc_occ_j = 0.0;

      for (int i = 0; i < nl; ++i) {
        const Vec3 ri{pose[3 * i], pose[3 * i + 1], pose[3 * i + 2]};
        const Vec3 d = ri - rj;  // r_i - r_j, matching diff_lp
        const double sq = Dot(d, d);
        const double dist = std::sqrt(sq + 1e-8);

        // --- Generalized Born: cross descreening, both directions ---
        if (e.use_gb) {
          const double rho_i = e.lig_rho[i] < 0.15 ? 0.15 : e.lig_rho[i];
          const double rho_ti = (rho_i - 0.09) < 0.10 ? 0.10 : (rho_i - 0.09);
          const double sr_i = e.lig_obc[i] * rho_ti;
          my_integ[i] += HctPairTerm(dist, rho_ti, sr_j);
          acc_integ_j += HctPairTerm(dist, rho_tj, sr_i);
        }

        // --- apolar occlusion, both directions ---
        {
          const double r_solv_i = 0.561231 * e.lig_sig[i] + kProbeRadiusWater;
          const double sum_rad = r_solv_i + r_solv_j;
          const double denom = sum_rad < 1.0 ? 1.0 : sum_rad;
          double gap = (sum_rad - dist) / denom;
          if (gap < 0.0) gap = 0.0;
          if (gap > 1.0) gap = 1.0;
          const double occ = 0.42 * gap * gap * gap;
          my_occ[i] += occ;
          acc_occ_j += occ;
        }

        const double sw_cut = SmoothSwitchC2(dist, e.switch_on, e.cutoff);

        if (sw_cut > 0.0) {
          // --- Coulomb with the Mehler-Solmajer dielectric ---
          const double eps_ij = MehlerSolmajer(dist, e.interior_eps);
          e_coul += kCoulomb * e.lig_q[i] * qj / (eps_ij * dist) * sw_cut;

          // --- soft-core 12-6 Lennard-Jones at lambda = 1 ---
          const double sig_ij = 0.5 * (e.lig_sig[i] + e.rec_sig[j]);
          const double eps_lj = std::sqrt(e.lig_eps[i] * e.rec_eps[j] + 1e-12);
          const double ros2 = sq / (sig_ij * sig_ij + 1e-8);
          const double ros6 = ros2 * ros2 * ros2;
          const double u_sc = ros6 + 1e-4;
          const double inv_u = 1.0 / u_sc;
          e_vdw += 4.0 * eps_lj * (inv_u * inv_u - inv_u) * sw_cut;
        }

        // --- directional H-bonds and charge-assisted salt bridges ---
        const double sw_hb = SmoothSwitchC2(dist, 3.8, 4.8);
        if (sw_hb > 0.0) {
          const double inv_dist = 1.0 / dist;
          // u_l_to_p = -diff_lp / dist_lp
          const Vec3 ulp{-d.x * inv_dist, -d.y * inv_dist, -d.z * inv_dist};
          double gate = 0.0;
          if (e.lig_don[i] && e.rec_acc[j]) {
            const Vec3 ax{daxes[3 * i], daxes[3 * i + 1], daxes[3 * i + 2]};
            const double c = (Dot(ax, ulp) - 0.15) / 0.85;
            if (c > 0.0) gate += c * c * c;
          }
          if (e.rec_don[j] && e.lig_acc[i]) {
            const Vec3 ax{e.rec_daxes[3 * j], e.rec_daxes[3 * j + 1], e.rec_daxes[3 * j + 2]};
            const double c = (-Dot(ax, ulp) - 0.15) / 0.85;  // u_p_to_l = -u_l_to_p
            if (c > 0.0) gate += c * c * c;
          }
          if (gate > 1.25) gate = 1.25;
          if (gate > 0.0) {
            const bool is_salt =
                (e.lig_cat[i] && e.rec_ani[j]) || (e.lig_ani[i] && e.rec_cat[j]);
            const double strength = e.hbond_well * (is_salt ? e.salt_boost : 1.0);
            const double r_hb2 = sq + 0.09;
            const double ratio2 = (2.85 * 2.85) / r_hb2;
            const double ratio10 = ratio2 * ratio2 * ratio2 * ratio2 * ratio2;
            const double ratio12 = ratio10 * ratio2;
            double v = 5.0 * ratio12 - 6.0 * ratio10;
            if (v < -1.0) v = -1.0;
            if (v > 2.0) v = 2.0;
            e_hb += strength * v * gate * sw_hb;
          }
        }

        // --- anisotropic sigma-hole halogen bonding ---
        if (e.lig_hal[i] && e.rec_acc[j]) {
          const double sw_xb = SmoothSwitchC2(dist, 4.2, 5.2);
          if (sw_xb > 0.0) {
            const double inv_dist = 1.0 / dist;
            const Vec3 ulp{-d.x * inv_dist, -d.y * inv_dist, -d.z * inv_dist};
            const Vec3 ax{haxes[3 * i], haxes[3 * i + 1], haxes[3 * i + 2]};
            double ca = Dot(ax, ulp);
            if (ca < -1.0) ca = -1.0;
            if (ca > 1.0) ca = 1.0;
            double crown = 2.0 * ca - 1.0;
            crown = crown > 0.0 ? crown * crown * crown : 0.0;
            const double s2 = 1.0 - ca * ca;
            const double belt = s2 * s2;
            const double r0 = e.lig_r0x[i];
            const double zc = (dist - r0) / 0.55;
            const double zb = (dist - (r0 - 0.10)) / 0.65;
            e_hal += e.lig_hstr[i] *
                     (-e.hal_crown * crown * std::exp(-0.5 * zc * zc) +
                      e.hal_belt * belt * std::exp(-0.5 * zb * zb)) *
                     sw_xb;
          }
        }
      }
      integ_r[j] += acc_integ_j;
      occ_r[j] += acc_occ_j;
      // Remember how much the ligand actually moved this atom's integral; the pair-sum
      // tolerance below is applied to this, not to a bitwise comparison.
      s.cross_integ[j] = acc_integ_j;
    }
  }

  t.coulomb = e_coul;
  t.vdw = e_vdw;
  t.hbond = e_hb;
  t.halogen = e_hal;

  // ---------------------------------------------------------------------------------
  // Apolar burial. sasa_comp_i = sasa_apo_i * exp(-occ_cross_i), so the buried area is
  // sasa_apo_i * (1 - exp(-occ_cross_i)) >= 0 and needs no clamp.
  // ---------------------------------------------------------------------------------
  {
    double buried = 0.0;
    for (int i = 0; i < nl; ++i) buried += e.lig_sasa_apo[i] * (1.0 - std::exp(-occ_l[i]));
    for (int j = 0; j < np; ++j) buried += e.rec_sasa_apo[j] * (1.0 - std::exp(-occ_r[j]));
    t.hydrophobic = e.sasa_tension * buried;
  }

  // ---------------------------------------------------------------------------------
  // Generalized Born desolvation (hybrid mode: dg_gb = gb_weight * (dG_lig + dG_rec)).
  // ---------------------------------------------------------------------------------
  if (e.use_gb) {
    double* born_l = s.born_l.data();
    double* born_r = s.born_r.data();
    for (int i = 0; i < nl; ++i) {
      const double rho = e.lig_rho[i] < 0.15 ? 0.15 : e.lig_rho[i];
      const double rho_t = (rho - 0.09) < 0.10 ? 0.10 : (rho - 0.09);
      born_l[i] = Obc2Close(integ_l[i], rho, rho_t, e.r_max_born);
    }
    uint8_t* changed = s.changed.data();
    for (int j = 0; j < np; ++j) {
      const double rho = e.rec_rho[j] < 0.15 ? 0.15 : e.rec_rho[j];
      const double rho_t = (rho - 0.09) < 0.10 ? 0.10 : (rho - 0.09);
      born_r[j] = Obc2Close(integ_r[j], rho, rho_t, e.r_max_born);
      // A receptor atom only matters to the desolvation pair sum if the ligand actually
      // moved its Born radius. A bitwise comparison is useless here: HctPairTerm is
      // nonzero at every separation, so every atom would qualify. The far field is
      // nevertheless negligible -- the leading 1/L - 1/U term cancels against the log and
      // middle terms, leaving an O(s^3/d^4) tail worth ~3e-6 at 30 A -- so the test is a
      // tolerance on the cross contribution. The default is 0.0, i.e. exact: measurement
      // showed a 1e-9 tolerance excludes almost nothing (the tail is still ~1.4e-5 at
      // 20 A) and buys only 2%, so trading exactness for it is not worth it. The knob
      // remains for callers who want to profile a looser setting; an integral shift of
      // `gb_changed_tol` moves the radius by about tol * rho_tilde.
      changed[j] = (std::fabs(s.cross_integ[j]) > e.gb_changed_tol) ? 1 : 0;
    }

    const double prefac = -0.5 * kCoulomb * ((1.0 / e.interior_eps) - (1.0 / e.solvent_eps));

    // Ligand: NL x NL, tiny, do it densely.
    double dg_l = 0.0;
    for (int i = 0; i < nl; ++i) {
      for (int k = 0; k < nl; ++k) {
        const double sq = e.lig_d2[static_cast<size_t>(i) * nl + k];
        const double f_comp = StillFgb(sq, born_l[i], born_l[k]);
        const double f_apo = StillFgb(sq, e.lig_born_apo[i], e.lig_born_apo[k]);
        dg_l += e.lig_q[i] * e.lig_q[k] * (f_apo - f_comp) / (f_comp * f_apo + 1e-12);
      }
    }
    dg_l *= prefac;

    // Receptor: NP x NP in principle, but (1/f_comp - 1/f_apo) is exactly zero unless
    // R_i or R_k moved, and the summand is symmetric in (i, k). Partitioning the atoms
    // into changed (C) and unchanged (U) sets leaves only |C|^2/2 + |C|*|U| evaluations
    // instead of |C|*|U|*2 + scanning all NP^2 pairs to discover which ones matter.
    // Reordering the sum shifts the float64 rounding by ~1e-15 relative, which is four
    // orders below the float32 noise of the reference being matched.
    int nc = 0, nu = 0;
    int* cl = s.changed_list.data();
    int* ul = s.unchanged_list.data();
    for (int j = 0; j < np; ++j) {
      if (changed[j]) {
        cl[nc++] = j;
      } else {
        ul[nu++] = j;
      }
    }

    s.n_changed_last = nc;
    double dg_r = 0.0;
    // (a) both atoms changed: the diagonal once, the off-diagonal twice.
    for (int a = 0; a < nc; ++a) {
      const int i = cl[a];
      const double qi = e.rec_q[i];
      const double bi = born_r[i];
      const size_t row = static_cast<size_t>(i) * np;
      const double fc0 = StillFgb(e.rec_d2[row + i], bi, bi);
      const double fa0 = e.rec_f_apo[row + i];
      dg_r += qi * qi * (fa0 - fc0) / (fc0 * fa0 + 1e-12);
      double acc = 0.0;
      for (int b = a + 1; b < nc; ++b) {
        const int k = cl[b];
        const double fc = StillFgb(e.rec_d2[row + k], bi, born_r[k]);
        const double fa = e.rec_f_apo[row + k];
        acc += e.rec_q[k] * (fa - fc) / (fc * fa + 1e-12);
      }
      dg_r += 2.0 * qi * acc;
    }
    // (b) one changed, one not: each such pair appears twice in the full sum.
    for (int a = 0; a < nc; ++a) {
      const int i = cl[a];
      const double qi = e.rec_q[i];
      const double bi = born_r[i];
      const size_t row = static_cast<size_t>(i) * np;
      double acc = 0.0;
      for (int b = 0; b < nu; ++b) {
        const int k = ul[b];
        const double fc = StillFgb(e.rec_d2[row + k], bi, born_r[k]);
        const double fa = e.rec_f_apo[row + k];
        acc += e.rec_q[k] * (fa - fc) / (fc * fa + 1e-12);
      }
      dg_r += 2.0 * qi * acc;
    }
    dg_r *= prefac;

    t.gb = e.gb_weight * (dg_l + dg_r);
  }

  // ---------------------------------------------------------------------------------
  // Aromatic terms. Ring counts are small, so these loops are negligible.
  // ---------------------------------------------------------------------------------
  {
    double e_cp = 0.0;
    // Ligand cation -> pocket ring
    for (int i = 0; i < nl; ++i) {
      if (!e.lig_cat[i]) continue;
      const Vec3 ci{pose[3 * i], pose[3 * i + 1], pose[3 * i + 2]};
      for (int r = 0; r < e.n_rring; ++r) {
        const Vec3 cr{e.rec_ringc[3 * r], e.rec_ringc[3 * r + 1], e.rec_ringc[3 * r + 2]};
        const Vec3 d = ci - cr;
        const double rr = std::sqrt(Dot(d, d) + 1e-8);
        const double sw = SmoothSwitchC2(rr, 5.2, 6.2);
        if (sw <= 0.0) continue;
        const Vec3 nr{e.rec_ringn[3 * r], e.rec_ringn[3 * r + 1], e.rec_ringn[3 * r + 2]};
        const double cp = Dot(d, nr) / rr;
        const double c2 = cp * cp;
        const double z = (rr - 4.0) / 0.85;
        e_cp += -e.cation_pi * (c2 * c2) * std::exp(-0.5 * z * z) * sw;
      }
    }
    // Pocket cation -> ligand ring
    for (int j = 0; j < np; ++j) {
      if (!e.rec_cat[j]) continue;
      const Vec3 cj{e.rec_xyz[3 * j], e.rec_xyz[3 * j + 1], e.rec_xyz[3 * j + 2]};
      for (int r = 0; r < e.n_lring; ++r) {
        const Vec3 cr{lringc[3 * r], lringc[3 * r + 1], lringc[3 * r + 2]};
        const Vec3 d = cj - cr;
        const double rr = std::sqrt(Dot(d, d) + 1e-8);
        const double sw = SmoothSwitchC2(rr, 5.2, 6.2);
        if (sw <= 0.0) continue;
        const Vec3 nr{lringn[3 * r], lringn[3 * r + 1], lringn[3 * r + 2]};
        const double cp = Dot(d, nr) / rr;
        const double c2 = cp * cp;
        const double z = (rr - 4.0) / 0.85;
        e_cp += -e.cation_pi * (c2 * c2) * std::exp(-0.5 * z * z) * sw;
      }
    }
    t.cation_pi = e_cp;

    // Pi-pi stacking: Hunter-Sanders parallel-displaced plus T-shaped.
    double e_pp = 0.0;
    for (int a = 0; a < e.n_lring; ++a) {
      const Vec3 ca{lringc[3 * a], lringc[3 * a + 1], lringc[3 * a + 2]};
      const Vec3 na{lringn[3 * a], lringn[3 * a + 1], lringn[3 * a + 2]};
      for (int b = 0; b < e.n_rring; ++b) {
        const Vec3 cb{e.rec_ringc[3 * b], e.rec_ringc[3 * b + 1], e.rec_ringc[3 * b + 2]};
        const Vec3 d = ca - cb;
        const double rr = std::sqrt(Dot(d, d) + 1e-8);
        const double sw = SmoothSwitchC2(rr, 5.5, 6.6);
        if (sw <= 0.0) continue;
        const Vec3 nb{e.rec_ringn[3 * b], e.rec_ringn[3 * b + 1], e.rec_ringn[3 * b + 2]};
        const double inv = 1.0 / rr;
        const Vec3 u{d.x * inv, d.y * inv, d.z * inv};
        const double dot_nn = Dot(na, nb);
        const double c12 = dot_nn * dot_nn;
        const double pa = Dot(na, u), pb2 = Dot(nb, u);
        const double p1 = pa * pa, p2 = pb2 * pb2;
        const double p_bar = 0.5 * (p1 + p2);
        double shape = 1.35 * p_bar * (1.20 - 0.50 * p_bar);
        if (shape < 0.0) shape = 0.0;
        if (shape > 1.2) shape = 1.2;
        const double zp = (rr - 3.95) / 0.75;
        const double zt = (rr - 4.95) / 0.80;
        const double v_pd = -e.pi_pd * c12 * shape * std::exp(-0.5 * zp * zp);
        const double v_t = -e.pi_t * (1.0 - c12) * (0.5 * (p1 + p2)) * std::exp(-0.5 * zt * zt);
        e_pp += (v_pd + v_t) * sw;
      }
    }
    t.pi_stack = e_pp;
  }

  return t;
}

// ---------------------------------------------------------------------------------------
// Quaternion helpers. (w, x, y, z), body -> lab.
// ---------------------------------------------------------------------------------------

void QuatToMatrix(const double* q, double* m) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  m[0] = 1 - 2 * (y * y + z * z);
  m[1] = 2 * (x * y - w * z);
  m[2] = 2 * (x * z + w * y);
  m[3] = 2 * (x * y + w * z);
  m[4] = 1 - 2 * (x * x + z * z);
  m[5] = 2 * (y * z - w * x);
  m[6] = 2 * (x * z - w * y);
  m[7] = 2 * (y * z + w * x);
  m[8] = 1 - 2 * (x * x + y * y);
}

void QuatMul(const double* a, const double* b, double* out) {
  out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

void QuatNormalize(double* q) {
  const double inv = 1.0 / std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  q[0] *= inv;
  q[1] *= inv;
  q[2] *= inv;
  q[3] *= inv;
}

// Exact flow of H_a = L_a^2/(2 I_a): L rotates about body axis `a`, attitude advances.
void FreeTopSubstep(double* q, double* L, const double* I, int a, double tau) {
  const double phi = tau * L[a] / I[a];
  if (phi == 0.0) return;
  const double c = std::cos(phi), s = std::sin(phi);
  const int b = (a + 1) % 3, cc = (a + 2) % 3;
  const double lb = L[b], lc = L[cc];
  L[b] = c * lb + s * lc;  // Rodrigues about coordinate axis `a` by -phi
  L[cc] = -s * lb + c * lc;
  double dq[4] = {std::cos(0.5 * phi), 0.0, 0.0, 0.0};
  dq[1 + a] = std::sin(0.5 * phi);
  double out[4];
  QuatMul(q, dq, out);
  q[0] = out[0];
  q[1] = out[1];
  q[2] = out[2];
  q[3] = out[3];
  QuatNormalize(q);
}

// Dullweber-Leimkuhler-McLachlan symmetric R-splitting: second order, and exactly
// angular-momentum conserving because every stage is a rotation. Measured against the
// gyroscopic-half-kick alternative in float64: energy drift scales as dt^2 here versus dt
// there, and |L| is conserved to 2e-14 versus 1.5e-4 at a 1 fs step.
void FreeTopFlow(double* q, double* L, const double* I, double dt) {
  FreeTopSubstep(q, L, I, 0, 0.5 * dt);
  FreeTopSubstep(q, L, I, 1, 0.5 * dt);
  FreeTopSubstep(q, L, I, 2, dt);
  FreeTopSubstep(q, L, I, 1, 0.5 * dt);
  FreeTopSubstep(q, L, I, 0, 0.5 * dt);
}

// Generalized rigid-body forces by central differences on the six degrees of freedom.
// `fnet` is the net force (kcal/mol/A), `tq` the LAB-frame torque (kcal/mol).
//
// The twelve perturbed energies are independent, so they run concurrently -- one task per
// perturbation, each with its own Scratch. This is deliberately NOT OpenMP: PyTorch
// already has an OpenMP runtime resident in the process, and loading a second libomp
// through ctypes crashes inside __kmp_suspend_initialize_thread on macOS.
void GeneralizedForces(const Engine& e, const double* q, const double* com, double h_trans,
                       double h_rot, double* fnet, double* tq) {
  struct Task {
    double rot[9];
    double com[3];
    double out;
  };
  std::array<Task, 12> tasks{};

  double rot0[9];
  QuatToMatrix(q, rot0);
  for (int k = 0; k < 3; ++k) {
    for (int sgn = 0; sgn < 2; ++sgn) {
      Task& t = tasks[2 * k + sgn];
      std::memcpy(t.rot, rot0, sizeof(rot0));
      t.com[0] = com[0];
      t.com[1] = com[1];
      t.com[2] = com[2];
      t.com[k] += (sgn == 0 ? h_trans : -h_trans);
    }
  }
  for (int k = 0; k < 3; ++k) {
    for (int sgn = 0; sgn < 2; ++sgn) {
      // Left-multiplying by a small rotation rotates the body about the LAB axis k.
      const double ang = (sgn == 0 ? h_rot : -h_rot);
      double dq[4] = {std::cos(0.5 * ang), 0, 0, 0};
      dq[1 + k] = std::sin(0.5 * ang);
      double qp[4];
      QuatMul(dq, q, qp);
      QuatNormalize(qp);
      Task& t = tasks[6 + 2 * k + sgn];
      QuatToMatrix(qp, t.rot);
      t.com[0] = com[0];
      t.com[1] = com[1];
      t.com[2] = com[2];
    }
  }

  const int workers = static_cast<int>(e.scratch.size());
  if (workers <= 1) {
    for (auto& t : tasks) t.out = Energy(e, t.rot, t.com, e.scratch[0]).Total();
  } else {
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (int w = 0; w < workers; ++w) {
      pool.emplace_back([&e, &tasks, &next, w]() {
        for (int i = next.fetch_add(1); i < 12; i = next.fetch_add(1)) {
          tasks[i].out = Energy(e, tasks[i].rot, tasks[i].com, e.scratch[w]).Total();
        }
      });
    }
    for (auto& th : pool) th.join();
  }

  for (int k = 0; k < 3; ++k) {
    fnet[k] = -(tasks[2 * k].out - tasks[2 * k + 1].out) / (2.0 * h_trans);
    tq[k] = -(tasks[6 + 2 * k].out - tasks[6 + 2 * k + 1].out) / (2.0 * h_rot);
  }
}

inline void Fill(std::vector<double>& v, const double* src, size_t n) {
  v.assign(src, src + n);
}

}  // namespace

// =======================================================================================
// C ABI
// =======================================================================================

/// Layout of the scalar parameter block handed to gex_create, in order.
struct GexParams {
  double interior_eps, solvent_eps, gb_weight, switch_on, cutoff;
  double hbond_well, salt_boost, hal_crown, hal_belt;
  double cation_pi, pi_pd, pi_t, sasa_tension, r_max_born;
  double use_gb, gb_changed_tol;
};

GEX_API void* gex_create(int nl, int np, int n_lring, int n_rring, const double* lig_body,
                         const double* lig_daxes_body, const double* lig_haxes_body,
                         const double* lig_ringc_body, const double* lig_ringn_body,
                         const double* lig_scalar,  // [7*nl]: q, sig, eps, rho, obc, hstr, r0x
                         const uint8_t* lig_flags,  // [6*nl]: don, acc, cat, ani, hyd, hal
                         const double* rec_xyz, const double* rec_daxes,
                         const double* rec_ringc, const double* rec_ringn,
                         const double* rec_scalar,  // [5*np]: q, sig, eps, rho, obc
                         const uint8_t* rec_flags,  // [5*np]: don, acc, cat, ani, hyd
                         const double* lig_intra_integral, const double* rec_intra_integral,
                         const double* lig_born_apo, const double* rec_born_apo,
                         const double* lig_sasa_apo, const double* rec_sasa_apo,
                         const double* lig_d2, const double* rec_d2, const double* rec_f_apo,
                         const GexParams* params);

GEX_API int gex_version() { return 7; }

/// Build an engine from one packed buffer, for callers that cannot marshal 27 arguments.
///
/// This exists for the WebAssembly build. Every array `gex_create` needs is produced by
/// setup code that already exists and is verified in Python -- the HCT descreening
/// integrals, the apo Born radii, the apo Still f_GB matrix, the apo SASA, the donor and
/// sigma-hole axes, the ring frames. Re-implementing that in JavaScript would be a second
/// implementation to keep in step with the first, so instead Python packs the finished
/// arrays and the browser only runs the hot loop.
///
/// Layout: a header of 4 int32 (nl, np, n_lring, n_rring), padded to 8 bytes, then the
/// GexParams block, then the float64 arrays in exactly the order gex_create takes them,
/// then the two uint8 flag arrays.
GEX_API void* gex_create_packed(const uint8_t* blob, int32_t n_bytes) {
  if (!blob || n_bytes < 32) return nullptr;
  const int32_t* hdr = reinterpret_cast<const int32_t*>(blob);
  const int nl = hdr[0], np = hdr[1], n_lring = hdr[2], n_rring = hdr[3];
  if (nl <= 0 || np <= 0) return nullptr;

  size_t off = 16;
  const GexParams* params = reinterpret_cast<const GexParams*>(blob + off);
  off += sizeof(GexParams);

  const double* d = reinterpret_cast<const double*>(blob + off);
  size_t k = 0;
  auto take = [&](size_t n) {
    const double* p = d + k;
    k += n;
    return p;
  };
  const double* lig_body = take(3u * nl);
  const double* lig_daxes = take(3u * nl);
  const double* lig_haxes = take(3u * nl);
  const double* lig_ringc = take(3u * static_cast<size_t>(n_lring > 0 ? n_lring : 1));
  const double* lig_ringn = take(3u * static_cast<size_t>(n_lring > 0 ? n_lring : 1));
  const double* lig_scalar = take(7u * nl);
  const double* rec_xyz = take(3u * np);
  const double* rec_daxes = take(3u * np);
  const double* rec_ringc = take(3u * static_cast<size_t>(n_rring > 0 ? n_rring : 1));
  const double* rec_ringn = take(3u * static_cast<size_t>(n_rring > 0 ? n_rring : 1));
  const double* rec_scalar = take(5u * np);
  const double* lig_intra = take(nl);
  const double* rec_intra = take(np);
  const double* lig_born_apo = take(nl);
  const double* rec_born_apo = take(np);
  const double* lig_sasa_apo = take(nl);
  const double* rec_sasa_apo = take(np);
  const double* lig_d2 = take(static_cast<size_t>(nl) * nl);
  // rec_d2 and rec_f_apo are NOT in the blob. They are two np x np float64 matrices --
  // 19 MB for a 1090-atom pocket, which is absurd to download when both are a few
  // milliseconds of O(N^2) arithmetic from data already present. Derive them here.
  std::vector<double> rec_d2(static_cast<size_t>(np) * np);
  std::vector<double> rec_f_apo(static_cast<size_t>(np) * np);
  for (int i = 0; i < np; ++i) {
    const double xi = rec_xyz[3 * i], yi = rec_xyz[3 * i + 1], zi = rec_xyz[3 * i + 2];
    const double bi = rec_born_apo[i];
    const size_t row = static_cast<size_t>(i) * np;
    for (int j = 0; j < np; ++j) {
      const double dx = xi - rec_xyz[3 * j];
      const double dy = yi - rec_xyz[3 * j + 1];
      const double dz = zi - rec_xyz[3 * j + 2];
      const double d2 = dx * dx + dy * dy + dz * dz;
      rec_d2[row + j] = d2;
      rec_f_apo[row + j] = StillFgb(d2, bi, rec_born_apo[j]);
    }
  }

  const uint8_t* flags = blob + off + k * sizeof(double);
  const uint8_t* lig_flags = flags;
  const uint8_t* rec_flags = flags + 6u * nl;

  const size_t need = off + k * sizeof(double) + 6u * nl + 5u * np;
  if (static_cast<size_t>(n_bytes) < need) return nullptr;

  return gex_create(nl, np, n_lring, n_rring, lig_body, lig_daxes, lig_haxes, lig_ringc,
                    lig_ringn, lig_scalar, lig_flags, rec_xyz, rec_daxes, rec_ringc,
                    rec_ringn, rec_scalar, rec_flags, lig_intra, rec_intra, lig_born_apo,
                    rec_born_apo, lig_sasa_apo, rec_sasa_apo, lig_d2, rec_d2.data(),
                    rec_f_apo.data(), params);
}

GEX_API void* gex_create(int nl, int np, int n_lring, int n_rring, const double* lig_body,
                         const double* lig_daxes_body, const double* lig_haxes_body,
                         const double* lig_ringc_body, const double* lig_ringn_body,
                         const double* lig_scalar,  // [7*nl]: q, sig, eps, rho, obc, hstr, r0x
                         const uint8_t* lig_flags,  // [6*nl]: don, acc, cat, ani, hyd, hal
                         const double* rec_xyz, const double* rec_daxes,
                         const double* rec_ringc, const double* rec_ringn,
                         const double* rec_scalar,  // [5*np]: q, sig, eps, rho, obc
                         const uint8_t* rec_flags,  // [5*np]: don, acc, cat, ani, hyd
                         const double* lig_intra_integral, const double* rec_intra_integral,
                         const double* lig_born_apo, const double* rec_born_apo,
                         const double* lig_sasa_apo, const double* rec_sasa_apo,
                         const double* lig_d2, const double* rec_d2, const double* rec_f_apo,
                         const GexParams* params) {
  Engine* e = new Engine();
  e->nl = nl;
  e->np = np;
  e->n_lring = n_lring;
  e->n_rring = n_rring;

  Fill(e->lig_body, lig_body, 3u * nl);
  Fill(e->lig_daxes_body, lig_daxes_body, 3u * nl);
  Fill(e->lig_haxes_body, lig_haxes_body, 3u * nl);
  if (n_lring > 0) {
    Fill(e->lig_ringc_body, lig_ringc_body, 3u * n_lring);
    Fill(e->lig_ringn_body, lig_ringn_body, 3u * n_lring);
  }
  e->lig_q.assign(lig_scalar + 0u * nl, lig_scalar + 1u * nl);
  e->lig_sig.assign(lig_scalar + 1u * nl, lig_scalar + 2u * nl);
  e->lig_eps.assign(lig_scalar + 2u * nl, lig_scalar + 3u * nl);
  e->lig_rho.assign(lig_scalar + 3u * nl, lig_scalar + 4u * nl);
  e->lig_obc.assign(lig_scalar + 4u * nl, lig_scalar + 5u * nl);
  e->lig_hstr.assign(lig_scalar + 5u * nl, lig_scalar + 6u * nl);
  e->lig_r0x.assign(lig_scalar + 6u * nl, lig_scalar + 7u * nl);
  e->lig_don.assign(lig_flags + 0u * nl, lig_flags + 1u * nl);
  e->lig_acc.assign(lig_flags + 1u * nl, lig_flags + 2u * nl);
  e->lig_cat.assign(lig_flags + 2u * nl, lig_flags + 3u * nl);
  e->lig_ani.assign(lig_flags + 3u * nl, lig_flags + 4u * nl);
  e->lig_hyd.assign(lig_flags + 4u * nl, lig_flags + 5u * nl);
  e->lig_hal.assign(lig_flags + 5u * nl, lig_flags + 6u * nl);

  Fill(e->rec_xyz, rec_xyz, 3u * np);
  Fill(e->rec_daxes, rec_daxes, 3u * np);
  if (n_rring > 0) {
    Fill(e->rec_ringc, rec_ringc, 3u * n_rring);
    Fill(e->rec_ringn, rec_ringn, 3u * n_rring);
  }
  e->rec_q.assign(rec_scalar + 0u * np, rec_scalar + 1u * np);
  e->rec_sig.assign(rec_scalar + 1u * np, rec_scalar + 2u * np);
  e->rec_eps.assign(rec_scalar + 2u * np, rec_scalar + 3u * np);
  e->rec_rho.assign(rec_scalar + 3u * np, rec_scalar + 4u * np);
  e->rec_obc.assign(rec_scalar + 4u * np, rec_scalar + 5u * np);
  e->rec_don.assign(rec_flags + 0u * np, rec_flags + 1u * np);
  e->rec_acc.assign(rec_flags + 1u * np, rec_flags + 2u * np);
  e->rec_cat.assign(rec_flags + 2u * np, rec_flags + 3u * np);
  e->rec_ani.assign(rec_flags + 3u * np, rec_flags + 4u * np);
  e->rec_hyd.assign(rec_flags + 4u * np, rec_flags + 5u * np);

  Fill(e->lig_intra_integral, lig_intra_integral, nl);
  Fill(e->rec_intra_integral, rec_intra_integral, np);
  Fill(e->lig_born_apo, lig_born_apo, nl);
  Fill(e->rec_born_apo, rec_born_apo, np);
  Fill(e->lig_sasa_apo, lig_sasa_apo, nl);
  Fill(e->rec_sasa_apo, rec_sasa_apo, np);
  Fill(e->lig_d2, lig_d2, static_cast<size_t>(nl) * nl);
  Fill(e->rec_d2, rec_d2, static_cast<size_t>(np) * np);
  Fill(e->rec_f_apo, rec_f_apo, static_cast<size_t>(np) * np);

  e->interior_eps = params->interior_eps;
  e->solvent_eps = params->solvent_eps;
  e->gb_weight = params->gb_weight;
  e->switch_on = params->switch_on;
  e->cutoff = params->cutoff;
  e->hbond_well = params->hbond_well;
  e->salt_boost = params->salt_boost;
  e->hal_crown = params->hal_crown;
  e->hal_belt = params->hal_belt;
  e->cation_pi = params->cation_pi;
  e->pi_pd = params->pi_pd;
  e->pi_t = params->pi_t;
  e->sasa_tension = params->sasa_tension;
  e->r_max_born = params->r_max_born;
  e->use_gb = params->use_gb != 0.0 ? 1 : 0;
  e->gb_changed_tol = params->gb_changed_tol;

  // One Scratch per worker. Twelve tasks per force call, so more than twelve workers
  // cannot help; hardware_concurrency may report 0, in which case fall back to serial.
  //
  // WebAssembly runs single-threaded unless the build opted into pthreads, which also
  // needs SharedArrayBuffer and COOP/COEP headers on the serving origin. Forcing one
  // worker here keeps the browser build from ever reaching std::thread, whose emscripten
  // stub throws at runtime rather than failing to link.
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
  const int workers = 1;
#else
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) hw = 1;
  const int workers = static_cast<int>(hw > 12u ? 12u : hw);
#endif
  e->scratch.resize(static_cast<size_t>(workers));
  for (auto& sc : e->scratch) sc.Resize(nl, np, n_lring);
  return e;
}

GEX_API void gex_destroy(void* handle) { delete static_cast<Engine*>(handle); }

/// Energy at a pose given by a quaternion and centre of mass. Writes the eight terms into
/// `out_terms` (coulomb, gb, vdw, hbond, halogen, cation_pi, pi_stack, hydrophobic) when
/// non-null, and returns the total.
GEX_API double gex_energy(void* handle, const double* quat, const double* com,
                          double* out_terms) {
  const Engine& e = *static_cast<Engine*>(handle);
  double rot[9];
  QuatToMatrix(quat, rot);
  const Terms t = Energy(e, rot, com, e.scratch[0]);
  if (out_terms) {
    out_terms[0] = t.coulomb;
    out_terms[1] = t.gb;
    out_terms[2] = t.vdw;
    out_terms[3] = t.hbond;
    out_terms[4] = t.halogen;
    out_terms[5] = t.cation_pi;
    out_terms[6] = t.pi_stack;
    out_terms[7] = t.hydrophobic;
  }
  return t.Total();
}

/// Net force and lab-frame torque by central differences on the six rigid-body DOF.
GEX_API void gex_forces(void* handle, const double* quat, const double* com, double h_trans,
                        double h_rot, double* out_force, double* out_torque) {
  const Engine& e = *static_cast<Engine*>(handle);
  GeneralizedForces(e, quat, com, h_trans, h_rot, out_force, out_torque);
}

/// Lab-frame atom coordinates of the ligand at a pose. `out` must hold 3*nl doubles.
GEX_API void gex_pose(void* handle, const double* quat, const double* com, double* out) {
  const Engine& e = *static_cast<Engine*>(handle);
  double rot[9];
  QuatToMatrix(quat, rot);
  BuildPose(e, rot, com, out);
}

/// Run the staged-then-released rigid-body Langevin trajectory entirely natively.
///
/// `cfg` layout (all doubles, in order):
///   0 dt, 1 temperature_k, 2 gamma_trans, 3 zeta_rot, 4 launch_distance,
///   5 stage_pull_velocity, 6 stage_spring_k, 7 stage_max_steps, 8 release_steps,
///   9 record_every, 10 seed, 11 h_trans, 12 h_rot
///
/// Each frame record is `21 + 3*nl` doubles: step, time, com[3], quat[4], u_total,
/// eight energy terms, T_trans, T_rot, min_contact, then the lab-frame coordinates.
/// `out_summary` receives 12 doubles. Returns the frame count, or -1 if the buffer is
/// too small.
GEX_API int gex_simulate(void* handle, const double* cfg, const double* moments,
                         const double* com0, const double* quat0, const double* escape,
                         double total_mass, double* out_frames, int max_frames,
                         double* out_summary) {
  const Engine& e = *static_cast<Engine*>(handle);
  const int nl = e.nl;

  const double dt = cfg[0];
  const double temp = cfg[1];
  const double gamma_t = cfg[2];
  const double zeta_r = cfg[3];
  const double launch = cfg[4];
  const double pull_v = cfg[5];
  const double spring_k = cfg[6];
  const int stage_max = static_cast<int>(cfg[7]);
  const int release = static_cast<int>(cfg[8]);
  const int rec_every = static_cast<int>(cfg[9]) > 0 ? static_cast<int>(cfg[9]) : 1;
  const uint64_t seed = static_cast<uint64_t>(cfg[10]);
  const double h_trans = cfg[11];
  const double h_rot = cfg[12];

  const double kt = temp * kBoltzmann;
  const double kt_red = kMdAccel * kt;

  double q[4] = {quat0[0], quat0[1], quat0[2], quat0[3]};
  double com[3] = {com0[0], com0[1], com0[2]};
  double vel[3] = {0, 0, 0};
  double L[3] = {0, 0, 0};
  const double I[3] = {moments[0], moments[1], moments[2]};

  std::mt19937_64 rng(seed);
  std::normal_distribution<double> gauss(0.0, 1.0);

  const double c1t = std::exp(-gamma_t * dt);
  const double c2t = std::sqrt(std::fmax(0.0, 1.0 - c1t * c1t));
  double c1r[3], c2r[3], l_sigma[3];
  for (int a = 0; a < 3; ++a) {
    const double rate = zeta_r / I[a];
    c1r[a] = std::exp(-rate * dt);
    c2r[a] = std::sqrt(std::fmax(0.0, 1.0 - c1r[a] * c1r[a]));
    l_sigma[a] = std::sqrt(I[a] * kt_red);
  }

  // One BAOAB step. `extra` is an external centre-of-mass force, or null.
  auto step = [&](const double* extra, bool thermostat) {
    double fnet[3], tq[3], rot[9];
    GeneralizedForces(e, q, com, h_trans, h_rot, fnet, tq);
    if (extra) {
      fnet[0] += extra[0];
      fnet[1] += extra[1];
      fnet[2] += extra[2];
    }
    QuatToMatrix(q, rot);
    const double tb[3] = {rot[0] * tq[0] + rot[3] * tq[1] + rot[6] * tq[2],
                          rot[1] * tq[0] + rot[4] * tq[1] + rot[7] * tq[2],
                          rot[2] * tq[0] + rot[5] * tq[1] + rot[8] * tq[2]};
    for (int k = 0; k < 3; ++k) {
      vel[k] += 0.5 * dt * kMdAccel * fnet[k] / total_mass;
      L[k] += 0.5 * dt * kMdAccel * tb[k];
    }
    for (int k = 0; k < 3; ++k) com[k] += 0.5 * dt * vel[k];
    FreeTopFlow(q, L, I, 0.5 * dt);
    if (thermostat) {
      for (int k = 0; k < 3; ++k) {
        vel[k] = c1t * vel[k] + c2t * std::sqrt(kt_red / total_mass) * gauss(rng);
        L[k] = c1r[k] * L[k] + c2r[k] * l_sigma[k] * gauss(rng);
      }
    }
    for (int k = 0; k < 3; ++k) com[k] += 0.5 * dt * vel[k];
    FreeTopFlow(q, L, I, 0.5 * dt);
    GeneralizedForces(e, q, com, h_trans, h_rot, fnet, tq);
    if (extra) {
      fnet[0] += extra[0];
      fnet[1] += extra[1];
      fnet[2] += extra[2];
    }
    QuatToMatrix(q, rot);
    const double tb2[3] = {rot[0] * tq[0] + rot[3] * tq[1] + rot[6] * tq[2],
                           rot[1] * tq[0] + rot[4] * tq[1] + rot[7] * tq[2],
                           rot[2] * tq[0] + rot[5] * tq[1] + rot[8] * tq[2]};
    for (int k = 0; k < 3; ++k) {
      vel[k] += 0.5 * dt * kMdAccel * fnet[k] / total_mass;
      L[k] += 0.5 * dt * kMdAccel * tb2[k];
    }
  };

  auto displacement = [&]() {
    const double dx = com[0] - com0[0], dy = com[1] - com0[1], dz = com[2] - com0[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  };

  // ---- Stage 1: steered MD outward. Not recorded. ----
  int staged = 0;
  double max_pull = 0.0;
  for (int s = 0; s < stage_max; ++s) {
    double travelled = pull_v * s * dt;
    if (travelled > launch) travelled = launch;
    double pull[3];
    for (int k = 0; k < 3; ++k) pull[k] = spring_k * (com0[k] + escape[k] * travelled - com[k]);
    const double pn = std::sqrt(pull[0] * pull[0] + pull[1] * pull[1] + pull[2] * pull[2]);
    if (pn > max_pull) max_pull = pn;
    step(pull, true);
    ++staged;
    if (displacement() >= launch) break;
  }
  const double launch_disp = displacement();

  // ---- Stage 2: release and record ----
  for (int k = 0; k < 3; ++k) {
    vel[k] = std::sqrt(kt_red / total_mass) * gauss(rng);
    L[k] = l_sigma[k] * gauss(rng);
  }

  const int stride = 21 + 3 * nl;
  int n_frames = 0;
  double ke_t_acc = 0.0, ke_r_acc = 0.0;
  double closest = 1e30;
  std::vector<double> pose(3u * nl);

  for (int s = 0; s < release; ++s) {
    step(nullptr, true);
    const double ke_t =
        0.5 * total_mass * (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]) / kMdAccel;
    const double ke_r =
        0.5 * (L[0] * L[0] / I[0] + L[1] * L[1] / I[1] + L[2] * L[2] / I[2]) / kMdAccel;
    ke_t_acc += ke_t;
    ke_r_acc += ke_r;

    if (s % rec_every == 0 || s == release - 1) {
      if (n_frames >= max_frames) return -1;
      double terms[8];
      const double u = gex_energy(handle, q, com, terms);
      double rot[9];
      QuatToMatrix(q, rot);
      BuildPose(e, rot, com, pose.data());
      double mind = 1e30;
      for (int i = 0; i < nl; ++i) {
        for (int j = 0; j < e.np; ++j) {
          const double dx = pose[3 * i] - e.rec_xyz[3 * j];
          const double dy = pose[3 * i + 1] - e.rec_xyz[3 * j + 1];
          const double dz = pose[3 * i + 2] - e.rec_xyz[3 * j + 2];
          const double d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < mind) mind = d2;
        }
      }
      mind = std::sqrt(mind);
      if (mind < closest) closest = mind;

      double* f = out_frames + static_cast<size_t>(n_frames) * stride;
      f[0] = static_cast<double>(s);
      f[1] = s * dt;
      f[2] = com[0];
      f[3] = com[1];
      f[4] = com[2];
      f[5] = q[0];
      f[6] = q[1];
      f[7] = q[2];
      f[8] = q[3];
      f[9] = u;
      for (int k = 0; k < 8; ++k) f[10 + k] = terms[k];
      f[18] = 2.0 * ke_t / (3.0 * kBoltzmann);
      f[19] = 2.0 * ke_r / (3.0 * kBoltzmann);
      f[20] = mind;
      std::memcpy(f + 21, pose.data(), sizeof(double) * 3u * nl);
      ++n_frames;
    }
  }

  const double inv_n = release > 0 ? 1.0 / release : 0.0;
  out_summary[0] = staged;
  out_summary[1] = launch_disp;
  out_summary[2] = max_pull;
  out_summary[3] = ke_t_acc * inv_n;
  out_summary[4] = ke_r_acc * inv_n;
  out_summary[5] = 2.0 * (ke_t_acc * inv_n) / (3.0 * kBoltzmann);
  out_summary[6] = 2.0 * (ke_r_acc * inv_n) / (3.0 * kBoltzmann);
  out_summary[7] = displacement();
  out_summary[8] = closest;
  out_summary[9] = 1.5 * kt;
  out_summary[10] = static_cast<double>(stride);
  out_summary[11] = static_cast<double>(n_frames);
  return n_frames;
}
