"""Amount-conserving oral parent-drug PK with buffered gut and reversible BBB transport."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, replace

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import PPoly
from scipy.optimize import least_squares

from geneuclidean.assays import (
    AqueousEquilibrium,
    aqueous_equilibrium,
    well_stirred_hepatic_clearance,
)

Vector = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class OralMedium:
    """A fixed, buffered lumen. The solubility cap is apparent at the specified pH.

    The optional partner is retained at constant total concentration and binds
    free solute reversibly in a 1:1 complex. Only free solute is absorbed.
    """

    ph: float
    volume_l: float
    free_solubility_um: float
    partner_total_um: float = 0.0
    partner_kd_um: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.volume_l) or self.volume_l <= 0:
            raise ValueError("volume_l must be finite and positive")
        self._partition(0.0, self.free_solubility_um)

    def _partition(self, total_um: float, free_limit_um: float) -> AqueousEquilibrium:
        return aqueous_equilibrium(
            total_um=total_um,
            intrinsic_solubility_um=free_limit_um,
            ph=self.ph,
            partner_total_um=self.partner_total_um,
            kd_um=self.partner_kd_um,
        )

    def free_amount_umol(self, amount_umol: float) -> float:
        """Return unbound dissolved drug after equilibration with any solid."""
        mixture = self._partition(amount_umol / self.volume_l, self.free_solubility_um)
        return mixture.free_um * self.volume_l

    def _free_in_liquid_umol(self, dissolved_umol: float) -> float:
        if self.partner_total_um == 0.0:
            return dissolved_umol
        total_um = dissolved_umol / self.volume_l
        # Supplying a cap at least as large as the liquid concentration solves only
        # binding. Supersaturated solute can then precipitate on its own time scale.
        mixture = self._partition(total_um, max(total_um, self.free_solubility_um))
        return mixture.free_um * self.volume_l

    @property
    def dissolved_capacity_umol(self) -> float:
        """Total free plus complexed drug that can dissolve at equilibrium."""
        if self.partner_total_um == 0.0:
            return self.free_solubility_um * self.volume_l
        # This load saturates the free solute even if every partner molecule binds.
        mixture = self._partition(
            self.free_solubility_um + self.partner_total_um, self.free_solubility_um
        )
        return mixture.dissolved_um * self.volume_l


@dataclass(frozen=True, kw_only=True)
class OralFormulation:
    """Empirical solid/liquid exchange for one well-mixed buffered lumen.

    The rates (1/h) relax dissolved drug toward the equilibrium inventory at the
    current total load. Dissolution is limited by the solid available and the
    remaining solution capacity. Supersaturated liquid precipitates at a rate
    proportional to its excess; precipitated drug joins the dissolvable solid.
    Both forms transit at the same rate. Use rates measured for the specified
    medium and material. No particle surface area or nucleation lag is modeled.
    """

    dissolution_per_h: float
    precipitation_per_h: float
    initial_dissolved_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in ("dissolution_per_h", "precipitation_per_h"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.initial_dissolved_fraction <= 1:
            raise ValueError("initial_dissolved_fraction must be in [0, 1]")

    def _net_dissolution(self, dissolved: float, solid: float, capacity: float) -> float:
        """Signed transfer from solid to liquid, in the supplied amount units per hour."""
        gap = capacity - max(dissolved, 0.0)
        if gap < 0:
            return self.precipitation_per_h * gap
        return self.dissolution_per_h * min(max(solid, 0.0), gap)


@dataclass(frozen=True, kw_only=True)
class GastricStage:
    """Optional gastric reservoir upstream of the intestinal absorption compartment.

    Supply gastric pH, apparent solubility, volume and material-specific rates.
    Dissolved drug, including soluble complex, empties with the liquid. Solid
    empties independently. Each medium retains its specified pH, volume and
    partner concentration. Only parent drug crosses compartments. All absorption
    and unabsorbed transit loss occur downstream in the intestine.
    """

    medium: OralMedium
    formulation: OralFormulation
    solid_emptying_per_h: float
    liquid_emptying_per_h: float

    def __post_init__(self) -> None:
        for name in ("solid_emptying_per_h", "liquid_emptying_per_h"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class SinkDissolutionFit:
    """Fitted release rate (1/h) and diagnostics for the specified dissolution assay."""

    dissolution_per_h: float
    rmse_umol: float
    capacity_to_load_ratio: float
    n_observations: int


def fit_sink_dissolution(
    medium: OralMedium,
    times_h: Sequence[float],
    dissolved_umol: Sequence[float],
    *,
    amount_umol: float,
    initial_dissolved_fraction: float = 0.0,
) -> SinkDissolutionFit:
    """Fit the formulation's dissolution rate to an unsaturated, closed in-vitro assay.

    Supply times since loading and measured total dissolved parent (free plus
    soluble complex) at each time. The assay has fixed volume, pH and material
    load, with no irreversible loss, absorption, or transit. Correct for sampling
    losses before fitting. When capacity exceeds the entire load A, the model
    reduces exactly to D(t) = A - (A - D0) exp(-k t). The fit minimizes squared
    errors on the measured amount scale, with a known initial fraction D0/A.
    Replicates contribute individual residuals. Observations above the nominal
    load remain in the residuals so assay over-recovery is visible. The fitted
    curve is bounded by the load. Formulation and medium determine transferability.
    """
    if not math.isfinite(amount_umol) or amount_umol <= 0:
        raise ValueError("amount_umol must be finite and positive")
    if not 0 <= initial_dissolved_fraction < 1:
        raise ValueError("The initial_dissolved_fraction for fitting must be in [0, 1)")
    capacity = medium.dissolved_capacity_umol
    if capacity <= amount_umol:
        raise ValueError("Unsaturated sink fitting requires a dissolved capacity above the load")
    times = np.asarray(times_h, dtype=np.float64)
    observed = np.asarray(dissolved_umol, dtype=np.float64)
    if times.ndim != 1 or observed.shape != times.shape:
        raise ValueError("Supply aligned one-dimensional time and dissolved amount series")
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("Sampling times must be finite and nonnegative")
    if not np.isfinite(observed).all() or (observed < 0).any():
        raise ValueError("Measured dissolved amounts must be finite and nonnegative")
    positive = times > 0
    if np.unique(times[positive]).size < 2:
        raise ValueError("Supply observations at two or more distinct positive sampling times")
    fractions = observed / amount_umol
    partial = positive & (fractions > initial_dissolved_fraction) & (fractions < 1)
    if not partial.any() or np.ptp(observed[positive]) == 0:
        raise ValueError("The time series must resolve changing, partial dissolution")

    # Fit a dimensionless rate on the fractional amount scale so µmol and hour
    # magnitudes do not set the optimizer's absolute convergence tolerances.
    scale_h = float(np.median(times[positive]))
    scaled_times = times / scale_h
    log_remaining = np.log1p(-fractions[partial]) - math.log1p(-initial_dissolved_fraction)
    initial_rate = float(np.median(-log_remaining / scaled_times[partial]))

    def residual(rate: Vector) -> Vector:
        released = -np.expm1(-rate[0] * scaled_times)
        return np.asarray(
            initial_dissolved_fraction + (1.0 - initial_dissolved_fraction) * released - fractions,
            dtype=np.float64,
        )

    result = least_squares(residual, [initial_rate], bounds=(0.0, np.inf))
    if not result.success:
        raise RuntimeError(f"Sink dissolution fit failed: {result.message}")
    return SinkDissolutionFit(
        dissolution_per_h=float(result.x[0]) / scale_h,
        rmse_umol=float(np.sqrt(np.mean(result.fun**2))) * amount_umol,
        capacity_to_load_ratio=capacity / amount_umol,
        n_observations=len(times),
    )


@dataclass(frozen=True, kw_only=True)
class Disposition:
    """Explicit scenario parameters. Clearances use L/h and volumes use L.

    central_volume_l relates central amount to total plasma concentration.
    peripheral_volume_l is plasma-equivalent. brain_unbound_volume_l relates
    brain amount to unbound interstitial concentration. These are apparent
    distribution volumes. BBB clearances act on unbound drug on each side.
    """

    absorption_per_h: float
    gut_transit_per_h: float
    gut_escape_fraction: float
    central_volume_l: float
    peripheral_volume_l: float
    peripheral_exchange_l_h: float
    brain_unbound_volume_l: float
    bbb_in_l_h: float
    bbb_out_l_h: float
    fraction_unbound_plasma: float
    blood_to_plasma_ratio: float
    liver_blood_flow_l_h: float
    liver_unbound_clint_l_h: float
    other_plasma_clearance_l_h: float

    def __post_init__(self) -> None:
        for parameter in fields(self):
            value = getattr(self, parameter.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{parameter.name} must be finite and nonnegative")
        for name in (
            "central_volume_l",
            "peripheral_volume_l",
            "brain_unbound_volume_l",
            "blood_to_plasma_ratio",
            "liver_blood_flow_l_h",
        ):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        if self.gut_escape_fraction > 1 or self.fraction_unbound_plasma > 1:
            raise ValueError("Escape and unbound fractions must be in [0, 1]")
        if self.fraction_unbound_plasma > self.blood_to_plasma_ratio:
            raise ValueError("Implied fraction unbound in whole blood exceeds 1")

    @property
    def hepatic_blood_clearance_l_h(self) -> float:
        ml_min_per_l_h = 1000.0 / 60.0
        return (
            well_stirred_hepatic_clearance(
                blood_flow_ml_min=self.liver_blood_flow_l_h * ml_min_per_l_h,
                fraction_unbound_blood=self.fraction_unbound_plasma / self.blood_to_plasma_ratio,
                unbound_clint_ml_min=self.liver_unbound_clint_l_h * ml_min_per_l_h,
            )
            / ml_min_per_l_h
        )


@dataclass(frozen=True, kw_only=True)
class CompetitiveInhibition:
    """Constant unbound liver inhibitor concentration acting on one clearance pathway.

    The low-substrate competitive model assumes linear intrinsic clearance and
    independent unaffected pathways. Inputs refer to the same enzyme system.
    """

    affected_clint_fraction: float
    unbound_inhibitor_um: float
    unbound_ki_um: float

    def __post_init__(self) -> None:
        if not 0 <= self.affected_clint_fraction <= 1:
            raise ValueError("affected_clint_fraction must be in [0, 1]")
        if not math.isfinite(self.unbound_inhibitor_um) or self.unbound_inhibitor_um < 0:
            raise ValueError("unbound_inhibitor_um must be finite and nonnegative")
        if not math.isfinite(self.unbound_ki_um) or self.unbound_ki_um <= 0:
            raise ValueError("unbound_ki_um must be finite and positive")

    @property
    def remaining_clint_fraction(self) -> float:
        return (
            1
            - self.affected_clint_fraction
            + self.affected_clint_fraction / (1 + self.unbound_inhibitor_um / self.unbound_ki_um)
        )


@dataclass(frozen=True)
class BrainExposureDriver:
    """Continuous unbound brain concentration on the accepted PK solver steps."""

    breakpoints_h: Vector
    _state: Callable[[float | Vector], Vector] | None = field(repr=False)
    _nm_per_fraction: float = field(repr=False)

    def unbound_nm(self, time_h: float) -> float:
        """Evaluate the solver interpolant between the dose and the final reporting time."""
        if not self.breakpoints_h[0] <= time_h <= self.breakpoints_h[-1]:
            raise ValueError("Brain exposure time lies outside the simulated interval")
        if self._state is None:  # a single t=0 sample has no integration steps
            return 0.0
        fraction = float(self._state(time_h)[3])
        if not math.isfinite(fraction) or fraction < -1e-8:
            raise ArithmeticError("Continuous brain inventory failed the nonnegative check")
        return max(0.0, fraction) * self._nm_per_fraction

    def peak(self) -> tuple[float, float]:
        """Return peak unbound concentration (nM) and its earliest time (h).

        Radau's dense output is cubic on each accepted step. Four evaluations
        recover each polynomial, so endpoints and real stationary points cover
        the complete numerical trajectory, including between reporting times.
        """
        mesh = self.breakpoints_h
        if self._state is None:
            return 0.0, float(mesh[0])

        nodes = np.linspace(0.0, 1.0, 4)
        times = mesh[:-1, None] + np.diff(mesh)[:, None] * nodes
        times[:, -1] = mesh[1:]
        samples = self._state(times.ravel())[3].reshape(times.shape)
        # Fit only changes in inventory to retain precision near a flat maximum.
        # Unit-width coordinates keep the cubic fit independent of PK step size.
        x = nodes[1:]
        basis = np.column_stack((x**3, x**2, x))
        cubic = np.linalg.solve(basis, (samples[:, 1:] - samples[:, :1]).T)
        step_index = np.arange(len(mesh), dtype=np.float64)
        derivative = PPoly(cubic * np.array([[3.0], [2.0], [1.0]]), step_index, extrapolate=False)
        roots = derivative.roots(discontinuity=False)
        # Flat intervals have no isolated stationary point. Their endpoints are included.
        interior = np.interp(roots[np.isfinite(roots)], step_index, mesh)
        candidates = np.union1d(mesh, interior)
        fractions = self._state(candidates)[3]
        if not np.isfinite(fractions).all() or fractions.min() < -1e-8:
            raise ArithmeticError("Continuous brain inventory failed the nonnegative check")
        nonnegative = np.maximum(fractions, 0.0)
        maximum = float(nonnegative.max())
        tied = nonnegative >= maximum * (1.0 - 8 * np.finfo(np.float64).eps)
        first = int(np.argmax(tied))
        return float(nonnegative[first]) * self._nm_per_fraction, float(candidates[first])


@dataclass(frozen=True)
class ExposureProfile:
    """Concentrations in µM, AUC in µM·h and inventories in µmol.

    Gut totals include the gastric and intestinal inventories of dissolved (free
    plus complexed) and solid parent. All GI mass is counted once as gut_umol.
    Simulations without a gastric stage return zero for both gastric inventories.
    """

    time_h: Vector
    gut_umol: Vector
    central_umol: Vector
    peripheral_umol: Vector
    brain_umol: Vector
    transit_loss_umol: Vector
    first_pass_loss_umol: Vector
    systemic_loss_umol: Vector
    total_plasma_um: Vector
    unbound_plasma_um: Vector
    unbound_brain_um: Vector
    unbound_plasma_auc_um_h: Vector
    unbound_brain_auc_um_h: Vector
    gut_dissolved_umol: Vector | None = None
    gut_solid_umol: Vector | None = None
    gastric_dissolved_umol: Vector | None = None
    gastric_solid_umol: Vector | None = None
    intestinal_dissolved_umol: Vector | None = None
    intestinal_solid_umol: Vector | None = None
    brain_driver: BrainExposureDriver | None = None

    @property
    def accounted_umol(self) -> Vector:
        return (
            self.gut_umol
            + self.central_umol
            + self.peripheral_umol
            + self.brain_umol
            + self.transit_loss_umol
            + self.first_pass_loss_umol
            + self.systemic_loss_umol
        )


def simulate_oral(
    disposition: Disposition,
    medium: OralMedium,
    times_h: Sequence[float],
    *,
    amount_umol: float,
    inhibition: CompetitiveInhibition | None = None,
    formulation: OralFormulation | None = None,
    gastric_stage: GastricStage | None = None,
    dense_brain: bool = False,
) -> ExposureProfile:
    """Simulate one initial oral load of unchanged parent at t=0.

    The default intestinal medium equilibrates instantaneously. A formulation
    sets finite intestinal exchange rates. With a gastric stage, the dose
    starts in the stomach, using the gastric formulation's initial dissolved
    fraction. The intestine then starts empty, and liquid and solid arrive at
    their respective emptying rates. Two integrals accumulate free plasma and
    brain exposure independently of sampling density. The solver works in
    fractions of the input for scale-independent tolerances. Set dense_brain to
    retain the continuous unbound concentration for a downstream kinetics model.
    """
    t = np.array(times_h, dtype=np.float64)
    if t.ndim != 1 or not t.size or not np.isfinite(t).all() or t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError("times_h must be finite, nonnegative and strictly increasing")
    if not math.isfinite(amount_umol) or amount_umol <= 0:
        raise ValueError("amount_umol must be finite and positive")
    p = disposition
    if inhibition is not None:
        p = replace(
            p,
            liver_unbound_clint_l_h=p.liver_unbound_clint_l_h * inhibition.remaining_clint_fraction,
        )
    hepatic_cl = p.hepatic_blood_clearance_l_h
    available = p.gut_escape_fraction * (1 - hepatic_cl / p.liver_blood_flow_l_h)
    systemic_cl = hepatic_cl * p.blood_to_plasma_ratio + p.other_plasma_clearance_l_h
    capacity_umol = medium.dissolved_capacity_umol
    capacity_fraction = capacity_umol / amount_umol
    gastric_capacity_fraction = (
        gastric_stage.medium.dissolved_capacity_umol / amount_umol
        if gastric_stage is not None
        else 0.0
    )

    def rates(_time: float, y: Vector) -> Vector:
        gut_pool, central, periph, brain = y[:4]
        dy = np.zeros_like(y)
        arriving_liquid = arriving_solid = 0.0
        if gastric_stage is not None:
            gastric_liquid, gastric_solid = y[10:12]
            arriving_liquid = gastric_stage.liquid_emptying_per_h * gastric_liquid
            arriving_solid = gastric_stage.solid_emptying_per_h * gastric_solid
            gastric_transfer = gastric_stage.formulation._net_dissolution(
                gastric_liquid, gastric_solid, gastric_capacity_fraction
            )
            dy[10] = gastric_transfer - arriving_liquid
            dy[11] = -gastric_transfer - arriving_solid
        gut_transit = p.gut_transit_per_h * gut_pool
        total_transit = gut_transit
        pool_umol = max(gut_pool, 0.0) * amount_umol
        if formulation is None:
            free_umol = medium.free_amount_umol(pool_umol)
            gut_in = arriving_liquid + arriving_solid
        else:
            # y[0] is the dissolved pool and y[9] the solid inventory.
            free_umol = medium._free_in_liquid_umol(pool_umol)
            solid = y[9]
            phase_transfer = formulation._net_dissolution(gut_pool, solid, capacity_fraction)
            solid_transit = p.gut_transit_per_h * solid
            total_transit += solid_transit
            dy[9] = arriving_solid - phase_transfer - solid_transit
            gut_in = arriving_liquid + phase_transfer
        # Concentrations below are per unit input during integration.
        cp = central / p.central_volume_l
        cu_p = p.fraction_unbound_plasma * cp
        cu_b = brain / p.brain_unbound_volume_l
        absorb = p.absorption_per_h * free_umol / amount_umol
        tissue_flux = p.peripheral_exchange_l_h * (cp - periph / p.peripheral_volume_l)
        brain_flux = p.bbb_in_l_h * cu_p - p.bbb_out_l_h * cu_b
        loss = systemic_cl * cp
        dy[:9] = (
            gut_in - absorb - gut_transit,
            absorb * available - tissue_flux - brain_flux - loss,
            tissue_flux,
            brain_flux,
            total_transit,
            absorb * (1 - available),
            loss,
            cu_p,
            cu_b,
        )
        return dy

    if gastric_stage is not None:
        initial = np.zeros(12)
        initial[10] = gastric_stage.formulation.initial_dissolved_fraction
        initial[11] = 1.0 - initial[10]
    elif formulation is None:
        initial = np.zeros(9)
        initial[0] = 1.0
    else:
        initial = np.zeros(10)
        initial[0] = formulation.initial_dissolved_fraction
        initial[9] = 1.0 - initial[0]
    brain_driver = None
    if t[-1] == 0:
        values = initial[:, None]
        if dense_brain:
            brain_driver = BrainExposureDriver(np.array([0.0]), None, 0.0)
    else:
        sol = solve_ivp(
            rates,
            (0, t[-1]),
            initial,
            t_eval=None if dense_brain else t,
            dense_output=dense_brain,
            method="Radau",
            rtol=1e-8,
            atol=1e-11,
        )
        if not sol.success:
            raise RuntimeError(f"Exposure integration failed: {sol.message}")
        if dense_brain:
            values = np.asarray(sol.sol(t))
            mesh = np.array(sol.t, dtype=np.float64, copy=True)
            mesh.flags.writeable = False
            brain_driver = BrainExposureDriver(
                mesh, sol.sol, amount_umol / p.brain_unbound_volume_l * 1000
            )
        else:
            values = np.asarray(sol.y)
    inventory = values[:7].sum(axis=0)
    if formulation is not None:
        inventory += values[9]
    if gastric_stage is not None:
        inventory += values[10] + values[11]
    if (
        not np.isfinite(values).all()
        or values.min() < -1e-8
        or np.max(np.abs(inventory - 1.0)) > 1e-7
    ):
        raise ArithmeticError("Integration failed the nonnegative mass-balance check")
    amounts = np.maximum(values, 0.0) * amount_umol
    gut, central, periph, brain, transit, pre, systemic, auc_p, auc_b = amounts[:9]
    if formulation is None:
        dissolved = np.minimum(gut, capacity_umol)
        solid = gut - dissolved
    else:
        dissolved, solid = gut, amounts[9]
        gut = dissolved + solid
    if gastric_stage is None:
        gastric_dissolved = np.zeros_like(gut)
        gastric_solid = np.zeros_like(gut)
        all_dissolved, all_solid = dissolved, solid
    else:
        gastric_dissolved, gastric_solid = amounts[10:12]
        all_dissolved = dissolved + gastric_dissolved
        all_solid = solid + gastric_solid
        gut = gut + gastric_dissolved + gastric_solid
    cp = central / p.central_volume_l
    return ExposureProfile(
        t,
        gut,
        central,
        periph,
        brain,
        transit,
        pre,
        systemic,
        cp,
        cp * p.fraction_unbound_plasma,
        brain / p.brain_unbound_volume_l,
        auc_p,
        auc_b,
        gut_dissolved_umol=all_dissolved,
        gut_solid_umol=all_solid,
        gastric_dissolved_umol=gastric_dissolved,
        gastric_solid_umol=gastric_solid,
        intestinal_dissolved_umol=dissolved,
        intestinal_solid_umol=solid,
        brain_driver=brain_driver,
    )
