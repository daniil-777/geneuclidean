"""Offline analysis of paired, sink-condition transport through barrier inserts.

The endpoint is apparent permeability of total parent in the specified cell model
and media. The two directions may reflect active transport, passive diffusion,
binding and pH partitioning. Physiological translation requires separate evidence.
See https://doi.org/10.3390/pharmaceutics13081146 for bidirectional insert
assays and https://doi.org/10.3390/pharmaceutics13060892 for a BBB model and
its integrity controls.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from hashlib import sha256
from statistics import mean, stdev
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from scipy.stats import linregress

from .dialysis import Compound

Text = Annotated[str, Field(min_length=1)]
Positive = Annotated[float, Field(gt=0)]
Nonnegative = Annotated[float, Field(ge=0)]
Fraction = Annotated[float, Field(ge=0, le=1)]
PH = Annotated[float, Field(ge=0, le=14)]
Direction = Literal["A_to_B", "B_to_A"]
Role = Literal["candidate", "reference", "candidate_blank", "reference_blank"]


class _Record(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, str_strip_whitespace=True
    )


class BarrierRange(_Record):
    low: Nonnegative
    high: Positive

    @model_validator(mode="after")
    def ordered(self) -> BarrierRange:
        if self.low > self.high:
            raise ValueError("The lower acceptance limit exceeds the upper limit")
        return self

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


class BarrierLimits(_Record):
    """Laboratory-qualified windows for this model, protocol, media and analytes.

    Supply a quantification limit qualified for both compartments for each analyte.
    Net TEER is blank-insert-corrected, area-normalized resistance in ohm cm².
    """

    source: Text
    candidate_lloq_um: Positive
    reference_lloq_um: Positive
    aqueous_recovery: BarrierRange
    reference_ab_cm_s: BarrierRange
    reference_efflux_ratio: BarrierRange
    min_net_teer_ohm_cm2: Positive
    max_loading_error_fraction: Fraction
    max_donor_change_fraction: Fraction
    max_sink_ratio: Annotated[float, Field(gt=0, lt=1)]
    min_fit_r_squared: Fraction
    max_replicate_cv: Nonnegative


class BarrierWell(_Record):
    """A preassigned physical insert. A and B denote apical and basolateral."""

    well: Text
    role: Role
    direction: Direction


class BarrierSample(_Record):
    """Pre-withdrawal total-parent concentrations in µM, with actual aliquot volumes.

    Each replacement is analyte-free. A null concentration is explicitly below the
    validated quantification limit. A missing reading must not be entered as null.
    """

    time_min: Nonnegative
    donor_um: Nonnegative | None
    receiver_um: Nonnegative | None
    donor_sample_ul: Nonnegative = 0
    receiver_sample_ul: Nonnegative = 0
    donor_replacement_ul: Nonnegative = 0
    receiver_replacement_ul: Nonnegative = 0


class BarrierTrace(_Record):
    """Measured samples from the same insert, with the independently recorded identity."""

    well: Text
    direction: Direction
    analyte: Compound
    material_lot: Text | None
    initial_donor_ul: Positive
    initial_receiver_ul: Positive
    net_teer_before_ohm_cm2: float | None = None
    net_teer_after_ohm_cm2: float | None = None
    samples: Annotated[tuple[BarrierSample, ...], Field(min_length=1)]

    def samples_with_volumes(self) -> Iterator[tuple[BarrierSample, float, float]]:
        """Yield each observation and the two compartment volumes immediately before its draws."""
        donor, receiver = self.initial_donor_ul, self.initial_receiver_ul
        for sample in self.samples:
            yield sample, donor, receiver
            donor += sample.donor_replacement_ul - sample.donor_sample_ul
            receiver += sample.receiver_replacement_ul - sample.receiver_sample_ul

    @model_validator(mode="after")
    def check_inventory(self) -> BarrierTrace:
        if self.samples[0].time_min != 0:
            raise ValueError("Each insert needs a time-zero sample before the first withdrawal")
        for i, (sample, donor, receiver) in enumerate(self.samples_with_volumes()):
            if i and sample.time_min <= self.samples[i - 1].time_min:
                raise ValueError("Sample times must strictly increase within each insert")
            if min(donor, receiver) <= 0:
                raise ValueError("Both compartments must contain liquid for the following interval")
            if sample.donor_sample_ul > donor or sample.receiver_sample_ul > receiver:
                raise ValueError("A withdrawal exceeds the volume present in the compartment")
        return self


class BarrierConditions(_Record):
    """Conditions shared by both transport directions. Concentrations are total parent."""

    cell_model: Text
    cell_lot: Text
    apical_medium: Text
    basolateral_medium: Text
    apical_ph: PH
    basolateral_ph: PH
    temperature_c: float
    area_cm2: Positive
    fit_start_min: Nonnegative
    fit_end_min: Positive

    @model_validator(mode="after")
    def check_fit_window(self) -> BarrierConditions:
        if self.fit_start_min >= self.fit_end_min:
            raise ValueError("The prespecified fit window must have positive duration")
        return self


class BarrierRun(BarrierConditions):
    """Paired transport plate with a positive reference and an unspiked blank per analyte.

    Initial receiver medium is prepared free of parent. Both directions for each
    analyte use the same nominal donor concentration, temperature and fit window.
    The volume model describes a well-mixed static insert with documented
    withdrawal and replacement events. Initial volumes are the actual fill volumes.
    """

    run_id: Text
    plate_id: Text
    observed_plate_id: Text
    protocol_version: Text
    instrument_id: Text
    completed_at: AwareDatetime
    raw_data_uri: Text
    raw_data_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    candidate: Compound
    candidate_lot: Text
    candidate_nominal_donor_um: Positive
    reference: Compound
    reference_lot: Text
    reference_nominal_donor_um: Positive
    limits: BarrierLimits
    layout: tuple[BarrierWell, ...]
    traces: tuple[BarrierTrace, ...]

    @model_validator(mode="after")
    def check_plan(self) -> BarrierRun:
        if (
            self.candidate.compound_id == self.reference.compound_id
            or self.candidate.smiles == self.reference.smiles
        ):
            raise ValueError("The reference requires a distinct compound ID and structure")
        if len({w.well for w in self.layout}) != len(self.layout):
            raise ValueError("Duplicate insert in the planned layout")
        for role in ("candidate", "reference"):
            for direction in ("A_to_B", "B_to_A"):
                if sum(w.role == role and w.direction == direction for w in self.layout) < 2:
                    raise ValueError(
                        f"Plan at least two independent {role} inserts for {direction}"
                    )
            if not any(w.role == f"{role}_blank" for w in self.layout):
                raise ValueError(f"Plan an unspiked matrix blank monitored for {role}")
        return self


class BarrierMassPoint(_Record):
    time_min: float
    donor_volume_ul: float
    receiver_volume_ul: float
    cumulative_receiver_pmol: float | None


class BarrierFit(_Record):
    """Per-insert diagnostics, including failed or censored observations."""

    well: str
    role: Role
    direction: Direction
    mass_points: tuple[BarrierMassPoint, ...] = ()
    aqueous_recovery_fraction: float | None = None
    max_receiver_to_donor_ratio: float | None = None
    max_donor_change_fraction: float | None = None
    fit_point_count: int = 0
    slope_pmol_min: float | None = None
    r_squared: float | None = None
    papp_cm_s: float | None = None
    reasons: tuple[str, ...] = ()


class BarrierDirectionSummary(_Record):
    role: Literal["candidate", "reference"]
    direction: Direction
    n_inserts: int
    mean_papp_cm_s: float | None
    replicate_cv: float | None


class BarrierResult(_Record):
    run_id: str
    candidate: Compound
    candidate_lot: str
    protocol_version: str
    cell_model: str
    assay_conditions: BarrierConditions
    qc_source: str
    raw_data_uri: str
    raw_data_sha256: str | None
    input_sha256: str
    papp_ab_cm_s: float | None
    papp_ba_cm_s: float | None
    efflux_ratio: float | None
    reference_efflux_ratio: float | None
    reasons: tuple[str, ...]
    inserts: tuple[BarrierFit, ...]
    directions: tuple[BarrierDirectionSummary, ...]


def _removed(previous: float | None, sample_ul: float, concentration: float | None) -> float | None:
    if sample_ul == 0 or previous is None:
        return previous
    return None if concentration is None else previous + sample_ul * concentration


def _fit_insert(run: BarrierRun, well: BarrierWell, trace: BarrierTrace) -> BarrierFit:
    limits = run.limits
    candidate = well.role.startswith("candidate")
    blank = well.role.endswith("_blank")
    compound = run.candidate if candidate else run.reference
    lot = run.candidate_lot if candidate else run.reference_lot
    lloq = limits.candidate_lloq_um if candidate else limits.reference_lloq_um
    nominal = run.candidate_nominal_donor_um if candidate else run.reference_nominal_donor_um
    reasons = []
    if trace.analyte != compound or trace.material_lot != (None if blank else lot):
        reasons.append("Analyte identity or dosed material lot differs from the plan")
    if trace.direction != well.direction:
        reasons.append("Observed transport direction differs from the plan")
    if trace.samples[-1].time_min < run.fit_end_min:
        reasons.append("Observations do not cover the fit window")
    if blank:
        if any(
            c is not None and c >= lloq for s in trace.samples for c in (s.donor_um, s.receiver_um)
        ):
            reasons.append("Unspiked matrix blank contains quantifiable parent")
        return BarrierFit(
            well=well.well, role=well.role, direction=well.direction, reasons=tuple(reasons)
        )

    teers = (trace.net_teer_before_ohm_cm2, trace.net_teer_after_ohm_cm2)
    if any(teer is None or teer < limits.min_net_teer_ohm_cm2 for teer in teers):
        reasons.append("Pre- or post-assay barrier integrity failed the net TEER criterion")
    initial_donor = trace.samples[0].donor_um
    if initial_donor is None or initial_donor < lloq:
        initial_donor = None
        reasons.append("Initial donor concentration is below the quantification limit")
    elif abs(initial_donor / nominal - 1) > limits.max_loading_error_fraction:
        reasons.append("Measured donor loading differs from the nominal concentration")

    removed_donor: float | None = 0.0
    removed_receiver: float | None = 0.0
    points: list[BarrierMassPoint] = []
    times, amounts = [], []
    recovery = None
    max_sink, max_donor_change = 0.0, 0.0
    for i, (sample, donor_volume, receiver_volume) in enumerate(trace.samples_with_volumes()):
        donor = sample.donor_um if sample.donor_um is not None and sample.donor_um >= lloq else None
        receiver = (
            sample.receiver_um
            if sample.receiver_um is not None and sample.receiver_um >= lloq
            else None
        )
        if i == 0:
            if receiver is not None:
                reasons.append(
                    "Prepared parent-free receiver contains quantifiable parent at baseline"
                )
            else:
                receiver = 0.0
        cumulative = (
            None
            if receiver is None or removed_receiver is None
            else receiver_volume * receiver + removed_receiver
        )
        points.append(
            BarrierMassPoint(
                time_min=sample.time_min,
                donor_volume_ul=donor_volume,
                receiver_volume_ul=receiver_volume,
                cumulative_receiver_pmol=cumulative,
            )
        )
        # The prepared parent-free receiver defines the initial inventory. Its
        # sub-LLOQ baseline is not a quantitative point for a free-intercept fit.
        if i > 0 and run.fit_start_min <= sample.time_min <= run.fit_end_min:
            if cumulative is None or donor is None or receiver is None:
                reasons.append("Fit window includes unquantifiable concentration or sampled mass")
            else:
                times.append(sample.time_min)
                amounts.append(cumulative)
                max_sink = max(max_sink, receiver / donor)
                if initial_donor is not None:
                    max_donor_change = max(max_donor_change, abs(donor / initial_donor - 1))
        if i == len(trace.samples) - 1:
            if (
                initial_donor is None
                or cumulative is None
                or donor is None
                or removed_donor is None
            ):
                reasons.append("Final fluid-phase mass recovery is unquantifiable")
            else:
                recovery = (donor * donor_volume + removed_donor + cumulative) / (
                    initial_donor * trace.initial_donor_ul
                )
                if not limits.aqueous_recovery.contains(recovery):
                    reasons.append(
                        "Final fluid-phase mass recovery lies outside the qualified range"
                    )
        removed_donor = _removed(removed_donor, sample.donor_sample_ul, donor)
        removed_receiver = _removed(removed_receiver, sample.receiver_sample_ul, receiver)

    if max_sink > limits.max_sink_ratio:
        reasons.append("Receiver-to-donor concentration exceeds the sink limit")
    if max_donor_change > limits.max_donor_change_fraction:
        reasons.append("Donor concentration changed beyond the sink-fit protocol limit")
    slope = r_squared = papp = None
    if len(times) < 3:
        reasons.append("Fewer than three quantified postdose times in the prespecified fit window")
    else:
        fit = linregress(times, amounts)
        slope = float(fit.slope)
        if max(amounts) != min(amounts):
            r_squared = float(fit.rvalue) ** 2
        if slope <= 0 or r_squared is None or r_squared < limits.min_fit_r_squared:
            reasons.append(
                "A positive linear receiver flux was not resolved within protocol limits"
            )
        if initial_donor is not None and slope > 0:
            # (pmol/min) / (pmol/µL) gives µL/min; 1 µL/min = 1/60000 cm³/s.
            papp = slope / (60_000 * run.area_cm2 * initial_donor)
    return BarrierFit(
        well=well.well,
        role=well.role,
        direction=well.direction,
        mass_points=tuple(points),
        aqueous_recovery_fraction=recovery,
        max_receiver_to_donor_ratio=max_sink if times else None,
        max_donor_change_fraction=max_donor_change if times and initial_donor is not None else None,
        fit_point_count=len(times),
        slope_pmol_min=slope,
        r_squared=r_squared,
        papp_cm_s=papp,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def evaluate_barrier(run: BarrierRun) -> BarrierResult:
    """Fit each insert separately and release the ratio of directional mean permeabilities.

    Cumulative receiver amount is the pre-draw amount plus all parent previously
    withdrawn from that compartment. Final fluid recovery also includes removed
    donor parent. Apparent permeability uses a free-intercept least-squares slope
    from at least three quantified postdose observations and the measured initial
    donor concentration, subject to sink and donor QC.
    Reference and candidate replicates are independent physical inserts.
    """

    reasons = []
    if run.observed_plate_id != run.plate_id:
        reasons.append("Observed plate identity differs from the plan")
    expected = {well.well: well for well in run.layout}
    counts = Counter(t.well for t in run.traces)
    if counts != Counter(expected.keys()):
        reasons.append("Trace IDs must cover every planned insert exactly once")
    fits = tuple(
        _fit_insert(run, expected[t.well], t)
        for t in run.traces
        if t.well in expected and counts[t.well] == 1
    )
    reasons.extend(f"{fit.well}: {reason}" for fit in fits for reason in fit.reasons)

    summaries = []
    means: dict[tuple[str, str], float] = {}
    for role in ("candidate", "reference"):
        for direction in ("A_to_B", "B_to_A"):
            values = [
                f.papp_cm_s
                for f in fits
                if f.role == role
                and f.direction == direction
                and not f.reasons
                and f.papp_cm_s is not None
            ]
            avg = mean(values) if values else None
            cv = stdev(values) / avg if len(values) >= 2 and avg is not None else None
            if cv is None or cv > run.limits.max_replicate_cv:
                reasons.append(f"{role} {direction} replicate precision is outside protocol limits")
            if avg is not None:
                means[role, direction] = avg
            summaries.append(
                BarrierDirectionSummary(
                    role=role,
                    direction=direction,
                    n_inserts=len(values),
                    mean_papp_cm_s=avg,
                    replicate_cv=cv,
                )
            )

    ref_ab, ref_ba = means.get(("reference", "A_to_B")), means.get(("reference", "B_to_A"))
    reference_er = ref_ba / ref_ab if ref_ab is not None and ref_ba is not None else None
    if ref_ab is None or not run.limits.reference_ab_cm_s.contains(ref_ab):
        reasons.append("Reference A-to-B permeability is outside the qualified range")
    if reference_er is None or not run.limits.reference_efflux_ratio.contains(reference_er):
        reasons.append("Reference efflux ratio is outside the qualified range")
    ab = means.get(("candidate", "A_to_B")) if not reasons else None
    ba = means.get(("candidate", "B_to_A")) if not reasons else None
    return BarrierResult(
        run_id=run.run_id,
        candidate=run.candidate,
        candidate_lot=run.candidate_lot,
        protocol_version=run.protocol_version,
        cell_model=run.cell_model,
        assay_conditions=BarrierConditions.model_validate(
            run.model_dump(include=set(BarrierConditions.model_fields))
        ),
        qc_source=run.limits.source,
        raw_data_uri=run.raw_data_uri,
        raw_data_sha256=run.raw_data_sha256,
        input_sha256=sha256(run.model_dump_json().encode()).hexdigest(),
        papp_ab_cm_s=ab,
        papp_ba_cm_s=ba,
        efflux_ratio=ba / ab if ab is not None and ba is not None else None,
        reference_efflux_ratio=reference_er,
        reasons=tuple(reasons),
        inserts=fits,
        directions=tuple(summaries),
    )
