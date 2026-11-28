"""Offline worklists and paired LC–MS observations from human plasma equilibrium dialysis."""

from __future__ import annotations

import random
from statistics import mean, stdev
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator
from rdkit import Chem

Text = Annotated[str, Field(min_length=1)]
Well = Annotated[str, Field(pattern=r"^[A-H](0[1-9]|1[0-2])$")]
Positive = Annotated[float, Field(gt=0)]
Nonnegative = Annotated[float, Field(ge=0)]
Fraction = Annotated[float, Field(gt=0, le=1)]
Role = Literal["candidate", "reference", "blank"]


class _Record(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, frozen=True, str_strip_whitespace=True
    )


class Compound(_Record):
    compound_id: Text
    smiles: Text

    @field_validator("smiles")
    @classmethod
    def canonical_structure(cls, value: str) -> str:
        mol = Chem.MolFromSmiles(value)
        if mol is None or not mol.GetNumHeavyAtoms():
            raise ValueError("A valid compound structure is required")
        return str(Chem.MolToSmiles(mol, isomericSmiles=True))


class DialysisWell(_Record):
    well: Well
    role: Role


class DialysisPlan(_Record):
    """Each coordinate denotes one paired plasma/buffer device position."""

    plate_id: Text
    candidate: Compound
    reference: Compound
    wells: tuple[DialysisWell, ...]

    @model_validator(mode="after")
    def validate_layout(self) -> DialysisPlan:
        if (
            self.candidate.compound_id == self.reference.compound_id
            or self.candidate.smiles == self.reference.smiles
        ):
            raise ValueError("The reference must be a distinct compound")
        if len({w.well for w in self.wells}) != len(self.wells):
            raise ValueError("Duplicate device position")
        for role in ("candidate", "reference", "blank"):
            if sum(w.role == role for w in self.wells) < 2:
                raise ValueError(f"At least two {role} pairs are required")
        return self


def plan_dialysis(
    candidate: Compound, reference: Compound, *, plate_id: str, replicates: int = 3, seed: int = 0
) -> DialysisPlan:
    """Randomize candidate, reference and blank pairs over a 96-position device."""
    if not 2 <= replicates <= 32:
        raise ValueError("replicates must be between 2 and 32 per role")
    positions = [f"{row}{col:02d}" for row in "ABCDEFGH" for col in range(1, 13)]
    positions = random.Random(seed).sample(positions, 3 * replicates)
    roles: tuple[Role, ...] = ("candidate", "reference", "blank")
    wells = [
        DialysisWell(well=positions[i * replicates + j], role=role)
        for i, role in enumerate(roles)
        for j in range(replicates)
    ]
    return DialysisPlan(
        plate_id=plate_id,
        candidate=candidate,
        reference=reference,
        wells=tuple(sorted(wells, key=lambda w: w.well)),
    )


class DialysisLimits(_Record):
    """Prequalified protocol-specific acceptance criteria, supplied by the laboratory."""

    source: Text
    quantification_limit_um: Positive
    recovery_low: Positive
    recovery_high: Positive
    reference_fu_low: Fraction
    reference_fu_high: Fraction
    max_replicate_cv: Positive

    @model_validator(mode="after")
    def ordered_windows(self) -> DialysisLimits:
        if self.recovery_low > self.recovery_high or self.reference_fu_low > self.reference_fu_high:
            raise ValueError("QC windows must be ordered")
        return self


class DialysisReadout(_Record):
    """Raw parent concentrations and actual volumes for one paired position."""

    well: Well
    compound_id: Text | None
    initial_plasma_um: Nonnegative
    final_plasma_um: Nonnegative
    final_buffer_um: Nonnegative
    initial_plasma_ul: Positive
    final_plasma_ul: Positive
    final_buffer_ul: Positive


class DialysisRun(_Record):
    """Join an executed paired readout to its preassigned plate and material lots."""

    plan: DialysisPlan
    observed_plate_id: Text
    run_id: Text
    protocol_version: Text
    instrument_id: Text
    completed_at: AwareDatetime
    raw_data_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    raw_data_uri: Text
    candidate_lot: Text
    reference_lot: Text
    plasma_lot: Text
    matrix: Literal["human_plasma"]
    ph: Annotated[float, Field(ge=0, le=14)]
    temperature_c: float
    equilibrium_verified: bool
    limits: DialysisLimits
    observations: tuple[DialysisReadout, ...]


class PairResult(_Record):
    well: Well
    recovered_fraction: float | None
    buffer_to_plasma: float | None


class DialysisResult(_Record):
    run_id: str
    candidate: Compound
    candidate_lot: str
    raw_data_uri: str
    fraction_unbound_plasma: float | None
    candidate_cv: float | None
    reference_cv: float | None
    reasons: tuple[str, ...]
    pairs: tuple[PairResult, ...]


def evaluate_dialysis(run: DialysisRun) -> DialysisResult:
    """Release a plasma unbound fraction only when all identity and QC checks pass."""
    limits = run.limits
    reasons: list[str] = []
    if run.plan.plate_id != run.observed_plate_id:
        reasons.append("Observed plate ID differs from the planned plate")
    if not run.equilibrium_verified:
        reasons.append("Equilibration has not been verified for this run")
    roles = {w.well: w.role for w in run.plan.wells}
    if sorted(o.well for o in run.observations) != sorted(roles):
        reasons.append("Readouts must cover every planned position exactly once")
    ratios: dict[str, list[float]] = {"candidate": [], "reference": []}
    pairs: list[PairResult] = []
    for obs in run.observations:
        role = roles.get(obs.well)
        if role is None:
            continue
        expected = {
            "candidate": run.plan.candidate.compound_id,
            "reference": run.plan.reference.compound_id,
            "blank": None,
        }[role]
        if obs.compound_id != expected:
            reasons.append(f"{obs.well} sample identity differs from the plan")
        recovery = ratio = None
        if role == "blank":
            if (
                obs.initial_plasma_um != 0
                or max(obs.final_plasma_um, obs.final_buffer_um) >= limits.quantification_limit_um
            ):
                reasons.append(f"{obs.well} blank contains quantifiable parent or was spiked")
        else:
            if (
                obs.initial_plasma_um < limits.quantification_limit_um
                or min(obs.final_plasma_um, obs.final_buffer_um) < limits.quantification_limit_um
            ):
                reasons.append(f"{obs.well} parent concentration below the quantification limit")
            else:
                recovery = (
                    obs.final_plasma_um * obs.final_plasma_ul
                    + obs.final_buffer_um * obs.final_buffer_ul
                ) / (obs.initial_plasma_um * obs.initial_plasma_ul)
                ratio = obs.final_buffer_um / obs.final_plasma_um
                if not limits.recovery_low <= recovery <= limits.recovery_high:
                    reasons.append(f"{obs.well} mass recovery outside protocol limits")
                if ratio > 1 or (
                    role == "reference"
                    and not limits.reference_fu_low <= ratio <= limits.reference_fu_high
                ):
                    reasons.append(f"{obs.well} unbound fraction outside acceptance limits")
                ratios[role].append(ratio)
        pairs.append(PairResult(well=obs.well, recovered_fraction=recovery, buffer_to_plasma=ratio))

    cvs = {
        role: stdev(vals) / mean(vals) if len(vals) >= 2 else None for role, vals in ratios.items()
    }
    for group, cv in cvs.items():
        if cv is None or cv > limits.max_replicate_cv:
            reasons.append(f"{group} replicate precision outside protocol limits")
    return DialysisResult(
        run_id=run.run_id,
        candidate=run.plan.candidate,
        candidate_lot=run.candidate_lot,
        raw_data_uri=run.raw_data_uri,
        fraction_unbound_plasma=None if reasons else mean(ratios["candidate"]),
        candidate_cv=cvs["candidate"],
        reference_cv=cvs["reference"],
        reasons=tuple(reasons),
        pairs=tuple(pairs),
    )
