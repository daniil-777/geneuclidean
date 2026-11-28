"""Curated 24-Target *Homo sapiens* Tri-Disorder CNS Atlas + Matched Counter-Targets.

Expands GenEuclidean's v3.1 12-target Tourette Syndrome & psychiatric registry to the
**v4.0 24-Target Tri-Disorder *Homo sapiens* Structural Pharmacology Atlas**:

- **12 Core v3.1 Tourette & Psychiatric Targets (`CNS_TARGET_REGISTRY`, preserved 100%
  backward-compatible):** ``DRD1``, ``VMAT2``, ``ADRA2A``, ``DRD2``, ``HRH3``, ``PDE10A``,
  ``CHRM4``, ``HTR2A``, ``SERT``, ``DRD3``, ``TAAR1``, ``DAT``.
- **6 New Tourette & Schizophrenia Targets:** ``CHRM1`` (``9UCP`` active, ``-40`` TM6/7
  offset documented), ``HTR1A`` (``9VJE`` active buspirone), ``GLYT1`` (``SLC6A9``
  ``9J8D`` iclepertin), ``MGLL`` (``9F8D`` 1.14 A reversible), ``PDE1B`` (``5UOY`` 1.82 A),
  ``GPR52`` (``8HMP`` active Gs).
- **6 New Alzheimer's Disease Targets:** ``ACHE`` (``4EY7`` donepezil), ``OGA``
  (``MGEA5``, ``CHEMBL5921``, ``9BA8`` ceperognastat), ``NLRP3`` (``9GU4`` ``NP3-253``
  retaining ``ADP702``), ``MAOB`` (``2V5Z`` safinamide retaining ``FAD600``), ``GSK3B``
  (``7SXJ`` ``BRD0705``), ``BACE1`` (``6EQM`` umibecestat + ``5HU1`` ``+61`` offset aliases).
- **7 Matched Structural Selectivity Counter-Targets (`V4_COUNTER_TARGET_REGISTRY`):**
  ``BCHE`` (``4BDS``), ``MAOA`` (``2Z5X`` retaining ``FAD600``), ``CDK2`` (``6Q4G``),
  ``BACE2`` (``7D5B``), ``HTR2B`` (``7SRQ``), ``HRH1`` (``8X5Y``), ``CHRM3`` (``8EA0``).

Every UniProt accession, human single-protein ChEMBL Target ID, RCSB PDB accession,
cofactor requirement, conformational state, canonical 3D anchor residue, isoform alias,
and RDKit canonical clinical SMILES has been live-verified. All ligands generated or scored
against these targets are strictly *in silico* research hypotheses and are NOT clinically
tested medicines or medical advice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import numpy as np
import requests
from Bio.PDB import MMCIFParser, PDBParser
from rdkit import Chem

from geneuclidean.data.featurizer import (
    PocketFeatures,
    extract_pocket_from_files,
    featurize_pdb_pocket_inference,
)
from geneuclidean.exceptions import RegistryError, TargetNotFoundError

DEFAULT_CNS_STRUCTURES_DIR = Path("data/cns_neuro/structures")
DEFAULT_V4_STRUCTURES_DIR = Path("data/cns_neuro.nosync/v4_expansion/structures")
DEFAULT_TS_CIRCUIT_STRUCTURES_DIR = Path("data/cns_neuro.nosync/ts_circuit_expansion/structures")
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"

DiseaseArea = Literal["tourette", "schizophrenia", "psychiatry", "alzheimer", "counter"]
MoAClass = Literal[
    "ANTAGONIST",
    "INVERSE_AGONIST",
    "AGONIST",
    "PARTIAL_AGONIST",
    "INHIBITOR",
    "PAM",
    "NAM",
]


@dataclass(frozen=True)
class ClinicalReferenceDrug:
    """Verified clinical or tool reference ligand for a CNS target."""

    name: str
    canonical_smiles: str
    clinical_status: str
    role: str


@dataclass(frozen=True)
class CNSTargetSpec:
    """Verified human CNS drug target specification with 3D templates and hotspot anchors."""

    target_id: str
    gene_symbol: str
    name: str
    target_class: str
    uniprot_id: str
    chembl_id: str
    indications: tuple[str, ...]
    therapeutic_moa: str
    desired_state: str
    primary_pdb_id: str
    secondary_pdb_ids: tuple[str, ...]
    primary_anchor_residue: str
    hotspot_residues: tuple[str, ...]
    hotspot_isoform_aliases: tuple[str, ...]
    reference_drugs: tuple[ClinicalReferenceDrug, ...]
    notes: str = ""
    disease_area: DiseaseArea = "psychiatry"
    moa_class: MoAClass = "INHIBITOR"
    keep_cofactors: tuple[str, ...] = ()
    counter_targets: tuple[str, ...] = ()

    @property
    def all_pdb_ids(self) -> tuple[str, ...]:
        """Return ordered tuple of primary + secondary RCSB PDB IDs."""
        return (self.primary_pdb_id, *self.secondary_pdb_ids)


# ---------------------------------------------------------------------------
# 1. Core v3.1 12-Target Human Tourette & Psychiatric Registry (100% Backward-Compatible)
# ---------------------------------------------------------------------------

CNS_TARGET_REGISTRY: dict[str, CNSTargetSpec] = {
    "DRD1": CNSTargetSpec(
        target_id="DRD1",
        gene_symbol="DRD1",
        name="Dopamine D1 Receptor",
        target_class="Aminergic GPCR (Class A)",
        uniprot_id="P21728",
        chembl_id="CHEMBL2056",
        indications=("Tourette Syndrome (Phase 3 D1AMOND breakthrough)", "Movement Disorders"),
        therapeutic_moa="Selective D1/D5 antagonist (>300x selective over DRD2 to avoid EPS)",
        desired_state="Inactive (antagonist-bound)",
        primary_pdb_id="9LLG",
        secondary_pdb_ids=("9URL", "9U5E", "7LJC"),
        primary_anchor_residue="Asp103",
        hotspot_residues=(
            "Asp103",
            "Ser198",
            "Ser199",
            "Ser202",
            "Trp285",
            "Phe288",
            "Phe289",
            "Asn292",
            "Lys81",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Ecopipam",
                canonical_smiles="CN1CCc2cc(Cl)c(O)cc2[C@H]2c3ccccc3CC[C@@H]21",
                clinical_status="Phase 3 positive (D1AMOND) / FDA NDA Priority Review for TS",
                role="Selective D1/D5 antagonist",
            ),
            ClinicalReferenceDrug(
                name="SCH-23390",
                canonical_smiles="CN1CCc2cc(Cl)c(O)cc2C(c2ccccc2)C1",
                clinical_status="Preclinical gold-standard tool",
                role="Benzazepine D1 antagonist",
            ),
        ),
        notes=(
            "9LLG (2.77 A, flupentixol) is the inactive antagonist template. "
            "9URL (3.04 A, SKF-83566) contains a Gs-bound signaling complex. "
            "9U5E (2.49 A) and 7LJC (3.00 A) are active controls."
        ),
        disease_area="tourette",
        moa_class="ANTAGONIST",
        counter_targets=("DRD2", "HTR2B"),
    ),
    "VMAT2": CNSTargetSpec(
        target_id="VMAT2",
        gene_symbol="SLC18A2",
        name="Vesicular Monoamine Transporter 2",
        target_class="MFS Antiporter (SLC18)",
        uniprot_id="Q05940",
        chembl_id="CHEMBL1893",
        indications=(
            "Tardive Dyskinesia (FDA-approved)",
            "Huntington Chorea",
            "Tourette Syndrome (investigational/off-label)",
        ),
        therapeutic_moa="Reversible non-competitive presynaptic vesicular monoamine inhibitor",
        desired_state="Lumen-facing occluded/inhibited",
        primary_pdb_id="8T69",
        secondary_pdb_ids=("9KQ8", "8T6A"),
        primary_anchor_residue="Glu312",
        hotspot_residues=(
            "Glu312",
            "Asp399",
            "Asp33",
            "Asp426",
            "Arg189",
            "Phe135",
            "Phe334",
            "Tyr341",
            "Tyr433",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Valbenazine",
                canonical_smiles=(
                    "COc1cc2c(cc1OC)[C@H]1C[C@@H](OC(=O)[C@@H](N)C(C)C)[C@H](CC(C)C)CN1CC2"
                ),
                clinical_status="FDA-approved (Ingrezza) for TD & Huntington chorea",
                role="Selective VMAT2 inhibitor prodrug",
            ),
            ClinicalReferenceDrug(
                name="Tetrabenazine",
                canonical_smiles="COc1cc2c(cc1OC)[C@H]1CC(=O)[C@H](CC(C)C)CN1CC2",
                clinical_status="FDA-approved for Huntington chorea",
                role="Reversible VMAT2 inhibitor (co-crystallized in 8T69)",
            ),
        ),
        notes=(
            "Verified human single-protein ChEMBL ID is CHEMBL1893 (NOT CHEMBL1985, which is "
            "human Glucagon receptor GCGR P47871)."
        ),
        disease_area="tourette",
        moa_class="INHIBITOR",
        counter_targets=("DRD2",),
    ),
    "ADRA2A": CNSTargetSpec(
        target_id="ADRA2A",
        gene_symbol="ADRA2A",
        name="Alpha-2A Adrenergic Receptor",
        target_class="Aminergic GPCR (Class A)",
        uniprot_id="P08913",
        chembl_id="CHEMBL1867",
        indications=(
            "First-line Pediatric Tourette Syndrome",
            "Comorbid ADHD",
            "Anxiety / PTSD",
        ),
        therapeutic_moa="Postsynaptic prefrontal Gi/o orthosteric agonist (HCN gating)",
        desired_state="Active (Gi/o-coupled agonist-bound)",
        primary_pdb_id="7EJ8",
        secondary_pdb_ids=("9CBL", "7EJ0", "6KUX"),
        primary_anchor_residue="Asp113",
        hotspot_residues=(
            "Asp113",
            "Ser200",
            "Ser204",
            "Cys117",
            "Trp387",
            "Phe390",
            "Phe391",
            "Tyr394",
            "Phe412",
        ),
        hotspot_isoform_aliases=("Asp128", "Ser215", "Ser219", "Trp402", "Phe405", "Phe406"),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Guanfacine",
                canonical_smiles="NC(N)=NC(=O)Cc1c(Cl)cccc1Cl",
                clinical_status="FDA-approved ADHD / AAN first-line pediatric TS",
                role="Selective ADRA2A agonist",
            ),
            ClinicalReferenceDrug(
                name="Clonidine",
                canonical_smiles="Clc1cccc(Cl)c1NC1=NCCN1",
                clinical_status="FDA-approved ADHD / AAN first-line pediatric TS",
                role="Imidazoline ADRA2A agonist",
            ),
            ClinicalReferenceDrug(
                name="Brimonidine",
                canonical_smiles="Brc1c(NC2=NCCN2)ccc2nccnc12",
                clinical_status="FDA-approved ophthalmic alpha-2A agonist (in 7EJ8)",
                role="Active-state ADRA2A agonist template",
            ),
        ),
        notes=(
            "Cryo-EM constructs 7EJ8/9CBL/7EJ0 carry a +15 residue numbering offset "
            "(Asp128 = canonical Asp113^3.32 in X-ray 6KUX)."
        ),
        disease_area="tourette",
        moa_class="AGONIST",
        counter_targets=("HRH1",),
    ),
    "DRD2": CNSTargetSpec(
        target_id="DRD2",
        gene_symbol="DRD2",
        name="Dopamine D2 Receptor",
        target_class="Aminergic GPCR (Class A)",
        uniprot_id="P14416",
        chembl_id="CHEMBL217",
        indications=(
            "Tourette Syndrome (FDA-approved: haloperidol, pimozide, aripiprazole)",
            "Schizophrenia",
            "Anti-EPS selectivity counter-target",
        ),
        therapeutic_moa="Calibrated partial agonist or low-occupancy antagonist (anti-EPS counter)",
        desired_state="Inactive (antagonist/inverse agonist)",
        primary_pdb_id="6CM4",
        secondary_pdb_ids=("6LUQ", "7DFP"),
        primary_anchor_residue="Asp114",
        hotspot_residues=(
            "Asp114",
            "Cys118",
            "Ser193",
            "Ser197",
            "Trp386",
            "Phe389",
            "Phe390",
            "His393",
            "Ile184",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Risperidone",
                canonical_smiles="Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2",
                clinical_status="FDA-approved atypical antipsychotic / strong TS RCT evidence",
                role="DRD2/HTR2A antagonist (co-crystallized in 6CM4)",
            ),
            ClinicalReferenceDrug(
                name="Haloperidol",
                canonical_smiles="O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
                clinical_status="FDA-approved for Tourette Syndrome & Schizophrenia",
                role="Classical DRD2 antagonist (co-crystallized in 6LUQ)",
            ),
            ClinicalReferenceDrug(
                name="Aripiprazole",
                canonical_smiles="O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1",
                clinical_status="FDA-approved pediatric Tourette Syndrome & Schizophrenia",
                role="DRD2 partial agonist",
            ),
        ),
        notes="Dual role: classical anti-tic/antipsychotic target and mandatory anti-EPS counter.",
        disease_area="schizophrenia",
        moa_class="ANTAGONIST",
        counter_targets=("HRH1", "HTR2B"),
    ),
    "HRH3": CNSTargetSpec(
        target_id="HRH3",
        gene_symbol="HRH3",
        name="Histamine H3 Receptor",
        target_class="Aminergic GPCR (Class A)",
        uniprot_id="Q9Y5N1",
        chembl_id="CHEMBL264",
        indications=(
            "Genetically Validated Tourette Syndrome (HDC LoF)",
            "Narcolepsy / EDS",
            "Cognitive Deficits in ADHD & Schizophrenia",
        ),
        therapeutic_moa="Presynaptic Gi/o inverse agonist / antagonist (central histamine boost)",
        desired_state="Inactive (inverse agonist/antagonist-bound)",
        primary_pdb_id="7F61",
        secondary_pdb_ids=("8YN5", "8YN8"),
        primary_anchor_residue="Asp114",
        hotspot_residues=(
            "Asp114",
            "Glu206",
            "Tyr115",
            "Phe193",
            "Trp371",
            "Tyr374",
            "Met378",
            "Trp402",
        ),
        hotspot_isoform_aliases=("Glu395",),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="PF-03654746",
                canonical_smiles="CCNC(=O)[C@H]1C[C@@](F)(c2ccc(CN3CCCC3)c(F)c2)C1",
                clinical_status="Phase 2 clinical candidate (co-crystallized in 7F61)",
                role="Selective HRH3 antagonist",
            ),
            ClinicalReferenceDrug(
                name="Pitolisant",
                canonical_smiles="Clc1ccc(CCCOCCCN2CCCCC2)cc1",
                clinical_status="FDA-approved (Wakix) for narcolepsy",
                role="HRH3 inverse agonist",
            ),
        ),
        notes="Genetically linked to familial TS via HDC W317X loss-of-function (NEJM 2010).",
        disease_area="tourette",
        moa_class="INVERSE_AGONIST",
        counter_targets=("HRH1",),
    ),
    "PDE10A": CNSTargetSpec(
        target_id="PDE10A",
        gene_symbol="PDE10A",
        name="Phosphodiesterase 10A",
        target_class="Cyclic Nucleotide Phosphodiesterase (Zn2+/Mg2+ metalloenzyme)",
        uniprot_id="Q9Y233",
        chembl_id="CHEMBL4409",
        indications=(
            "Striatal Medium Spiny Neuron Tic Modulation",
            "Tourette Syndrome (Phase 2b Gemlapodect Allevia-2)",
            "Schizophrenia",
            "Huntington Disease",
        ),
        therapeutic_moa=(
            "Selective catalytic-site inhibitor engaging Q2 selectivity pocket (Tyr693)"
        ),
        desired_state="Catalytic inhibited (Q2 pocket engaged)",
        primary_pdb_id="5SJX",
        secondary_pdb_ids=("5SFE", "3WYM"),
        primary_anchor_residue="Gln726",
        hotspot_residues=(
            "Gln726",
            "Tyr693",
            "Phe729",
            "Phe696",
            "Met713",
            "Gly725",
            "Ile692",
        ),
        hotspot_isoform_aliases=("Gln716", "Phe719", "Met703", "Gly715", "Ile682"),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Balipodect (TAK-063)",
                canonical_smiles="COc1cn(-c2ccc(-n3cccn3)cc2F)nc(-c2ccnn2-c2ccccc2)c1=O",
                clinical_status="Phase 2 clinical candidate (co-crystallized in 3WYM)",
                role="Fast off-rate selective PDE10A inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Mardepodect (MP-10 / PF-02545920)",
                canonical_smiles="Cn1cc(-c2ccncc2)c(-c2ccc(OCc3ccc4ccccc4n3)cc2)n1",
                clinical_status="Phase 2 clinical candidate (human complex in 5SJX)",
                role="Selective PDE10A inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Gemlapodect (NOE-105)",
                canonical_smiles="Cn1ncc(C(=O)N2CCOCC2)c1C(=O)Nc1ccn2nc(-c3ccccc3)nc2c1",
                clinical_status="Phase 2a ALLEVIA-1 win / Phase 2b Allevia-2 for TS",
                role="Selective clinical PDE10A inhibitor",
            ),
        ),
        notes=(
            "Uses verified Homo sapiens Q9Y233 structures 5SJX (2.06 A MP-10), 5SFE (1.86 A), and "
            "3WYM (2.00 A TAK-063, -10 isoform numbering offset). Note: 3HR1 is Rattus norvegicus "
            "Q9QYJ6 and is excluded in favor of human 5SJX."
        ),
        disease_area="tourette",
        moa_class="INHIBITOR",
        counter_targets=("PDE1B",),
    ),
    "CHRM4": CNSTargetSpec(
        target_id="CHRM4",
        gene_symbol="CHRM4",
        name="Muscarinic Acetylcholine Receptor M4",
        target_class="Aminergic GPCR (Class A)",
        uniprot_id="P08173",
        chembl_id="CHEMBL1821",
        indications=(
            "Schizophrenia (Cobenfy FDA-approved Sep 2024)",
            "Alzheimer's Disease Psychosis (Phase 3 ADEPT)",
            "Hyperdopaminergic Basal-Ganglia Psychosis / Tic Modulation",
        ),
        therapeutic_moa="Orthosteric agonist or bitopic/allosteric M4 PAM (non-D2 striatal brake)",
        desired_state="Active (Gi1-coupled agonist-bound)",
        primary_pdb_id="8FX5",
        secondary_pdb_ids=("5DSG", "7TRK", "7TRQ"),
        primary_anchor_residue="Asp112",
        hotspot_residues=(
            "Asp112",
            "Tyr113",
            "Trp413",
            "Tyr416",
            "Asn417",
            "Tyr439",
            "Tyr443",
            "Phe186",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Xanomeline",
                canonical_smiles="CCCCCCOc1nsnc1C1=CCCN(C)C1",
                clinical_status="FDA-approved (Cobenfy / KarXT, Sep 2024)",
                role="Dual orthosteric + allosteric M4/M1 agonist (in 8FX5)",
            ),
            ClinicalReferenceDrug(
                name="Direclidine (NBI-1117568)",
                canonical_smiles="CCOC(=O)N1CC[C@]2(C1)C[C@@H](N1CCC(c3ccnn3C)CC1)C2",
                clinical_status="Phase 2 positive (Aug 2024) / entering Phase 3 for Schizophrenia",
                role="Selective CHRM4 orthosteric agonist",
            ),
            ClinicalReferenceDrug(
                name="Emraclidine",
                canonical_smiles="Cc1cc(C)c2c(n1)CN(C(=O)CC1CN(c3ccnc(C(F)(F)F)c3)C1)C2",
                clinical_status="Phase 2 (EMPOWER-1/2 missed primary endpoints Nov 2024)",
                role="Selective M4 positive allosteric modulator (PAM)",
            ),
        ),
        notes="8FX5 (2.45 A) captures dual orthosteric + extracellular allosteric xanomeline.",
        disease_area="schizophrenia",
        moa_class="AGONIST",
        counter_targets=("CHRM3", "CHRM1"),
    ),
    "HTR2A": CNSTargetSpec(
        target_id="HTR2A",
        gene_symbol="HTR2A",
        name="Serotonin 5-HT2A Receptor",
        target_class="Aminergic GPCR (Class A)",
        uniprot_id="P28223",
        chembl_id="CHEMBL224",
        indications=(
            "Atypical Antipsychotics (high 5-HT2A/D2 ratio)",
            "Tourette Augmentation",
            "Bipolar Depression & MDD Adjunct",
            "Parkinson Disease Psychosis",
        ),
        therapeutic_moa=(
            "Selective inverse agonist / antagonist (counter-screen HTR2B valvulopathy)"
        ),
        desired_state="Inactive (inverse agonist/antagonist-bound)",
        primary_pdb_id="7WC8",
        secondary_pdb_ids=("9WZO", "6A93", "6A94"),
        primary_anchor_residue="Asp155",
        hotspot_residues=(
            "Asp155",
            "Ser159",
            "Ser242",
            "Trp336",
            "Phe339",
            "Phe340",
            "Asn343",
            "Leu229",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Lumateperone",
                canonical_smiles="CN1CCN2c3c(cccc31)[C@@H]1CN(CCCC(=O)c3ccc(F)cc3)CC[C@@H]12",
                clinical_status="FDA-approved (Caplyta) for Schizophrenia & Bipolar Depression",
                role="5-HT2A antagonist / D2 modulator (in 7WC8)",
            ),
            ClinicalReferenceDrug(
                name="Pimavanserin",
                canonical_smiles="CC(C)COc1ccc(CNC(=O)N(Cc2ccc(F)cc2)C2CCN(C)CC2)cc1",
                clinical_status="FDA-approved (Nuplazid) for PD psychosis (in 9WZO)",
                role="Pure 5-HT2A inverse agonist",
            ),
        ),
        notes="Always counter-screen against HTR2B agonism (cardiac valvulopathy risk).",
        disease_area="psychiatry",
        moa_class="INVERSE_AGONIST",
        counter_targets=("HTR2B", "HRH1"),
    ),
    "SERT": CNSTargetSpec(
        target_id="SERT",
        gene_symbol="SLC6A4",
        name="Serotonin Transporter",
        target_class="Neurotransmitter Sodium Symporter (SLC6)",
        uniprot_id="P31645",
        chembl_id="CHEMBL228",
        indications=(
            "Comorbid OCD (#1 Tourette Comorbidity, 30-50%)",
            "Major Depressive Disorder",
            "Generalized Anxiety Disorder",
        ),
        therapeutic_moa="Competitive S1 / allosteric S2 serotonin reuptake inhibitor (SSRI)",
        desired_state="Outward-open inhibited (S1/S2)",
        primary_pdb_id="5I6X",
        secondary_pdb_ids=("5I71", "6AWO", "6AWP"),
        primary_anchor_residue="Asp98",
        hotspot_residues=(
            "Asp98",
            "Tyr95",
            "Ile172",
            "Ser336",
            "Phe335",
            "Phe341",
            "Ser438",
            "Thr439",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Escitalopram",
                canonical_smiles="CN(C)CCC[C@@]1(c2ccc(F)cc2)OCc2cc(C#N)ccc21",
                clinical_status="FDA-approved SSRI (S1 + S2 allosteric in 5I71)",
                role="Selective serotonin reuptake inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Paroxetine",
                canonical_smiles="Fc1ccc([C@@H]2CCNC[C@H]2COc2ccc3c(c2)OCO3)cc1",
                clinical_status="FDA-approved SSRI for OCD/MDD (in 5I6X)",
                role="High-affinity S1 SSRI",
            ),
        ),
        notes="5I71 resolves both orthosteric S1 and vestibular allosteric S2 binding sites.",
        disease_area="psychiatry",
        moa_class="INHIBITOR",
        counter_targets=("HRH1",),
    ),
    "DRD3": CNSTargetSpec(
        target_id="DRD3",
        gene_symbol="DRD3",
        name="Dopamine D3 Receptor",
        target_class="Aminergic GPCR (Class A)",
        uniprot_id="P35462",
        chembl_id="CHEMBL234",
        indications=(
            "Bipolar Depression",
            "Schizophrenia Negative Symptoms",
            "Limbic Reward / Tic Circuitry",
        ),
        therapeutic_moa="D3-preferring bitopic partial agonist or selective antagonist",
        desired_state="Inactive / Partial-agonist bound",
        primary_pdb_id="3PBL",
        secondary_pdb_ids=("9T8C", "7CMV"),
        primary_anchor_residue="Asp110",
        hotspot_residues=(
            "Asp110",
            "Val111",
            "Ser192",
            "Ser196",
            "Trp342",
            "Phe345",
            "Phe346",
            "His349",
            "Glu90",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Cariprazine",
                canonical_smiles="CN(C)C(=O)N[C@H]1CC[C@H](CCN2CCN(c3cccc(Cl)c3Cl)CC2)CC1",
                clinical_status="FDA-approved (Vraylar) for Bipolar & Schizophrenia (in 9T8C)",
                role="D3-preferring bitopic partial agonist",
            ),
            ClinicalReferenceDrug(
                name="Eticlopride",
                canonical_smiles="CCc1cc(Cl)c(OC)c(C(=O)NC[C@@H]2CCCN2CC)c1O",
                clinical_status="Reference antagonist (co-crystallized in 3PBL)",
                role="D2/D3 orthosteric antagonist",
            ),
        ),
        notes="9T8C (2.90 A) resolves clinical bitopic drug cariprazine engaging SBP Glu90/Gly94.",
        disease_area="psychiatry",
        moa_class="PARTIAL_AGONIST",
        counter_targets=("DRD2", "HTR2B"),
    ),
    "TAAR1": CNSTargetSpec(
        target_id="TAAR1",
        gene_symbol="TAAR1",
        name="Trace Amine-Associated Receptor 1",
        target_class="Aminergic GPCR (Class A)",
        uniprot_id="Q96RJ0",
        chembl_id="CHEMBL5857",
        indications=(
            "Non-D2 Antipsychotic Mechanism",
            "Schizophrenia",
            "Mood & Metabolic Regulation",
            "Substance Use Disorders",
        ),
        therapeutic_moa="Orthosteric Gs-coupled agonist dampening mesolimbic DA/5-HT firing",
        desired_state="Active (Gs-coupled agonist-bound)",
        primary_pdb_id="8W88",
        secondary_pdb_ids=("8JLO", "8WC8"),
        primary_anchor_residue="Asp103",
        hotspot_residues=(
            "Asp103",
            "Ser107",
            "Ile104",
            "Phe186",
            "Trp264",
            "Phe267",
            "Phe268",
            "Ile290",
            "Tyr294",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Ulotaront (SEP-363856)",
                canonical_smiles="CNC[C@@H]1OCCc2ccsc21",
                clinical_status="Phase 3 evaluated (co-crystallized in 8W88 & 8JLO)",
                role="TAAR1/5-HT1A orthosteric agonist",
            ),
        ),
        notes=(
            "Verified human single-protein ChEMBL ID is CHEMBL5857 (NOT CHEMBL4908, which is "
            "Mus musculus Taar1 Q923Y8)."
        ),
        disease_area="schizophrenia",
        moa_class="AGONIST",
        counter_targets=("HTR2B",),
    ),
    "DAT": CNSTargetSpec(
        target_id="DAT",
        gene_symbol="SLC6A3",
        name="Dopamine Transporter",
        target_class="Neurotransmitter Sodium Symporter (SLC6)",
        uniprot_id="Q01959",
        chembl_id="CHEMBL238",
        indications=(
            "Comorbid ADHD (50-60% of Tourette Syndrome)",
            "Bipolar Depression & Anhedonia",
            "Atypical Non-Abuse-Liable Modulators",
        ),
        therapeutic_moa="Atypical / slow-onset DAT reuptake inhibitor (occluded-state biased)",
        desired_state="Outward-facing / Intermediate inhibited",
        primary_pdb_id="8Y2G",
        secondary_pdb_ids=("9JKI", "8Y2F", "8Y2E"),
        primary_anchor_residue="Asp79",
        hotspot_residues=(
            "Asp79",
            "Phe76",
            "Val152",
            "Tyr156",
            "Phe320",
            "Phe326",
            "Ser422",
            "Ala423",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Dexmethylphenidate",
                canonical_smiles="COC(=O)[C@H](c1ccccc1)[C@H]1CCCCN1",
                clinical_status="FDA-approved for ADHD (co-crystallized in 8Y2G)",
                role="Outward-open DAT reuptake inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Bupropion",
                canonical_smiles="C[C@H](NC(C)(C)C)C(=O)c1cccc(Cl)c1",
                clinical_status="FDA-approved antidepressant/NDRI (co-crystallized in 9JKI)",
                role="Atypical non-abuse-liable DAT inhibitor",
            ),
        ),
        notes="8Y2G (2.83 A, dexmethylphenidate) & 9JKI (2.69 A, bupropion) are native human DAT.",
        disease_area="psychiatry",
        moa_class="INHIBITOR",
        counter_targets=(),
    ),
}

# ---------------------------------------------------------------------------
# 2. v4.0 Expansion: 6 New Tourette/Schizophrenia + 6 New Alzheimer's Targets
# ---------------------------------------------------------------------------

V4_EXPANSION_TARGET_REGISTRY: dict[str, CNSTargetSpec] = {
    # --- 6 New Tourette Syndrome & Schizophrenia Targets ---
    "CHRM1": CNSTargetSpec(
        target_id="CHRM1",
        gene_symbol="CHRM1",
        name="Muscarinic Acetylcholine Receptor M1",
        target_class="Aminergic GPCR (Class A)",
        uniprot_id="P11229",
        chembl_id="CHEMBL216",
        indications=(
            "Schizophrenia Pro-Cognitive & Psychosis (Cobenfy / ML-007C-MA ZEPHYR Ph2 win)",
            "Alzheimer's Disease Psychosis & Cognition (FDA Fast Track)",
        ),
        therapeutic_moa="Selective Gq/11 orthosteric agonist / PAM sparing peripheral CHRM3",
        desired_state="Active (Gq/11-coupled agonist-bound)",
        primary_pdb_id="9UCP",
        secondary_pdb_ids=("6OIJ", "6ZG4"),
        primary_anchor_residue="Asp105",
        hotspot_residues=(
            "Asp105",
            "Tyr106",
            "Ser109",
            "Ala196",
            "Phe197",
            "Trp378",
            "Tyr381",
            "Asn382",
            "Tyr404",
            "Tyr408",
        ),
        # Exact -40 TM6/TM7 numbering offset in 9UCP ICL3-truncated cryo-EM construct
        hotspot_isoform_aliases=("Trp338", "Tyr341", "Asn342", "Tyr364", "Tyr368"),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Betovumeline (ML-007C)",
                canonical_smiles="Cc1nc([C@@]23C[C@H]3CNC2)on1",
                clinical_status="Phase 2 ZEPHYR win (Jul 2026) / FDA Fast Track ADP",
                role="Compact dual CHRM1/CHRM4 agonist",
            ),
            ClinicalReferenceDrug(
                name="HTL0009936",
                canonical_smiles="CCOC(=O)N1CCC[C@H](N2CCC(C(=O)NC3(C)CCC3)CC2)CC1",
                clinical_status="Phase 1b/2a selective M1 agonist (in 6ZG4)",
                role="Selective CHRM1 agonist",
            ),
            ClinicalReferenceDrug(
                name="Xanomeline",
                canonical_smiles="CCCCCCOc1nsnc1C1=CCCN(C)C1",
                clinical_status="FDA-approved (Cobenfy, Sep 2024)",
                role="Dual CHRM1/CHRM4 agonist",
            ),
        ),
        notes=(
            "9UCP (2.88 A Active EM, Oct 2025) avoids the GPCRdb=Inactive StaR trap of "
            "6ZFZ/6ZG4; TM6/TM7 carries a -40 residue numbering offset (Trp338=Trp378, "
            "Tyr364=Tyr404, Tyr368=Tyr408)."
        ),
        disease_area="schizophrenia",
        moa_class="AGONIST",
        counter_targets=("CHRM3",),
    ),
    "HTR1A": CNSTargetSpec(
        target_id="HTR1A",
        gene_symbol="HTR1A",
        name="Serotonin 5-HT1A Receptor",
        target_class="Aminergic GPCR (Class A)",
        uniprot_id="P08908",
        chembl_id="CHEMBL214",
        indications=(
            "Schizophrenia Negative Symptoms & Anti-EPS (Brilaroxazine Ph3 RECOVER win)",
            "Tourette Anxiety / Mood Comorbidity",
            "Major Depressive Disorder & Generalized Anxiety",
        ),
        therapeutic_moa=(
            "Presynaptic/postsynaptic Gi/o partial agonist (anti-EPS & negative symptoms)"
        ),
        desired_state="Active (Gi/o-coupled partial-agonist bound)",
        primary_pdb_id="9VJE",
        secondary_pdb_ids=("9HYI", "9VJ6", "7E2Z"),
        primary_anchor_residue="Asp116",
        hotspot_residues=(
            "Asp116",
            "Ser199",
            "Tyr390",
            "Phe361",
            "Thr121",
            "Cys187",
            "Asn386",
            "Gln97",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Buspirone",
                canonical_smiles="O=C1CC2(CCCC2)CC(=O)N1CCCCN1CCN(c2ncccn2)CC1",
                clinical_status="FDA-approved anxiolytic (co-crystallized in 9VJE)",
                role="5-HT1A partial agonist",
            ),
            ClinicalReferenceDrug(
                name="Brilaroxazine (RP5063)",
                canonical_smiles="O=C1COc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1",
                clinical_status="Phase 3 RECOVER positive in Schizophrenia",
                role="Multimodal D2/D3/5-HT1A partial agonist",
            ),
        ),
        notes="9VJE (2.47 A Active GoA EM with Buspirone YLX, Nov 2025); anchor Asp116 at 2.53 A.",
        disease_area="schizophrenia",
        moa_class="PARTIAL_AGONIST",
        counter_targets=("HTR2B", "HRH1"),
    ),
    "GLYT1": CNSTargetSpec(
        target_id="GLYT1",
        gene_symbol="SLC6A9",
        name="Glycine Transporter 1",
        target_class="Neurotransmitter Sodium Symporter (SLC6)",
        uniprot_id="P48067",
        chembl_id="CHEMBL2337",
        indications=(
            "Schizophrenia Cognitive Impairment (CIAS / NMDA Hypofunction)",
            "Non-Aminergic Glutamatergic Modulation",
        ),
        therapeutic_moa="Non-basic-amine competitive/non-competitive glycine reuptake inhibitor",
        desired_state="Inward-open / Outward-open inhibited",
        primary_pdb_id="9J8D",
        secondary_pdb_ids=("8WFL", "8WFK"),
        primary_anchor_residue="Ser317",
        hotspot_residues=(
            "Ser317",
            "Cys421",
            "Tyr62",
            "Gly67",
            "Tyr142",
            "Gly319",
            "Trp322",
            "Thr418",
            "Asp474",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Iclepertin (BI 425809)",
                canonical_smiles=(
                    "C[C@@H](Oc1ccc(S(C)(=O)=O)cc1C(=O)N1C[C@@H]2C[C@]2(c2cc(C(F)(F)F)on2)C1)C(F)(F)F"
                ),
                clinical_status="Phase 3 CONNEX evaluated (landmark human cryo-EM 9J8D, Mar 2025)",
                role="Neutral non-basic-amine GlyT1 inhibitor",
            ),
        ),
        notes="3rd human SLC6 symporter in GenEuclidean; neutral scaffolds natively spare hERG.",
        disease_area="schizophrenia",
        moa_class="INHIBITOR",
        counter_targets=(),
    ),
    "MGLL": CNSTargetSpec(
        target_id="MGLL",
        gene_symbol="MGLL",
        name="Monoacylglycerol Lipase (MAGL)",
        target_class="Serine Hydrolase (alpha/beta-Hydrolase)",
        uniprot_id="Q99685",
        chembl_id="CHEMBL4191",
        indications=(
            "Tourette Syndrome Endocannabinoid Tic Modulation (reversible next-gen MAGL)",
            "Neuroinflammation & Movement Disorders",
        ),
        therapeutic_moa=(
            "Reversible non-covalent catalytic inhibitor (avoids CB1 desensitization from "
            "irreversible covalent Lu AG06466)"
        ),
        desired_state="Non-covalent catalytic inhibited",
        primary_pdb_id="9F8D",
        secondary_pdb_ids=("7ZPG", "3PE6", "5ZUN"),
        primary_anchor_residue="Ser122",
        hotspot_residues=(
            "Ser122",
            "Ala51",
            "Met123",
            "His121",
            "Arg57",
            "Tyr194",
            "Glu53",
            "Leu184",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="MGLL Reversible Tool (9F8D / A1IA1)",
                canonical_smiles="O=C1COc2ccc(C(=O)N3CCC(c4c[nH]c5ccc(Cl)cc45)CC3)cc2N1",
                clinical_status="1.14 A human non-covalent X-ray benchmark (Jan 2025)",
                role="100% reversible non-covalent MGLL inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Lu AG06466 (Elcubragistat)",
                canonical_smiles=("O=C(OC(C(F)(F)F)C(F)(F)F)N1CCN(Cc2ccc(C(F)(F)F)cc2N2CCCC2)CC1"),
                clinical_status="Phase 2a discontinued in adult TS (covalent CB1 tolerance lesson)",
                role="Historical covalent comparator (avoid HFIP-carbamate warhead)",
            ),
        ),
        notes=(
            "9F8D (1.14 A X-ray, 0 covalent _struct_conn links); Ser122 oxyanion/catalytic anchor."
        ),
        disease_area="tourette",
        moa_class="INHIBITOR",
        counter_targets=(),
    ),
    "PDE1B": CNSTargetSpec(
        target_id="PDE1B",
        gene_symbol="PDE1B",
        name="Phosphodiesterase 1B",
        target_class="Cyclic Nucleotide Phosphodiesterase (Zn2+/Mg2+ metalloenzyme)",
        uniprot_id="Q01064",
        chembl_id="CHEMBL4425",
        indications=(
            "Tourette Syndrome Striatal D1-MSN Modulation",
            "Schizophrenia Cognitive Impairment (CIAS)",
            "Parkinson & Movement Disorders",
        ),
        therapeutic_moa="Selective catalytic-site inhibitor (Zn2+/Mg2+ bimetallic cleft)",
        desired_state="Catalytic inhibited (Gln421 clamp engaged)",
        primary_pdb_id="5UOY",
        secondary_pdb_ids=("4NPW", "5B25", "5UP0"),
        primary_anchor_residue="Gln421",
        hotspot_residues=(
            "Gln421",
            "His373",
            "Phe424",
            "Leu388",
            "Met389",
            "Tyr222",
            "Pro408",
            "Leu409",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="PDE1B Co-Crystal Inhibitor (5UOY / 8HM)",
                canonical_smiles="COc1ccc(Cn2c(=O)n3ncnc3c3c4c(sc32)CN(CC2CCOCC2)CC4)cc1",
                clinical_status="1.82 A human X-ray lead-optimization benchmark",
                role="Selective non-basic-amine PDE1B inhibitor",
            ),
        ),
        notes="Retains catalytic Zn2+/Mg2+ ions cleanly via featurizer.py; complements PDE10A.",
        disease_area="tourette",
        moa_class="INHIBITOR",
        counter_targets=("PDE10A",),
    ),
    "GPR52": CNSTargetSpec(
        target_id="GPR52",
        gene_symbol="GPR52",
        name="G-Protein Coupled Receptor 52",
        target_class="Orphan Striatal GPCR (Class A)",
        uniprot_id="Q9Y2T5",
        chembl_id="CHEMBL3297639",
        indications=(
            "Non-D2 Antipsychotic Mechanism (HTL0048149 Phase 2-ready)",
            "Striatal D2-MSN Indirect-Pathway Tic Brake (Tourette Syndrome)",
        ),
        therapeutic_moa="Gs-coupled side-pocket agonist (non-basic-amine, anti-D2 striatal brake)",
        desired_state="Active (Gs-coupled agonist-bound)",
        primary_pdb_id="8HMP",
        secondary_pdb_ids=("6LI0", "9IJR"),
        primary_anchor_residue="Cys40",
        hotspot_residues=(
            "Cys40",
            "Asp38",
            "Glu191",
            "Ile189",
            "Asp188",
            "Cys94",
            "Phe117",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Compound c17 (8HMP / EN6)",
                canonical_smiles=("Cc1nn(-c2cccc3cc(Cc4cccc(C(F)(F)F)c4)sc23)c(CO)c1C(=O)NCCO"),
                clinical_status="Co-crystallized active Gs-coupled agonist (8HMP)",
                role="Non-basic-amine GPR52 agonist",
            ),
            ClinicalReferenceDrug(
                name="FTBMT",
                canonical_smiles="Cc1cc(-n2nc(Cc3cc(F)cc(C(F)(F)F)c3)nc2C)ccc1C(N)=O",
                clinical_status="Preclinical gold-standard tool agonist",
                role="Selective GPR52 agonist",
            ),
        ),
        notes="8HMP (2.77 A Active Gs EM) avoids the GPCRdb=Inactive trap of X-ray 6LI0.",
        disease_area="schizophrenia",
        moa_class="AGONIST",
        counter_targets=("DRD2",),
    ),
    # --- 6 New Alzheimer's Disease Targets ---
    "ACHE": CNSTargetSpec(
        target_id="ACHE",
        gene_symbol="ACHE",
        name="Acetylcholinesterase",
        target_class="Cholinesterase (alpha/beta-Hydrolase Gorge)",
        uniprot_id="P22303",
        chembl_id="CHEMBL220",
        indications=(
            "Alzheimer's Disease Symptomatic Cognition (FDA Standard of Care)",
            "Lewy Body Dementia & Geriatric Cholinergic Deficit",
        ),
        therapeutic_moa=(
            "Reversible dual CAS+PAS inhibitor (>100x selective over BCHE; de-risk donepezil hERG)"
        ),
        desired_state="Reversible dual CAS+PAS inhibited",
        primary_pdb_id="4EY7",
        secondary_pdb_ids=("6O4W", "4EY6", "4EY5"),
        primary_anchor_residue="Phe295",
        hotspot_residues=(
            "Phe295",
            "Trp86",
            "Trp286",
            "Tyr337",
            "Tyr341",
            "Tyr124",
            "Glu202",
            "Ser203",
            "His447",
            "Tyr72",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Donepezil",
                canonical_smiles="COc1cc2c(cc1OC)C(=O)[C@H](CC1CCN(Cc3ccccc3)CC1)C2",
                clinical_status="FDA-approved AD standard of care (in 4EY7; Known-TdP hERG risk)",
                role="Dual CAS+PAS reversible AChE inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Galantamine",
                canonical_smiles="COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2",
                clinical_status="FDA-approved AD therapy (in 4EY6; clean hERG profile)",
                role="Reversible AChE inhibitor & nAChR allosteric modulator",
            ),
            ClinicalReferenceDrug(
                name="Rivastigmine",
                canonical_smiles="CCN(C)C(=O)Oc1cccc([C@H](C)N(C)C)c1",
                clinical_status="FDA-approved AD & PD dementia therapy",
                role="Pseudo-irreversible carbamate cholinesterase inhibitor",
            ),
        ),
        notes=(
            "Dual Trp286 (PAS) + Phe295 (mid-gorge) + Trp86 (CAS) engagement drives "
            "ACHE/BCHE selectivity."
        ),
        disease_area="alzheimer",
        moa_class="INHIBITOR",
        counter_targets=("BCHE",),
    ),
    "OGA": CNSTargetSpec(
        target_id="OGA",
        gene_symbol="MGEA5",
        name="O-GlcNAcase (Beta-Hexosaminidase)",
        target_class="Glycoside Hydrolase (GH84)",
        uniprot_id="O60502",
        chembl_id="CHEMBL5921",
        indications=(
            "Alzheimer's Disease Tau O-GlcNAcylation / Anti-Tangle Stabilization",
            "Progressive Supranuclear Palsy (Primary Tauopathy)",
        ),
        therapeutic_moa="Non-sugar BBB-penetrant catalytic OGA inhibitor (precision occupancy)",
        desired_state="Catalytic inhibited (Asp174/Asp175/Asp285 cleft)",
        primary_pdb_id="9BA8",
        secondary_pdb_ids=("11LJ", "5M7S", "7OU6"),
        primary_anchor_residue="Asp285",
        hotspot_residues=(
            "Asp285",
            "Asp174",
            "Asp175",
            "Lys98",
            "Asn280",
            "Tyr219",
            "Tyr69",
            "Trp278",
        ),
        # Partner-protomer arm-exchanged homodimer ceiling residues present in 11LJ full dimer
        hotspot_isoform_aliases=("Trp645", "Trp679"),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Ceperognastat (LY3372689)",
                canonical_smiles="CC(=O)Nc1nc(F)c(CN2CC[C@H](OCc3noc(C)n3)C[C@@H]2C)s1",
                clinical_status="Phase 2 PROSPECT-ALZ (reduced tau-PET/p-tau217; in 9BA8)",
                role="Clinical non-sugar CNS OGA inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Janssen 5-Azaindole (11LJ / A1DE3)",
                canonical_smiles="Cc1ccnc(c1Nc2c3ccn(c3ccn2)CC(=O)N(C)C)C(C)C",
                clinical_status="2026 preclinical/clinical-track 5-azaindole (in 11LJ)",
                role="High-MPO non-sugar OGA inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Thiamet-G",
                canonical_smiles="CCN=C1N[C@@H]2[C@@H](O)[C@H](O)[C@@H](CO)O[C@@H]2S1",
                clinical_status="Preclinical sugar-mimetic tool (in 5M7S)",
                role="Polar glycoside-mimetic OGA inhibitor",
            ),
        ),
        notes=(
            "Verified human single-protein ChEMBL ID is CHEMBL5921. 9BA8 (2.54 A, 2025) is "
            "clean monomeric chain A; 11LJ (2.33 A, 2026) retains arm-exchanged dimer ceiling."
        ),
        disease_area="alzheimer",
        moa_class="INHIBITOR",
        counter_targets=(),
    ),
    "NLRP3": CNSTargetSpec(
        target_id="NLRP3",
        gene_symbol="NLRP3",
        name="NLRP3 Inflammasome NACHT Domain",
        target_class="NOD-Like Receptor ATPase (Allosteric NACHT Clamp)",
        uniprot_id="Q96P20",
        chembl_id="CHEMBL1741208",
        indications=(
            "Alzheimer's Disease Microglial Neuroinflammation (TARBELA-AD Ph2 / NT-0796 / VTX3232)",
            "Parkinson's Disease & Neurodegeneration",
        ),
        therapeutic_moa=(
            "Brain-penetrant non-sulfonylurea NACHT allosteric clamp locking inactive ADP state"
        ),
        desired_state="Inactive ADP-bound NACHT allosteric inhibited",
        primary_pdb_id="9GU4",
        secondary_pdb_ids=("9HG4", "8WSM", "7ALV"),
        primary_anchor_residue="Arg578",
        hotspot_residues=(
            "Arg578",
            "Glu629",
            "Ala228",
            "Tyr443",
            "Phe575",
            "Met661",
            "Ile411",
            "Thr439",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="NP3-253 (9GU4 / A1IPJ)",
                canonical_smiles="CCN1CCC[C@@H](Nc2ccc(-c3c(C)cc(C(F)(F)F)cc3O)nn2)C1",
                clinical_status="2024 brain-penetrant aminopyridazine lead (in 9GU4)",
                role="CNS-penetrant non-sulfonylurea NLRP3 inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Ruvonoflast (NT-0796)",
                canonical_smiles=("CC(C)OC(=O)[C@@H](Cc1ncccn1)OC(=O)Nc1c2c(cc3c1CCC3)CCC2"),
                clinical_status="Phase 1b/2a positive CSF IL-1b/IL-6 normalization",
                role="Microglial CES1-activated CNS NLRP3 prodrug",
            ),
            ClinicalReferenceDrug(
                name="DFV890 (9HG4 / A1IUG)",
                canonical_smiles=("CC(C)(O)c1ncc([S@](N)(=O)=NC(=O)Nc2c3c(cc4c2CCC4)CCC3)s1"),
                clinical_status="Phase 2 clinical NLRP3 inhibitor (Novartis, in 9HG4)",
                role="Sulfonimidamide NLRP3 NACHT inhibitor",
            ),
        ),
        notes=(
            "CRITICAL: Retains integral Walker A/B ADP702 cofactor (7.56 A from ligand in 9GU4) "
            "via keep_cofactors=('ADP',) to preserve inactive NACHT electrostatics."
        ),
        disease_area="alzheimer",
        moa_class="INHIBITOR",
        keep_cofactors=("ADP",),
        counter_targets=(),
    ),
    "MAOB": CNSTargetSpec(
        target_id="MAOB",
        gene_symbol="MAOB",
        name="Monoamine Oxidase B",
        target_class="Flavoenzyme Amine Oxidase (FAD-Dependent)",
        uniprot_id="P27338",
        chembl_id="CHEMBL2039",
        indications=(
            "Alzheimer's Disease Reactive Astrocyte Tonic GABA (KDS2010 Ph2a MCI/AD)",
            "Parkinson's Disease (FDA-approved Safinamide)",
        ),
        therapeutic_moa=(
            "Reversible bipartite inhibitor (>100-1000x selective over MAOA; no cheese effect)"
        ),
        desired_state="Reversible open-gate (Ile199) FAD-bound inhibited",
        primary_pdb_id="2V5Z",
        secondary_pdb_ids=("6FW0", "9FJT"),
        primary_anchor_residue="Gln206",
        hotspot_residues=(
            "Gln206",
            "Tyr435",
            "Tyr398",
            "Ile199",
            "Tyr326",
            "Cys172",
            "Trp119",
            "Leu164",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Safinamide",
                canonical_smiles="C[C@H](NCc1ccc(OCc2cccc(F)c2)cc1)C(N)=O",
                clinical_status="FDA-approved PD reversible MAO-B inhibitor (in 2V5Z)",
                role="Reversible >100x selective MAO-B inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Tisolagiline (KDS2010)",
                canonical_smiles="C[C@H](NCc1ccc(-c2ccc(C(F)(F)F)cc2)cc1)C(N)=O",
                clinical_status="Phase 2a MCI/mild Alzheimer's disease (NeuroBiogen)",
                role="Reversible >1000x selective astrocyte MAO-B inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Sembragiline (RO4602522)",
                canonical_smiles="CC(=O)N[C@H]1CC(=O)N(c2ccc(OCc3cccc(F)c3)cc2)C1",
                clinical_status="Phase 2b MAYA evaluated in AD (neutral low-hERG scaffold)",
                role="Reversible neutral-lactam MAO-B inhibitor",
            ),
        ),
        notes=(
            "CRITICAL: Retains integral FAD600 cofactor (3.00 A contact in 2V5Z) via "
            "keep_cofactors=('FAD',). 2V5Z (1.60 A) is detergent-free (no C15 artifact)."
        ),
        disease_area="alzheimer",
        moa_class="INHIBITOR",
        keep_cofactors=("FAD",),
        counter_targets=("MAOA",),
    ),
    "GSK3B": CNSTargetSpec(
        target_id="GSK3B",
        gene_symbol="GSK3B",
        name="Glycogen Synthase Kinase-3 Beta",
        target_class="Serine/Threonine Protein Kinase (CMGC)",
        uniprot_id="P49841",
        chembl_id="CHEMBL262",
        indications=(
            "Alzheimer's Disease Tau Hyperphosphorylation",
            "Bipolar Disorder & Neurodegeneration",
        ),
        therapeutic_moa=(
            "ATP-hinge / back-pocket paralog-selective inhibitor (>100x selective over CDK2)"
        ),
        desired_state="ATP-hinge inhibited (DFG-in, Val135/Asp133 hinge)",
        primary_pdb_id="7SXJ",
        secondary_pdb_ids=("1Q5K", "4AFJ", "7B6F"),
        primary_anchor_residue="Val135",
        hotspot_residues=(
            "Val135",
            "Asp133",
            "Lys85",
            "Asp200",
            "Cys199",
            "Leu132",
            "Tyr134",
            "Gln185",
            "Ile62",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="BRD0705",
                canonical_smiles="CC[C@@]1(c2ccccc2)C2=C(CC(C)(C)CC2=O)Nc2n[nH]cc21",
                clinical_status="Paralog-selective GSK3 benchmark (co-crystallized in 7SXJ)",
                role="Selective ATP-competitive GSK3 inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Tideglusib",
                canonical_smiles="O=c1sn(-c2cccc3ccccc23)c(=O)n1Cc1ccccc1",
                clinical_status="Phase 2 ARGO evaluated in AD & PSP",
                role="Non-ATP-competitive GSK3B inhibitor",
            ),
        ),
        notes=(
            "7SXJ (1.85 A, 351 aa kinase domain) avoids the AKT2+peptide false-positive PDB trap "
            "(2JDO/3QKL). Counter-screen against CDK2 (6Q4G) via Cys199/Leu132 selectivity handle."
        ),
        disease_area="alzheimer",
        moa_class="INHIBITOR",
        counter_targets=("CDK2",),
    ),
    "BACE1": CNSTargetSpec(
        target_id="BACE1",
        gene_symbol="BACE1",
        name="Beta-Secretase 1 (Memapsin-2)",
        target_class="Aspartyl Protease (Pepsin A1)",
        uniprot_id="P56817",
        chembl_id="CHEMBL4822",
        indications=(
            "Alzheimer's Disease Amyloidogenic APP Processing (Low-Occupancy Primary Prevention)",
            "CNS Low-Efflux Polar-Warhead Benchmark",
        ),
        therapeutic_moa=(
            "Reversible catalytic aspartyl-dyad inhibitor (low-occupancy, >10-100x vs BACE2/CTSD)"
        ),
        desired_state="Flap-closed catalytic dyad (Asp32/Asp228) inhibited",
        primary_pdb_id="6EQM",
        secondary_pdb_ids=("5HU1", "7MYI", "6UWP"),
        primary_anchor_residue="Asp32",
        hotspot_residues=(
            "Asp32",
            "Asp228",
            "Tyr71",
            "Gly230",
            "Gly34",
            "Phe108",
            "Ser35",
            "Trp115",
        ),
        # Exact +61 precursor offset in 5HU1 (Asp93=Asp32, Asp289=Asp228, Tyr132=Tyr71)
        hotspot_isoform_aliases=("Asp93", "Asp289", "Tyr132"),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Umibecestat (CNP520)",
                canonical_smiles=(
                    "C[C@@]1(c2nc(NC(=O)c3ncc(C(F)(F)F)cc3Cl)ccc2F)CO[C@@](C)(C(F)(F)F)C(N)=N1"
                ),
                clinical_status="Phase 2/3 Generation Study evaluated (co-crystallized in 6EQM)",
                role="Fluorinated amino-oxazine BACE1 inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Verubecestat (MK-8931)",
                canonical_smiles="CN1C(N)=N[C@](C)(c2cc(NC(=O)c3ccc(F)cn3)ccc2F)CS1(=O)=O",
                clinical_status="Phase 3 EPOCH/APECS evaluated (matched pair in 5HU1 & BACE2 7D5B)",
                role="Iminothiadiazine dioxide BACE1 inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Elenbecestat (E2609)",
                canonical_smiles=(
                    "C[C@H]1OC[C@]2(c3cc(NC(=O)c4cnc(C(F)F)cn4)ccc3F)N=C(N)SC[C@H]12"
                ),
                clinical_status="Phase 3 MissionAD evaluated",
                role="Aminothiazine BACE1 inhibitor",
            ),
        ),
        notes=(
            "6EQM (1.35 A, mature numbering Asp32/Asp228) is primary; 5HU1 (1.50 A, verubecestat) "
            "uses +61 precursor numbering (Asp93/Asp289/Tyr132) mapped via hotspot_isoform_aliases."
        ),
        disease_area="alzheimer",
        moa_class="INHIBITOR",
        counter_targets=("BACE2",),
    ),
}

# ---------------------------------------------------------------------------
# 3. Matched Structural Selectivity & Avoid-ome Counter-Targets (7 Human PDBs)
# ---------------------------------------------------------------------------

V4_COUNTER_TARGET_REGISTRY: dict[str, CNSTargetSpec] = {
    "BCHE": CNSTargetSpec(
        target_id="BCHE",
        gene_symbol="BCHE",
        name="Butyrylcholinesterase (ACHE Counter-Target)",
        target_class="Cholinesterase (Selectivity Counter-Target)",
        uniprot_id="P06276",
        chembl_id="CHEMBL1914",
        indications=("Selectivity Counter-Target for ACHE (Peripheral Cholinergic Margin)",),
        therapeutic_moa="Counter-screen to spare peripheral BCHE (avoid GI/bradycardia)",
        desired_state="Catalytic gorge inhibited (counter-screen)",
        primary_pdb_id="4BDS",
        secondary_pdb_ids=("6ZWI", "6EQP"),
        primary_anchor_residue="Trp82",
        hotspot_residues=("Trp82", "Ser198", "His438", "Glu197", "Ala328", "Leu286", "Tyr332"),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Tacrine",
                canonical_smiles="Nc1c2c(nc3ccccc13)CCCC2",
                clinical_status="Historical AD drug (withdrawn for hepatotoxicity; in 4BDS)",
                role="Dual AChE/BChE reference ligand",
            ),
        ),
        notes="Larger ~500 A^3 aliphatic gorge vs ACHE ~300 A^3 aromatic gorge.",
        disease_area="counter",
        moa_class="INHIBITOR",
    ),
    "MAOA": CNSTargetSpec(
        target_id="MAOA",
        gene_symbol="MAOA",
        name="Monoamine Oxidase A (MAOB Anti-Tyramine Counter-Target)",
        target_class="Flavoenzyme Amine Oxidase (Selectivity Counter-Target)",
        uniprot_id="P21397",
        chembl_id="CHEMBL1951",
        indications=("Selectivity Counter-Target for MAOB (Avoid Tyramine Hypertensive Crisis)",),
        therapeutic_moa="Counter-screen to maintain >100-1000x MAOB/MAOA selectivity",
        desired_state="Monopartite FAD-bound cavity (counter-screen)",
        primary_pdb_id="2Z5X",
        secondary_pdb_ids=("2Z5Y",),
        primary_anchor_residue="Phe208",
        hotspot_residues=("Phe208", "Ile335", "Gln215", "Tyr407", "Tyr444", "Asn181", "Phe352"),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Harmine",
                canonical_smiles="COc1ccc2c(c1)[nH]c1c(C)nccc12",
                clinical_status="Reversible selective MAO-A tool (in 2Z5X)",
                role="MAOA reference ligand",
            ),
        ),
        notes=(
            "Retains integral FAD600 cofactor via keep_cofactors=('FAD',); "
            "Phe208/Ile335 steric gate."
        ),
        disease_area="counter",
        moa_class="INHIBITOR",
        keep_cofactors=("FAD",),
    ),
    "CDK2": CNSTargetSpec(
        target_id="CDK2",
        gene_symbol="CDK2",
        name="Cyclin-Dependent Kinase 2 (GSK3B Anti-Cell-Cycle Counter-Target)",
        target_class="Serine/Threonine Protein Kinase (Selectivity Counter-Target)",
        uniprot_id="P24941",
        chembl_id="CHEMBL301",
        indications=("Selectivity Counter-Target for GSK3B (Avoid Cell-Cycle/Wnt Toxicity)",),
        therapeutic_moa="Counter-screen to maintain >50-100x GSK3B/CDK2 selectivity",
        desired_state="ATP-hinge inhibited (counter-screen)",
        primary_pdb_id="6Q4G",
        secondary_pdb_ids=("1HCK",),
        primary_anchor_residue="Leu83",
        hotspot_residues=("Leu83", "Glu81", "Phe80", "Lys33", "Asp145", "Ala144", "Ile10"),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="CDK2 Purine Tool (6Q4G / HJK)",
                canonical_smiles="Nc1nc(-c2cccc(CC(=O)O)c2)c2nc[nH]c2n1",
                clinical_status="0.98 A atomic-resolution human CDK2 counter-screen (in 6Q4G)",
                role="CDK2 ATP-hinge reference",
            ),
        ),
        notes="Bulky Phe80 gatekeeper & Ala144 contrast with GSK3B Leu132 & Cys199.",
        disease_area="counter",
        moa_class="INHIBITOR",
    ),
    "BACE2": CNSTargetSpec(
        target_id="BACE2",
        gene_symbol="BACE2",
        name="Beta-Secretase 2 (BACE1 Matched-Ligand Counter-Target)",
        target_class="Aspartyl Protease (Selectivity Counter-Target)",
        uniprot_id="Q9Y5Z0",
        chembl_id="CHEMBL2525",
        indications=("Selectivity Counter-Target for BACE1 (Avoid PMEL/Pigmentation Toxicity)",),
        therapeutic_moa="Matched-ligand counter-screen using Verubecestat (66F) in 7D5B vs 5HU1",
        desired_state="Catalytic dyad inhibited (counter-screen)",
        primary_pdb_id="7D5B",
        secondary_pdb_ids=("7N4N", "6JSZ"),
        primary_anchor_residue="Asp48",
        hotspot_residues=("Asp48", "Asp241", "Tyr87", "Gly243", "Phe121", "Trp128", "Ser51"),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Verubecestat (66F)",
                canonical_smiles="CN1C(N)=N[C@](C)(c2cc(NC(=O)c3ccc(F)cn3)ccc2F)CS1(=O)=O",
                clinical_status="Exact matched co-crystal ligand in BACE1 5HU1 & BACE2 7D5B",
                role="Matched-pair selectivity benchmark ligand",
            ),
        ),
        notes="1.31 A human BACE2 with exact Verubecestat (66F) for matched-pair delta-delta-G.",
        disease_area="counter",
        moa_class="INHIBITOR",
    ),
    "HTR2B": CNSTargetSpec(
        target_id="HTR2B",
        gene_symbol="HTR2B",
        name="Serotonin 5-HT2B Receptor (Cardiac Valvulopathy Avoid-ome)",
        target_class="Aminergic GPCR (Avoid-ome Kill-Switch)",
        uniprot_id="P41595",
        chembl_id="CHEMBL1833",
        indications=("Avoid-ome Kill-Switch: Fatal Cardiac Valvulopathy (Mandatory Off-Target)",),
        therapeutic_moa="Mandatory Avoid-ome counter-screen (EC50 > 10 uM / zero agonism)",
        desired_state="Active agonist-bound (valvulopathy counter-screen)",
        primary_pdb_id="7SRQ",
        secondary_pdb_ids=("4IB4",),
        primary_anchor_residue="Asp135",
        hotspot_residues=("Asp135", "Ser139", "Val136", "Leu209", "Phe340", "Phe341", "Asn344"),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Ergotamine",
                canonical_smiles=(
                    "CN1C[C@H](C(=O)N[C@]2(C)O[C@@]3(O)[C@@H]4CCCN4C(=O)[C@H](Cc4ccccc4)N3C2=O)"
                    "C=C2c3cccc4[nH]cc(c34)C[C@H]21"
                ),
                clinical_status="Known 5-HT2B valvulopathy positive control (in 4IB4)",
                role="Avoid-ome positive control",
            ),
        ),
        notes="Every dopaminergic/serotonergic CNS amine must clear 5-HT2B valvulopathy risk.",
        disease_area="counter",
        moa_class="AGONIST",
    ),
    "HRH1": CNSTargetSpec(
        target_id="HRH1",
        gene_symbol="HRH1",
        name="Histamine H1 Receptor (Sedation & Weight-Gain Avoid-ome)",
        target_class="Aminergic GPCR (Avoid-ome Kill-Switch)",
        uniprot_id="P35367",
        chembl_id="CHEMBL231",
        indications=("Avoid-ome Kill-Switch: Daytime Sedation & Antipsychotic Weight Gain",),
        therapeutic_moa="Avoid-ome counter-screen minimizing H1 antagonism",
        desired_state="Inactive antagonist-bound (counter-screen)",
        primary_pdb_id="8X5Y",
        secondary_pdb_ids=("3RZE",),
        primary_anchor_residue="Asp107",
        hotspot_residues=("Asp107", "Tyr108", "Thr112", "Trp428", "Tyr431", "Phe432", "Lys191"),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Doxepin",
                canonical_smiles="CN(C)CC/C=C1/c2ccccc2COc2ccccc21",
                clinical_status="FDA-approved H1 antagonist positive control (in 3RZE)",
                role="Avoid-ome positive control",
            ),
        ),
        notes="Primary driver of olanzapine/clozapine metabolic weight gain & pediatric sedation.",
        disease_area="counter",
        moa_class="ANTAGONIST",
    ),
    "CHRM3": CNSTargetSpec(
        target_id="CHRM3",
        gene_symbol="CHRM3",
        name="Muscarinic Acetylcholine Receptor M3 (Peripheral GI/Cardiac Avoid-ome)",
        target_class="Aminergic GPCR (Avoid-ome Kill-Switch)",
        uniprot_id="P20309",
        chembl_id="CHEMBL245",
        indications=("Avoid-ome Counter-Screen for CHRM4/CHRM1 (Peripheral GI/Salivation)",),
        therapeutic_moa="Counter-screen sparing peripheral M3 agonism/antagonism",
        desired_state="Active agonist-bound (counter-screen)",
        primary_pdb_id="8EA0",
        secondary_pdb_ids=(),
        primary_anchor_residue="Asp148",
        hotspot_residues=("Asp148", "Tyr149", "Ser152", "Trp504", "Tyr507", "Asn508", "Tyr530"),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Xanomeline",
                canonical_smiles="CCCCCCOc1nsnc1C1=CCCN(C)C1",
                clinical_status="Clinical comparator requiring peripheral trospium co-dosing",
                role="Non-selective muscarinic agonist reference",
            ),
        ),
        notes="Selectivity over CHRM3 is the defining goal of trospium-free next-gen M4/M1 drugs.",
        disease_area="counter",
        moa_class="AGONIST",
    ),
}

# Unified 24-Target Tri-Disorder Primary Atlas (12 v3.1 + 6 TS/SCZ + 6 AD)
TRI_DISORDER_TARGET_REGISTRY: dict[str, CNSTargetSpec] = {
    **CNS_TARGET_REGISTRY,
    **V4_EXPANSION_TARGET_REGISTRY,
}

# Complete 31-Target Human Structural Registry (24 Primary + 7 Matched Counter-Targets)
ALL_CNS_TARGET_REGISTRY: dict[str, CNSTargetSpec] = {
    **TRI_DISORDER_TARGET_REGISTRY,
    **V4_COUNTER_TARGET_REGISTRY,
}

# Alias map supporting target IDs, gene symbols (SLC18A2, SLC6A4, SLC6A3, SLC6A9, MGEA5),
# ChEMBL IDs, and UniProt IDs across ALL 31 registered human targets.
_TARGET_ALIAS_MAP: dict[str, str] = {}
for _tid, _spec in ALL_CNS_TARGET_REGISTRY.items():
    _TARGET_ALIAS_MAP[_tid.upper()] = _tid
    _TARGET_ALIAS_MAP[_spec.gene_symbol.upper()] = _tid
    _TARGET_ALIAS_MAP[_spec.chembl_id.upper()] = _tid
    _TARGET_ALIAS_MAP[_spec.uniprot_id.upper()] = _tid
# Common pharmacology aliases
_TARGET_ALIAS_MAP.update(
    {
        "MAGL": "MGLL",
        "5-HT2B": "HTR2B",
        "5-HT1A": "HTR1A",
        "5-HT2A": "HTR2A",
        "H1": "HRH1",
        "H3": "HRH3",
        "M1": "CHRM1",
        "M3": "CHRM3",
        "M4": "CHRM4",
    }
)


_CUSTOM_TARGET_LOCK = RLock()
_CUSTOM_TARGET_REGISTRY: dict[str, CNSTargetSpec] = {}
_CUSTOM_TARGET_ALIASES: dict[str, str] = {}


def register_cns_target(
    spec: CNSTargetSpec,
    *,
    aliases: Sequence[str] = (),
    override: bool = False,
) -> CNSTargetSpec:
    """Register a custom protein target specification at runtime.

    Custom targets are stored separately from the immutable curated benchmark atlases
    (``CNS_TARGET_REGISTRY``, ``TRI_DISORDER_TARGET_REGISTRY``, ``ALL_CNS_TARGET_REGISTRY``)
    so scientific benchmark counts remain invariant while ``get_cns_target``,
    ``list_cns_targets``, and ``Pocket.from_target`` seamlessly resolve custom targets.
    """
    tid = spec.target_id.strip().upper()
    if not tid:
        raise RegistryError("CNSTargetSpec.target_id cannot be empty.")

    with _CUSTOM_TARGET_LOCK:
        if not override and (tid in _CUSTOM_TARGET_REGISTRY or tid in ALL_CNS_TARGET_REGISTRY):
            raise RegistryError(
                f"Target {tid!r} is already registered.",
                hint="Choose a unique target_id or pass override=True.",
            )
        if tid in _CUSTOM_TARGET_REGISTRY:
            for ak in [k for k, v in _CUSTOM_TARGET_ALIASES.items() if v == tid]:
                del _CUSTOM_TARGET_ALIASES[ak]

        _CUSTOM_TARGET_REGISTRY[tid] = spec
        index_keys = {
            tid,
            spec.gene_symbol.strip().upper(),
            spec.chembl_id.strip().upper(),
            spec.uniprot_id.strip().upper(),
            *(a.strip().upper() for a in aliases if a.strip()),
        }
        for key in index_keys:
            if key:
                _CUSTOM_TARGET_ALIASES[key] = tid
    return spec


def unregister_cns_target(target_id: str) -> CNSTargetSpec:
    """Remove a previously registered custom protein target from the runtime registry."""
    key = target_id.strip().upper()
    with _CUSTOM_TARGET_LOCK:
        resolved = _CUSTOM_TARGET_ALIASES.get(key, key)
        if resolved not in _CUSTOM_TARGET_REGISTRY:
            raise TargetNotFoundError(
                f"Custom target {target_id!r} is not registered.",
                hint="Only runtime-registered custom targets can be unregistered.",
            )
        removed = _CUSTOM_TARGET_REGISTRY.pop(resolved)
        for ak in [k for k, v in _CUSTOM_TARGET_ALIASES.items() if v == resolved]:
            del _CUSTOM_TARGET_ALIASES[ak]
        return removed


def get_cns_target(identifier: str) -> CNSTargetSpec:
    """Look up a ``CNSTargetSpec`` by target ID, gene symbol, ChEMBL ID, or UniProt ID."""
    key = identifier.strip().upper()
    with _CUSTOM_TARGET_LOCK:
        custom_tid = _CUSTOM_TARGET_ALIASES.get(key)
        if custom_tid is not None and custom_tid in _CUSTOM_TARGET_REGISTRY:
            return _CUSTOM_TARGET_REGISTRY[custom_tid]

    resolved = _TARGET_ALIAS_MAP.get(key)
    if resolved is not None:
        return ALL_CNS_TARGET_REGISTRY[resolved]
    # Clean optional fallback into the 4 verified Tourette CSTC / GABAergic expansion targets
    # without altering CNS_TARGET_REGISTRY (12), V4_EXPANSION_TARGET_REGISTRY (12), or
    # ALL_CNS_TARGET_REGISTRY (31) counts.
    from geneuclidean.cns.tourette_circuit import (
        TOURETTE_CIRCUIT_EXPANSION_REGISTRY,
        get_tourette_circuit_target,
    )

    try:
        return get_tourette_circuit_target(identifier)
    except KeyError:
        with _CUSTOM_TARGET_LOCK:
            custom_keys = set(_CUSTOM_TARGET_REGISTRY.keys())
        valid = ", ".join(
            sorted(
                set(ALL_CNS_TARGET_REGISTRY.keys())
                | set(TOURETTE_CIRCUIT_EXPANSION_REGISTRY)
                | custom_keys
            )
        )
        raise TargetNotFoundError(
            f"Unknown CNS target {identifier!r}. Registered targets: {valid}",
            hint=(
                "Inspect available targets via geneuclidean.list_targets() or "
                "register a custom target via geneuclidean.register_target(...)."
            ),
        ) from None


def list_cns_targets(
    indication_filter: str | None = None,
    *,
    disease_area: str | None = None,
    include_v4: bool = False,
    include_counters: bool = False,
    include_ts_circuit: bool = False,
) -> list[CNSTargetSpec]:
    """Return curated human CNS target specs with 100% v3.1 & v4.0 backward compatibility.

    - Called with no arguments (``list_cns_targets()``), returns the 12 core v3.1 targets.
    - Pass ``include_v4=True`` or ``disease_area=...`` (``"tourette"``, ``"schizophrenia"``,
      ``"psychiatry"``, ``"alzheimer"``, ``"counter"``, or ``"all"``) to query the expanded
      24-target Tri-Disorder Human CNS Atlas (plus optional matched counter-targets).
    - Pass ``include_ts_circuit=True`` to also append the 4 verified Tourette CSTC Basal
      Ganglia & GABAergic expansion targets (``GABRA5``, ``GAT1``, ``ADORA2A``, ``GRM5``).
    """
    use_v4 = include_v4 or (disease_area is not None)
    if include_counters or (disease_area is not None and disease_area.strip().lower() == "counter"):
        pool = list(ALL_CNS_TARGET_REGISTRY.values())
    elif use_v4:
        pool = list(TRI_DISORDER_TARGET_REGISTRY.values())
    else:
        pool = list(CNS_TARGET_REGISTRY.values())

    if include_ts_circuit:
        from geneuclidean.cns.tourette_circuit import TOURETTE_CIRCUIT_EXPANSION_REGISTRY

        pool.extend(TOURETTE_CIRCUIT_EXPANSION_REGISTRY.values())

    with _CUSTOM_TARGET_LOCK:
        if _CUSTOM_TARGET_REGISTRY:
            seen_ids = {s.target_id.upper() for s in pool}
            for custom_spec in _CUSTOM_TARGET_REGISTRY.values():
                if custom_spec.target_id.upper() not in seen_ids:
                    pool.append(custom_spec)

    if disease_area is not None and disease_area.strip():
        area = disease_area.strip().lower()
        if area != "all":
            pool = [spec for spec in pool if spec.disease_area.lower() == area]

    if indication_filter is None or not indication_filter.strip():
        return pool

    needle = indication_filter.strip().lower()
    return [
        spec
        for spec in pool
        if any(needle in ind.lower() for ind in spec.indications)
        or needle in spec.therapeutic_moa.lower()
        or needle in spec.disease_area.lower()
    ]


def list_tri_disorder_targets(
    *,
    disease_area: str | None = None,
    include_counters: bool = False,
) -> list[CNSTargetSpec]:
    """Return all 24 verified *Homo sapiens* Tri-Disorder targets (or filter by area)."""
    return list_cns_targets(
        disease_area=disease_area or "all",
        include_v4=True,
        include_counters=include_counters,
    )


def find_local_pocket_files(
    target_id: str,
    *,
    pdb_id: str | None = None,
    structures_dir: str | Path = DEFAULT_CNS_STRUCTURES_DIR,
) -> tuple[Path, Path | None] | None:
    """Locate pre-cleaned ``*_pocket.pdb`` and ``*_lig.sdf`` under ``structures_dir``."""
    spec = get_cns_target(target_id)
    root = Path(structures_dir)
    candidate_roots = [root]
    # When using the default v3.1 directory, also search v4_expansion & ts_circuit_expansion
    # automatically without breaking offline CI isolation when callers pass a custom tmp_path.
    if root == DEFAULT_CNS_STRUCTURES_DIR:
        if DEFAULT_V4_STRUCTURES_DIR.exists():
            candidate_roots.append(DEFAULT_V4_STRUCTURES_DIR)
        if DEFAULT_TS_CIRCUIT_STRUCTURES_DIR.exists():
            candidate_roots.append(DEFAULT_TS_CIRCUIT_STRUCTURES_DIR)

    candidate_pdbs = (pdb_id.strip().upper(),) if pdb_id else spec.all_pdb_ids

    for base in candidate_roots:
        clean_dir = base / "clean_pockets"
        if clean_dir.exists():
            for pid in candidate_pdbs:
                matches = sorted(clean_dir.glob(f"{pid}_{spec.target_id}_*_pocket.pdb"))
                if not matches:
                    matches = sorted(clean_dir.glob(f"{pid}_*_pocket.pdb"))
                if matches:
                    pocket_path = matches[0]
                    sdf_candidate = Path(str(pocket_path).replace("_pocket.pdb", "_lig.sdf"))
                    return pocket_path, (sdf_candidate if sdf_candidate.exists() else None)

        raw_dir = base / "raw_cif"
        if raw_dir.exists():
            for pid in candidate_pdbs:
                for suffix in (".cif", ".pdb"):
                    raw_path = raw_dir / f"{pid}{suffix}"
                    if raw_path.exists():
                        return raw_path, None
    return None


def fetch_rcsb_structure(
    pdb_id: str,
    dest_dir: str | Path,
    *,
    fmt: Literal["cif", "pdb"] = "cif",
    timeout: float = 30.0,
) -> Path:
    """Download an atomic 3D structure directly from RCSB PDB into ``dest_dir``."""
    pid = pdb_id.strip().upper()
    out_dir = Path(dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}.{fmt}"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url = f"{RCSB_DOWNLOAD_URL}/{pid}.{fmt}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


def load_cns_target_pocket(
    target_id: str,
    *,
    pdb_id: str | None = None,
    structures_dir: str | Path = DEFAULT_CNS_STRUCTURES_DIR,
    allow_fetch: bool = False,
    pocket_cutoff: float = 10.0,
    contact_cutoff: float = 4.5,
    max_atoms: int = 384,
) -> PocketFeatures:
    """Load and featurize a human CNS target's 3D binding pocket into ``PocketFeatures``.

    Automatically propagates target-specific ``spec.keep_cofactors`` (e.g., ``("FAD",)`` for
    ``MAOB``/``MAOA`` and ``("ADP",)`` for ``NLRP3``) to ``featurizer.py``.
    """
    spec = get_cns_target(target_id)
    chosen_pdb = (pdb_id or spec.primary_pdb_id).strip().upper()
    local = find_local_pocket_files(spec.target_id, pdb_id=pdb_id, structures_dir=structures_dir)
    cofactors = spec.keep_cofactors

    if local is not None:
        pocket_path, lig_sdf = local
        if lig_sdf is not None:
            fallback = spec.reference_drugs[0].canonical_smiles if spec.reference_drugs else None
            return extract_pocket_from_files(
                pocket_path,
                ligand_sdf=lig_sdf,
                fallback_smiles=fallback,
                pdb_id=pocket_path.stem.split("_")[0].lower(),
                pocket_cutoff=pocket_cutoff,
                contact_cutoff=contact_cutoff,
                max_atoms=max_atoms,
                keep_cofactors=cofactors,
            )
        return featurize_pdb_pocket_inference(
            pocket_path,
            pocket_cutoff=pocket_cutoff,
            max_atoms=max_atoms,
            pdb_id=chosen_pdb.lower(),
            keep_cofactors=cofactors,
        )

    if not allow_fetch:
        raise FileNotFoundError(
            f"No cached 3D pocket found for {spec.target_id} ({chosen_pdb}) under "
            f"{Path(structures_dir)}. Pass allow_fetch=True to download from RCSB PDB."
        )

    fetched = fetch_rcsb_structure(chosen_pdb, Path(structures_dir) / "fetched")
    return featurize_pdb_pocket_inference(
        fetched,
        pocket_cutoff=pocket_cutoff,
        max_atoms=max_atoms,
        pdb_id=chosen_pdb.lower(),
        keep_cofactors=cofactors,
    )


def _normalize_residue_label(resname: str, resseq: int) -> str:
    """Format residue label as ``Asp103`` style (title-case 3-letter + integer seq)."""
    return f"{resname.strip().capitalize()}{int(resseq)}"


def audit_pocket_hotspots(
    pocket_pdb: str | Path,
    target_spec: CNSTargetSpec,
    *,
    ligand_sdf: str | Path | None = None,
    contact_cutoff: float = 4.5,
) -> dict[str, Any]:
    """Audit whether canonical pharmacological hotspot residues and cofactors appear in a pocket."""
    path = Path(pocket_pdb)
    parser: Any = (
        MMCIFParser(QUIET=True)
        if path.suffix.lower() in (".cif", ".mmcif")
        else PDBParser(QUIET=True)
    )
    structure = parser.get_structure(path.stem, str(path))
    model = structure[0]

    lig_coords: np.ndarray | None = None
    if ligand_sdf is not None and Path(ligand_sdf).exists():
        suppl = Chem.SDMolSupplier(str(ligand_sdf), sanitize=False, removeHs=False)
        if len(suppl) > 0 and suppl[0] is not None and suppl[0].GetNumConformers() > 0:
            conf = suppl[0].GetConformer()
            pts = [
                [
                    float(conf.GetAtomPosition(a.GetIdx()).x),
                    float(conf.GetAtomPosition(a.GetIdx()).y),
                    float(conf.GetAtomPosition(a.GetIdx()).z),
                ]
                for a in suppl[0].GetAtoms()
                if a.GetAtomicNum() > 1
            ]
            if pts:
                lig_coords = np.asarray(pts, dtype=np.float32)

    all_expected = {r.lower(): r for r in target_spec.hotspot_residues}
    alias_expected = {r.lower(): r for r in target_spec.hotspot_isoform_aliases}
    anchor_prefix = target_spec.primary_anchor_residue[:3].lower()
    anchor_keys = {target_spec.primary_anchor_residue.lower()}
    if target_spec.hotspot_isoform_aliases:
        first_alias = target_spec.hotspot_isoform_aliases[0]
        if first_alias[:3].lower() == anchor_prefix:
            anchor_keys.add(first_alias.lower())

    expected_cofactors = {c.strip().upper() for c in target_spec.keep_cofactors}
    present_cofactors: set[str] = set()
    present_hotspots: set[str] = set()
    contact_hotspots: set[str] = set()
    anchor_in_contact = False

    for chain in model:
        for residue in chain:
            resname_up = str(residue.get_resname()).strip().upper()
            if resname_up in expected_cofactors:
                present_cofactors.add(resname_up)
            if str(residue.id[0]) != " ":
                continue
            label = _normalize_residue_label(str(residue.get_resname()), int(residue.id[1]))
            lkey = label.lower()
            if lkey not in all_expected and lkey not in alias_expected:
                continue
            canonical_label = all_expected[lkey] if lkey in all_expected else alias_expected[lkey]
            present_hotspots.add(canonical_label)

            if lig_coords is not None:
                res_pts = np.asarray(
                    [atom.get_coord() for atom in residue.get_atoms() if atom.element != "H"],
                    dtype=np.float32,
                )
                if len(res_pts) > 0:
                    diff = res_pts[:, None, :] - lig_coords[None, :, :]
                    min_d = float(np.sqrt(np.sum(diff * diff, axis=-1)).min())
                    if min_d <= float(contact_cutoff):
                        contact_hotspots.add(canonical_label)
                        if lkey in anchor_keys:
                            anchor_in_contact = True

    return {
        "target_id": target_spec.target_id,
        "pocket_path": str(path),
        "primary_anchor": target_spec.primary_anchor_residue,
        "expected_hotspots": list(target_spec.hotspot_residues),
        "present_hotspots": sorted(present_hotspots),
        "contact_hotspots_4p5A": sorted(contact_hotspots),
        "anchor_in_contact": anchor_in_contact,
        "hotspot_coverage": (
            min(1.0, len(present_hotspots) / len(target_spec.hotspot_residues))
            if target_spec.hotspot_residues
            else 0.0
        ),
        "expected_cofactors": sorted(expected_cofactors),
        "present_cofactors": sorted(present_cofactors),
    }


__all__ = [
    "ALL_CNS_TARGET_REGISTRY",
    "CNS_TARGET_REGISTRY",
    "DEFAULT_CNS_STRUCTURES_DIR",
    "DEFAULT_TS_CIRCUIT_STRUCTURES_DIR",
    "DEFAULT_V4_STRUCTURES_DIR",
    "RCSB_DOWNLOAD_URL",
    "TRI_DISORDER_TARGET_REGISTRY",
    "V4_COUNTER_TARGET_REGISTRY",
    "V4_EXPANSION_TARGET_REGISTRY",
    "CNSTargetSpec",
    "ClinicalReferenceDrug",
    "DiseaseArea",
    "MoAClass",
    "audit_pocket_hotspots",
    "fetch_rcsb_structure",
    "find_local_pocket_files",
    "get_cns_target",
    "list_cns_targets",
    "list_tri_disorder_targets",
    "load_cns_target_pocket",
    "register_cns_target",
    "unregister_cns_target",
]
