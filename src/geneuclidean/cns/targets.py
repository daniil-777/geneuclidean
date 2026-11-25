"""Curated 12-target human Tourette Syndrome & psychiatric 3D structural pharmacology registry.

Every UniProt accession, human single-protein ChEMBL Target ID (including verified fixes
``VMAT2 = CHEMBL1893`` and ``TAAR1 = CHEMBL5857``), RCSB PDB accession (using verified
human ``PDE10A`` ``5SJX``/``5SFE``/``3WYM`` rather than rat ``3HR1``), clinical reference
drug canonical SMILES, and canonical 3D binding-site hotspot residue has been cross-verified
against RCSB PDB, ChEMBL, PubChem, and RDKit.

All ligands generated or scored against these targets are strictly *in silico* research
hypotheses and are NOT clinically tested medicines or medical advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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

DEFAULT_CNS_STRUCTURES_DIR = Path("data/cns_neuro/structures")
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"


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

    @property
    def all_pdb_ids(self) -> tuple[str, ...]:
        """Return ordered tuple of primary + secondary RCSB PDB IDs."""
        return (self.primary_pdb_id, *self.secondary_pdb_ids)


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
            "9LLG (2.77 A, flupentixol) and 9URL (3.04 A, SKF-83566 benzazepine) provide "
            "inactive antagonist pockets; 9U5E (2.49 A) and 7LJC (3.00 A) are active controls."
        ),
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
        ),
        notes=(
            "Uses verified Homo sapiens Q9Y233 structures 5SJX (2.06 A MP-10), 5SFE (1.86 A), and "
            "3WYM (2.00 A TAK-063, -10 isoform numbering offset). Note: 3HR1 is Rattus norvegicus "
            "Q9QYJ6 and is excluded in favor of human 5SJX."
        ),
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
                name="Emraclidine",
                canonical_smiles="Cc1cc(C)c2c(n1)CN(C(=O)CC1CN(c3ccnc(C(F)(F)F)c3)C1)C2",
                clinical_status="Phase 2 (EMPOWER-1/2 missed primary endpoints Nov 2024)",
                role="Selective M4 positive allosteric modulator (PAM)",
            ),
        ),
        notes="8FX5 (2.45 A) captures dual orthosteric + extracellular allosteric xanomeline.",
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
    ),
}

# Alias map supporting gene symbols (SLC18A2, SLC6A4, SLC6A3) and ChEMBL IDs
_TARGET_ALIAS_MAP: dict[str, str] = {}
for _tid, _spec in CNS_TARGET_REGISTRY.items():
    _TARGET_ALIAS_MAP[_tid.upper()] = _tid
    _TARGET_ALIAS_MAP[_spec.gene_symbol.upper()] = _tid
    _TARGET_ALIAS_MAP[_spec.chembl_id.upper()] = _tid
    _TARGET_ALIAS_MAP[_spec.uniprot_id.upper()] = _tid


def get_cns_target(identifier: str) -> CNSTargetSpec:
    """Look up a ``CNSTargetSpec`` by target ID, gene symbol, ChEMBL ID, or UniProt ID."""
    key = identifier.strip().upper()
    resolved = _TARGET_ALIAS_MAP.get(key)
    if resolved is None:
        valid = ", ".join(sorted(CNS_TARGET_REGISTRY.keys()))
        raise KeyError(f"Unknown CNS target {identifier!r}. Registered targets: {valid}")
    return CNS_TARGET_REGISTRY[resolved]


def list_cns_targets(indication_filter: str | None = None) -> list[CNSTargetSpec]:
    """Return all 12 curated human CNS target specs, optionally filtered by indication keyword."""
    specs = list(CNS_TARGET_REGISTRY.values())
    if indication_filter is None or not indication_filter.strip():
        return specs
    needle = indication_filter.strip().lower()
    return [
        spec
        for spec in specs
        if any(needle in ind.lower() for ind in spec.indications)
        or needle in spec.therapeutic_moa.lower()
    ]


def find_local_pocket_files(
    target_id: str,
    *,
    pdb_id: str | None = None,
    structures_dir: str | Path = DEFAULT_CNS_STRUCTURES_DIR,
) -> tuple[Path, Path | None] | None:
    """Locate pre-cleaned ``*_pocket.pdb`` and ``*_lig.sdf`` under ``structures_dir``."""
    spec = get_cns_target(target_id)
    root = Path(structures_dir)
    clean_dir = root / "clean_pockets"
    candidate_pdbs = (pdb_id.strip().upper(),) if pdb_id else spec.all_pdb_ids

    if clean_dir.exists():
        for pid in candidate_pdbs:
            matches = sorted(clean_dir.glob(f"{pid}_{spec.target_id}_*_pocket.pdb"))
            if not matches:
                matches = sorted(clean_dir.glob(f"{pid}_*_pocket.pdb"))
            if matches:
                pocket_path = matches[0]
                sdf_candidate = Path(str(pocket_path).replace("_pocket.pdb", "_lig.sdf"))
                return pocket_path, (sdf_candidate if sdf_candidate.exists() else None)

    raw_dir = root / "raw_cif"
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

    Prefers pre-cleaned ligand-free pockets + CCD-bonded ligand SDFs from
    ``data/cns_neuro/structures/clean_pockets/``. If unavailable locally and
    ``allow_fetch=True``, downloads the requested RCSB PDB/mmCIF structure and safely
    strips non-protein HETATMs via ``featurizer._load_protein_heavy_atoms``.
    """
    spec = get_cns_target(target_id)
    chosen_pdb = (pdb_id or spec.primary_pdb_id).strip().upper()
    local = find_local_pocket_files(spec.target_id, pdb_id=pdb_id, structures_dir=structures_dir)

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
            )
        return featurize_pdb_pocket_inference(
            pocket_path,
            pocket_cutoff=pocket_cutoff,
            max_atoms=max_atoms,
            pdb_id=chosen_pdb.lower(),
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
    """Audit whether canonical pharmacological hotspot residues appear in a 3D pocket.

    If ``ligand_sdf`` is supplied, also verifies which hotspot residues (including known
    isoform-numbering aliases such as ``ADRA2A Asp128`` or ``PDE10A Gln716``) lie within
    ``contact_cutoff`` Angstroms of any ligand heavy atom.
    """
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
    anchor_keys = {
        target_spec.primary_anchor_residue.lower(),
        *(a.lower() for a in target_spec.hotspot_isoform_aliases[:1]),
    }

    present_hotspots: set[str] = set()
    contact_hotspots: set[str] = set()
    anchor_in_contact = False

    for chain in model:
        for residue in chain:
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
    }


__all__ = [
    "CNS_TARGET_REGISTRY",
    "DEFAULT_CNS_STRUCTURES_DIR",
    "RCSB_DOWNLOAD_URL",
    "CNSTargetSpec",
    "ClinicalReferenceDrug",
    "audit_pocket_hotspots",
    "fetch_rcsb_structure",
    "find_local_pocket_files",
    "get_cns_target",
    "list_cns_targets",
    "load_cns_target_pocket",
]
