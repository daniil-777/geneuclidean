"""Cortico-Striato-Thalamo-Cortical (CSTC) Basal Ganglia & GABAergic Tourette Circuit Module.

Extends GenEuclidean's Tourette Syndrome structural pharmacology with:

1. **Audited 13-Target CSTC Basal Ganglia & GABAergic Circuit Ontology**
   (:data:`TOURETTE_CSTC_CIRCUIT_AUDIT`): Maps all 9 existing Tourette-relevant targets
   (``DRD1``, ``DRD2``, ``VMAT2``, ``ADRA2A``, ``HRH3``, ``PDE10A``, ``MGLL``, ``PDE1B``,
   ``GPR52``) plus the 4 newly curated *Homo sapiens* GABAergic impulse-braking and
   Basal Ganglia signal-filtering pockets (``GABRA5``, ``GAT1``, ``ADORA2A``, ``GRM5``) to
   their exact anatomical CSTC node, conformational-state verification, clinical evidence
   grade, and pharmacological caveats.
2. **4 Verified New *Homo sapiens* GABAergic & Basal-Ganglia Signal-Filtering Targets**
   (:data:`TOURETTE_CIRCUIT_EXPANSION_REGISTRY`):
   - ``GABRA5`` (``8BHG``, 2.39 Å X-ray, multi-chain ``α5+/γ2−`` interface with Bretazenil;
     secondary native pentamer ``6X3X`` Diazepam): Extrasynaptic/striatal tonic GABAergic
     impulse brake sparing ``α1`` sedation.
   - ``GAT1`` (``SLC6A1``, ``7Y7Z``, 3.20 Å cryo-EM, 100% WT human with Tiagabine):
     Presynaptic/astrocytic GABA reuptake inhibitor compensating for the ~50% loss of
     striatal PV+ GABAergic fast-spiking interneurons in Tourette Syndrome.
   - ``ADORA2A`` (``9EE8``, 2.63 Å cryo-EM 2025, GPCRdb ``Active`` miniGs-coupled with
     endogenous Adenosine): Striatopallidal ``A2A–D2`` heteromer agonist strengthening the
     indirect "No-Go" basal-ganglia motor filter (explicitly distinguished from inactive
     Parkinson's antagonists ``5NM4``/``8GNG``).
   - ``GRM5`` (``mGluR5``, ``4OO9``, 2.60 Å X-ray, GPCRdb ``Inactive`` canonical numbering
     with Mavoglurant): 7TM allosteric negative allosteric modulator (NAM) dampening
     corticostriatal & subthalamic (STN) hyper-glutamatergic drive.
3. **Multi-Chain Interface Pocket Extractor** (:func:`extract_multichain_interface_pocket`):
   Extracts clean, ligand-free multi-chain interface binding pockets around a specific
   ``(chain_id, resname, resseq)`` ligand instance — essential for heteropentameric
   ``GABA_A`` receptors where distinct copies of the same benzodiazepine occupy either the
   canonical extracellular ``α+/γ−`` site or low-affinity transmembrane anesthetic sites.
4. **Mechanism-Aware Tourette Circuit Scorer** (:func:`score_tourette_circuit_candidate`):
   Combines industrial MedChem hygiene gating (:mod:`geneuclidean.cns.medchem_triage`),
   Pfizer CNS MPO v2 with mechanism-specific polarity notes (zwitterionic ``GAT1`` acids,
   nucleoside vs non-ribose ``ADORA2A`` agonists, neutral ``GRM5`` NAMs / ``GABRA5`` PAMs),
   12-endpoint v4.0 Avoid-ome/ADMET predictions, and 3D Vina + PLIF + anchor verification.

All generated or scored compounds are strictly *in silico* research hypotheses and are NOT
clinically tested medicines or medical advice.
"""

from __future__ import annotations

import gzip
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from Bio.PDB import PDBIO, MMCIFParser, PDBParser, Select
from rdkit import Chem

from geneuclidean.cns.admet import CNSAdmetSuite, PharmaAdmetPrediction
from geneuclidean.cns.medchem_triage import MedChemTriageResult, triage_molecule
from geneuclidean.cns.mpo import CNSMPOResult, compute_cns_mpo
from geneuclidean.cns.targets import (
    ClinicalReferenceDrug,
    CNSTargetSpec,
)
from geneuclidean.data.featurizer import (
    PocketFeatures,
    extract_pocket_from_files,
    featurize_pdb_pocket_inference,
)

DEFAULT_TS_CIRCUIT_DIR = Path("data/cns_neuro.nosync/ts_circuit_expansion")
DEFAULT_TS_CIRCUIT_STRUCTURES_DIR = DEFAULT_TS_CIRCUIT_DIR / "structures"

CSTCNode = Literal[
    "Direct D1-MSN 'Go'",
    "Indirect D2-MSN 'No-Go'",
    "Striatal GABAergic FSI / Tonic Brake",
    "Corticostriatal / STN Excitatory Filter",
    "Presynaptic Monoamine / Histamine / eCB Terminal",
    "Prefrontal Cortex Top-Down Gate",
]

ClinicalEvidenceGrade = Literal[
    "FDA Approved TS",
    "Phase 3 Win (NDA Priority Review)",
    "AAN First-Line Pediatric TS (Off-Label)",
    "Phase 2b Active",
    "Phase 2/3 Failed in TS",
    "Human Genetics / Phase 2 Failed",
    "Clinical Tool / Off-Label TS",
    "Preclinical / Phase 2 Hyperkinetic",
    "Preclinical",
]


@dataclass(frozen=True)
class TouretteCircuitAuditEntry:
    """Audited neuropharmacological and structural record for a Tourette CSTC target."""

    target_id: str
    gene_symbol: str
    uniprot_id: str
    chembl_id: str
    cstc_node: CSTCNode
    circuit_role: str
    therapeutic_polarity: str
    primary_pdb_id: str
    primary_pdb_resolution_angstrom: float
    conformational_state_verified: bool
    gpcrdb_or_structural_state: str
    clinical_evidence_grade: ClinicalEvidenceGrade
    audit_status: Literal["VERIFIED", "VERIFIED_WITH_CAVEAT"]
    audit_notes: str
    is_new_expansion: bool = False
    secondary_chembl_ids: tuple[str, ...] = ()
    multichain_interface: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize audit record to a JSON-compatible dictionary."""
        return {
            "target_id": self.target_id,
            "gene_symbol": self.gene_symbol,
            "uniprot_id": self.uniprot_id,
            "chembl_id": self.chembl_id,
            "secondary_chembl_ids": list(self.secondary_chembl_ids),
            "cstc_node": self.cstc_node,
            "circuit_role": self.circuit_role,
            "therapeutic_polarity": self.therapeutic_polarity,
            "primary_pdb_id": self.primary_pdb_id,
            "primary_pdb_resolution_angstrom": self.primary_pdb_resolution_angstrom,
            "conformational_state_verified": self.conformational_state_verified,
            "gpcrdb_or_structural_state": self.gpcrdb_or_structural_state,
            "clinical_evidence_grade": self.clinical_evidence_grade,
            "audit_status": self.audit_status,
            "audit_notes": self.audit_notes,
            "is_new_expansion": self.is_new_expansion,
            "multichain_interface": list(self.multichain_interface),
        }


# ---------------------------------------------------------------------------
# 1. 4 Verified New Homo sapiens GABAergic & Basal-Ganglia CSTC Target Specs
# ---------------------------------------------------------------------------

TOURETTE_CIRCUIT_EXPANSION_REGISTRY: dict[str, CNSTargetSpec] = {
    "GABRA5": CNSTargetSpec(
        target_id="GABRA5",
        gene_symbol="GABRA5",
        name="GABA-A Receptor alpha-5/gamma-2 Allosteric Interface",
        target_class="Cys-Loop Pentameric Ligand-Gated Cl- Channel (Multi-Chain Interface)",
        uniprot_id="P31644",
        chembl_id="CHEMBL5112",
        indications=(
            "Tourette Syndrome Striatal & Cortical Tonic GABAergic Impulse Brake",
            "Non-Sedating Subtype-Selective Benzodiazepine-Site Partial PAM",
            "Sensorimotor Gating / CSTC Disinhibition Rescue",
        ),
        therapeutic_moa=(
            "Partial positive allosteric modulator (PAM) at the extracellular alpha5+/gamma2- "
            "benzodiazepine cleft enhancing tonic Cl- inhibition while sparing alpha1 sedation"
        ),
        desired_state="Allosteric PAM-bound extracellular alpha5+/gamma2- heteromeric interface",
        primary_pdb_id="8BHG",
        secondary_pdb_ids=("6X3X", "7QNE", "8DD2"),
        primary_anchor_residue="His105",
        hotspot_residues=(
            "His105",
            "Phe77",
            "Thr142",
            "Ser209",
            "Tyr163",
            "Tyr213",
            "Tyr58",
            "Thr210",
            "Ile206",
            "Phe103",
        ),
        # Native alpha1 (6X3X/7QNE/8DD2) equivalent mature numbering aliases
        hotspot_isoform_aliases=("His102", "Ser205", "Tyr160", "Tyr210", "Phe100"),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Bretazenil",
                canonical_smiles="CC(C)(C)OC(=O)c1ncn2c1[C@@H]1CCCN1C(=O)c1c(Br)cccc1-2",
                clinical_status="Clinical partial BZD-site PAM (co-crystallized in 8BHG, 2.39 A)",
                role="Partial GABA-A alpha5+/gamma2- allosteric PAM",
            ),
            ClinicalReferenceDrug(
                name="Diazepam",
                canonical_smiles="CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
                clinical_status="FDA-approved classical BZD PAM (native alpha1+/gamma2- in 6X3X)",
                role="Full pan-alpha1/2/3/5 GABA-A PAM comparator",
            ),
            ClinicalReferenceDrug(
                name="Clonazepam",
                canonical_smiles="O=C1CN=C(c2ccccc2Cl)c2cc([N+](=O)[O-])ccc2N1",
                clinical_status="Clinical off-label adjunctive benzodiazepine for severe TS tics",
                role="High-potency clinical TS GABA-A PAM comparator",
            ),
            ClinicalReferenceDrug(
                name="Basmisanil",
                canonical_smiles="Cc1onc(-c2ccc(F)cc2)c1COc1ccc(C(=O)N2CCS(=O)(=O)CC2)cn1",
                clinical_status="Phase 2 alpha5-selective GABA-A ligand (in 8BHA, NAM polarity)",
                role="Alpha5-selective pocket-shape comparator (NAM control)",
            ),
        ),
        notes=(
            "8BHG (2.39 A X-ray, 0 pocket mutations) resolves Bretazenil (B:QMU502) at the exact "
            "multi-chain alpha5+ (Chain C, P31644) / gamma2- (Chain B, P18507) cleft; backed by "
            "CHEMBL5112 (524) + CHEMBL2094122 alpha5/beta3/gamma2 complex (861 human records). "
            "Note: Human GABRA6 (Q16445, DK-I-56-1 preclinical TS target) has 0 PDB structures."
        ),
        disease_area="tourette",
        moa_class="PAM",
        counter_targets=("HRH1",),
    ),
    "GAT1": CNSTargetSpec(
        target_id="GAT1",
        gene_symbol="SLC6A1",
        name="Sodium- and Chloride-Dependent GABA Transporter 1",
        target_class="Neurotransmitter Sodium Symporter (SLC6)",
        uniprot_id="P30531",
        chembl_id="CHEMBL1903",
        indications=(
            "Tourette Syndrome Synaptic GABA Reuptake Brake (PV+ FSI Interneuron Compensation)",
            "Epilepsy / Focal Motor Hyperexcitability (FDA-approved Tiagabine)",
            "CSTC Basal Ganglia Inhibitory Tone Restoration",
        ),
        therapeutic_moa=(
            "Selective GAT1 (SLC6A1) reuptake inhibitor elevating synaptic GABA in striatum, "
            "GPe/GPi, and motor cortex to slow aberrant tic impulses"
        ),
        desired_state="Inward-open inhibited (Tiagabine-locked S1 + vestibule)",
        primary_pdb_id="7Y7Z",
        secondary_pdb_ids=("7Y7W", "7Y7Y", "7SK2"),
        primary_anchor_residue="Gly65",
        hotspot_residues=(
            "Gly65",
            "Tyr60",
            "Ser302",
            "Phe294",
            "Gly59",
            "Gly63",
            "Ser396",
            "Leu303",
            "Cys399",
            "Tyr140",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Tiagabine",
                canonical_smiles="Cc1ccsc1C(=CCCN1CCC[C@@H](C(=O)O)C1)c1sccc1C",
                clinical_status="FDA-approved selective GAT1 inhibitor (co-crystallized in 7Y7Z)",
                role="Zwitterionic lipophilic nipecotic-acid GAT1 inhibitor",
            ),
            ClinicalReferenceDrug(
                name="SKF-89976A",
                canonical_smiles="O=C(O)C1CCCN(CCC=C(c2ccccc2)c2ccccc2)C1",
                clinical_status="Preclinical gold-standard selective GAT1 inhibitor tool",
                role="Diphenylbutenyl nipecotic-acid GAT1 inhibitor",
            ),
            ClinicalReferenceDrug(
                name="Nipecotic acid",
                canonical_smiles="O=C(O)[C@H]1CCCNC1",
                clinical_status="Polar orthosteric GAT1 substrate-inhibitor (in 7Y7Y, 2.40 A)",
                role="Core zwitterionic GABA-mimetic pharmacophore",
            ),
        ),
        notes=(
            "7Y7Z (3.20 A cryo-EM) is 100.0% wild-type human SLC6A1 (532/532 residues) bound to "
            "Tiagabine (TGI); 4th human SLC6 symporter in GenEuclidean alongside SERT, DAT, GLYT1."
        ),
        disease_area="tourette",
        moa_class="INHIBITOR",
        counter_targets=(),
    ),
    "ADORA2A": CNSTargetSpec(
        target_id="ADORA2A",
        gene_symbol="ADORA2A",
        name="Adenosine A2A Receptor (Active Agonist-Bound Indirect-Pathway Brake)",
        target_class="Purinergic GPCR (Class A)",
        uniprot_id="P29274",
        chembl_id="CHEMBL251",
        indications=(
            "Tourette Syndrome Striatopallidal A2A-D2 Heteromer Indirect 'No-Go' Motor Brake",
            "Hyperkinetic Movement Disorders (Antagonistic D2 Trans-Inhibition)",
        ),
        therapeutic_moa=(
            "CNS-penetrant active-state A2A receptor agonist (or partial agonist) activating "
            "Gs/Golf on striatopallidal D2-MSNs and allosterically trans-inhibiting DRD2 "
            "(OPPOSITE polarity to Parkinson's A2A antagonists like istradefylline!)"
        ),
        desired_state="Active (miniGs-coupled agonist-bound engaging Ser277/His278/Thr88 switch)",
        primary_pdb_id="9EE8",
        secondary_pdb_ids=("2YDV", "3QAK", "5G53"),
        primary_anchor_residue="Asn253",
        hotspot_residues=(
            "Asn253",
            "Ser277",
            "His278",
            "Phe168",
            "Thr88",
            "Glu169",
            "His250",
            "Trp246",
            "Leu249",
            "Ile274",
            "Met177",
        ),
        hotspot_isoform_aliases=(),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Adenosine",
                canonical_smiles="Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
                clinical_status="Endogenous physiological A2A agonist (co-crystallized in 9EE8)",
                role="Active-state ribose A2A agonist benchmark",
            ),
            ClinicalReferenceDrug(
                name="NECA",
                canonical_smiles="CCNC(=O)[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",
                clinical_status="Canonical high-affinity A2A agonist tool (in 2YDV & 5G53)",
                role="Active-state ribose-5'-amide A2A agonist",
            ),
            ClinicalReferenceDrug(
                name="Regadenoson",
                canonical_smiles=(
                    "CNC(=O)c1cnn(-c2nc(N)c3ncn([C@@H]4O[C@H](CO)[C@@H](O)[C@H]4O)c3n2)c1"
                ),
                clinical_status="FDA-approved peripheral A2A agonist (vascular stress; low BBB)",
                role="Clinical selective A2A agonist (highlights BBB & hypotension hurdle)",
            ),
            ClinicalReferenceDrug(
                name="LUF5833",
                canonical_smiles="N#Cc1c(N)nc(SCc2ncc[nH]2)c(C#N)c1-c1ccccc1",
                clinical_status="Non-ribose partial A2A agonist (in 7ARO; CNS-favorable MPO 4.12)",
                role="Non-ribose CNS-penetrant A2A partial-agonist archetype",
            ),
        ),
        notes=(
            "CRITICAL MoA PRECISION: 9EE8 (2.63 A cryo-EM, Apr 2025, GPCRdb=Active, miniGs-bound, "
            "0 pocket mutations) captures the active agonist conformation required to strengthen "
            "the striatal indirect 'No-Go' brake via A2A-D2 trans-inhibition. Inactive Parkinson's "
            "antagonist structures (5NM4 ZM241385, 8GNG istradefylline) are explicitly excluded. "
            "Translational caveat: systemic full A2A agonists cause hypotension/vasodilation, "
            "motivating non-ribose partial or CNS-biased A2A agonists."
        ),
        disease_area="tourette",
        moa_class="AGONIST",
        counter_targets=("DRD2",),
    ),
    "GRM5": CNSTargetSpec(
        target_id="GRM5",
        gene_symbol="GRM5",
        name="Metabotropic Glutamate Receptor 5 (mGluR5 7TM Allosteric NAM Site)",
        target_class="Class C Glutamate GPCR (7TM Allosteric Pocket)",
        uniprot_id="P41594",
        chembl_id="CHEMBL3227",
        indications=(
            "Tourette Syndrome & Hyperkinetic CSTC Corticostriatal / STN Glutamate Filter",
            "L-DOPA-Induced Dyskinesia (Dipraglurant / Mavoglurant Phase 2)",
            "Repetitive Motor / Compulsive Basal Ganglia Plasticity",
        ),
        therapeutic_moa=(
            "Selective 7TM negative allosteric modulator (NAM) dampening excessive corticostriatal "
            "and subthalamic (STN) glutamatergic excitation without blocking basal glutamate tone"
        ),
        desired_state="Inactive 7TM allosteric NAM-bound (Asn747/Ser809/Ser805 cleft)",
        primary_pdb_id="4OO9",
        secondary_pdb_ids=("6FFI", "6FFH", "5CGD"),
        primary_anchor_residue="Asn747",
        hotspot_residues=(
            "Asn747",
            "Ser809",
            "Ser805",
            "Tyr659",
            "Pro743",
            "Leu744",
            "Ser658",
            "Ala813",
            "Trp785",
            "Phe788",
        ),
        # +160 T4L-fusion numbering offset in 6FFI (2.20 A) and +1000 offset in 6FFH (2.65 A)
        hotspot_isoform_aliases=(
            "Ser969",
            "Ser965",
            "Trp945",
            "Phe948",
            "Ser1809",
            "Trp1785",
            "Phe1788",
        ),
        reference_drugs=(
            ClinicalReferenceDrug(
                name="Mavoglurant (AFQ056)",
                canonical_smiles="COC(=O)N1CC[C@@H]2[C@H]1CCC[C@@]2(O)C#Cc1cccc(C)c1",
                clinical_status="Phase 2 evaluated in LID & Fragile X (co-crystallized in 4OO9)",
                role="Selective clinical mGluR5 7TM NAM",
            ),
            ClinicalReferenceDrug(
                name="Dipraglurant (ADX48621)",
                canonical_smiles="Fc1ccc2nc(CCC#Cc3ccccn3)cn2c1",
                clinical_status="Phase 2 positive anti-dyskinetic signal in PD-LID (Addex)",
                role="Fast-acting CNS mGluR5 NAM for hyperkinetic disorders",
            ),
            ClinicalReferenceDrug(
                name="Basimglurant",
                canonical_smiles="Cc1nc(C#Cc2ccnc(Cl)c2)c(C)n1-c1ccc(F)cc1",
                clinical_status="Phase 2 CNS-penetrant mGluR5 NAM (Roche)",
                role="High-selectivity mGluR5 7TM NAM",
            ),
            ClinicalReferenceDrug(
                name="Fenobam",
                canonical_smiles="CN1CC(=O)NC1=NC(=O)Nc1cccc(Cl)c1",
                clinical_status="Clinical non-alkyne mGluR5 NAM (co-crystallized in 6FFH)",
                role="Non-acetylene mGluR5 NAM scaffold (avoids alkyne metabolic alert)",
            ),
        ),
        notes=(
            "4OO9 (2.60 A X-ray, GPCRdb=Inactive, 0 pocket mutations) preserves canonical UniProt "
            "P41594 numbering (Asn747 2.45 A, Ser809 2.88 A), whereas 6FFI (+160) and 6FFH "
            "(+1000) carry T4L-fusion TM6/7 offsets mapped via hotspot_isoform_aliases. Circuit "
            "duality note: mGluR5 NAMs dampen corticostriatal/STN hyper-excitation but also "
            "attenuate A2A-mGluR5 synergy in striatopallidal heteromers."
        ),
        disease_area="tourette",
        moa_class="NAM",
        counter_targets=(),
    ),
}

# ---------------------------------------------------------------------------
# 2. Complete 13-Target Tourette CSTC Circuit Independent Validity Audit Table
# ---------------------------------------------------------------------------

TOURETTE_CSTC_CIRCUIT_AUDIT: dict[str, TouretteCircuitAuditEntry] = {
    "DRD1": TouretteCircuitAuditEntry(
        target_id="DRD1",
        gene_symbol="DRD1",
        uniprot_id="P21728",
        chembl_id="CHEMBL2056",
        cstc_node="Direct D1-MSN 'Go'",
        circuit_role=(
            "Postsynaptic striatonigral D1-MSN accelerator; blockade dampens tic execution"
        ),
        therapeutic_polarity="ANTAGONIST (Inactive state)",
        primary_pdb_id="9LLG",
        primary_pdb_resolution_angstrom=2.77,
        conformational_state_verified=True,
        gpcrdb_or_structural_state="GPCRdb=Inactive (9LLG flupentixol + NAM GEM, no G-protein)",
        clinical_evidence_grade="Phase 3 Win (NDA Priority Review)",
        audit_status="VERIFIED_WITH_CAVEAT",
        audit_notes=(
            "Primary 9LLG (2.77 A) is confirmed GPCRdb=Inactive antagonist-bound (valid!). "
            "CAVEAT on secondary PDB 9URL (3.04 A, SKF-83566): live RCSB & coordinate audit "
            "proves 9URL is a Gs-coupled DRD1-Gs complex (TM3 Arg121-TM6 Thr268 CA = 20.09 A vs "
            "19.46 A in active 7LJC), so 9LLG—not 9URL—must be used for inactive-state docking."
        ),
    ),
    "DRD2": TouretteCircuitAuditEntry(
        target_id="DRD2",
        gene_symbol="DRD2",
        uniprot_id="P14416",
        chembl_id="CHEMBL217",
        cstc_node="Indirect D2-MSN 'No-Go'",
        circuit_role="Postsynaptic striatopallidal Gi/o brake-suppressor; blockade restores No-Go",
        therapeutic_polarity="ANTAGONIST / PARTIAL_AGONIST (Inactive state; anti-EPS counter)",
        primary_pdb_id="6CM4",
        primary_pdb_resolution_angstrom=2.87,
        conformational_state_verified=True,
        gpcrdb_or_structural_state="GPCRdb=Inactive (6CM4 risperidone; 6LUQ 3.10 A haloperidol)",
        clinical_evidence_grade="FDA Approved TS",
        audit_status="VERIFIED",
        audit_notes=(
            "Clinically gold-standard TS mechanism (haloperidol, pimozide, aripiprazole FDA "
            "approved for TS), but >80% striatal D2 occupancy drives EPS, tardive dyskinesia, "
            "and hyperprolactinemia; serves dual role as benchmark & anti-EPS counter-target."
        ),
    ),
    "VMAT2": TouretteCircuitAuditEntry(
        target_id="VMAT2",
        gene_symbol="SLC18A2",
        uniprot_id="Q05940",
        chembl_id="CHEMBL1893",
        cstc_node="Presynaptic Monoamine / Histamine / eCB Terminal",
        circuit_role=(
            "Presynaptic vesicular dopamine/monoamine loader; inhibition depletes DA release"
        ),
        therapeutic_polarity="INHIBITOR (Lumen-facing occluded/inhibited)",
        primary_pdb_id="8T69",
        primary_pdb_resolution_angstrom=2.89,
        conformational_state_verified=True,
        gpcrdb_or_structural_state="Lumen-facing tetrabenazine-inhibited (8T69; 9KQ8 valbenazine)",
        clinical_evidence_grade="Phase 2/3 Failed in TS",
        audit_status="VERIFIED_WITH_CAVEAT",
        audit_notes=(
            "Structurally 100% valid human Q05940/CHEMBL1893 (8T69 2.89 A, 9KQ8 3.38 A). "
            "CRITICAL CLINICAL AUDIT: Both valbenazine (T-Force GREEN/GOLD, T-Forward) and "
            "deutetrabenazine (ARTISTS 1 & 2) FAILED primary YGTSS-TTS endpoints in TS trials "
            "(FDA-approved for TD & Huntington chorea; off-label/exploratory in TS)."
        ),
    ),
    "ADRA2A": TouretteCircuitAuditEntry(
        target_id="ADRA2A",
        gene_symbol="ADRA2A",
        uniprot_id="P08913",
        chembl_id="CHEMBL1867",
        cstc_node="Prefrontal Cortex Top-Down Gate",
        circuit_role=(
            "Postsynaptic PFC dendritic spine Gi/o closure of HCN channels -> top-down CSTC gating"
        ),
        therapeutic_polarity="AGONIST (Active Gi/o-coupled state)",
        primary_pdb_id="7EJ8",
        primary_pdb_resolution_angstrom=3.00,
        conformational_state_verified=True,
        gpcrdb_or_structural_state=(
            "GPCRdb=Active (7EJ8 brimonidine + GoA; 9CBL 2.80 A epinephrine)"
        ),
        clinical_evidence_grade="AAN First-Line Pediatric TS (Off-Label)",
        audit_status="VERIFIED_WITH_CAVEAT",
        audit_notes=(
            "7EJ8 (3.00 A) & 9CBL (2.80 A) are verified Active agonist-bound templates (+15 "
            "numbering offset properly aliased: Asp128=Asp113). Anatomically PFC top-down gate "
            "(NOT striatal!). Note: secondary PDB 6KUX (RS79948) is an Inactive antagonist control."
        ),
    ),
    "HRH3": TouretteCircuitAuditEntry(
        target_id="HRH3",
        gene_symbol="HRH3",
        uniprot_id="Q9Y5N1",
        chembl_id="CHEMBL264",
        cstc_node="Presynaptic Monoamine / Histamine / eCB Terminal",
        circuit_role="Presynaptic histaminergic Gi/o autoreceptor & striatal MSN heteroreceptor",
        therapeutic_polarity="INVERSE_AGONIST / ANTAGONIST (Inactive state -> boosts histamine)",
        primary_pdb_id="7F61",
        primary_pdb_resolution_angstrom=2.60,
        conformational_state_verified=True,
        gpcrdb_or_structural_state="GPCRdb=Inactive (7F61 PF-03654746 antagonist)",
        clinical_evidence_grade="Human Genetics / Phase 2 Failed",
        audit_status="VERIFIED_WITH_CAVEAT",
        audit_notes=(
            "Polarity VERIFIED: HDC W317X familial TS causes central histamine deficiency, so "
            "blocking presynaptic HRH3 (Inactive 7F61) increases histamine release. CAVEATS: "
            "(1) Secondary PDBs 8YN5/8YN8 are Active agonist controls, not antagonist templates; "
            "(2) AZD5213 FAILED Ph2 in adolescent TS (NCT01904773); pitolisant lacks Ph3 TS RCT."
        ),
    ),
    "PDE10A": TouretteCircuitAuditEntry(
        target_id="PDE10A",
        gene_symbol="PDE10A",
        uniprot_id="Q9Y233",
        chembl_id="CHEMBL4409",
        cstc_node="Indirect D2-MSN 'No-Go'",
        circuit_role=(
            "Striatal D1/D2-MSN cAMP/cGMP hydrolase; inhibition preferentially activates D2 No-Go"
        ),
        therapeutic_polarity="INHIBITOR (Catalytic Q2-pocket engaged)",
        primary_pdb_id="5SJX",
        primary_pdb_resolution_angstrom=2.06,
        conformational_state_verified=True,
        gpcrdb_or_structural_state="Human Q9Y233 catalytic inhibited (5SJX MP-10, Zn2+/Mg2+)",
        clinical_evidence_grade="Phase 2b Active",
        audit_status="VERIFIED",
        audit_notes=(
            "Verified Homo sapiens Q9Y233 (rat 3HR1 excluded); strongest active non-D1 clinical TS "
            "pipeline: gemlapodect (NOE-105) met Phase 2a ALLEVIA-1 endpoints and is in global "
            "double-blind Phase 2b ALLEVIA-2."
        ),
    ),
    "MGLL": TouretteCircuitAuditEntry(
        target_id="MGLL",
        gene_symbol="MGLL",
        uniprot_id="Q99685",
        chembl_id="CHEMBL4191",
        cstc_node="Presynaptic Monoamine / Histamine / eCB Terminal",
        circuit_role=(
            "Degrades striatal 2-AG; reversible inhibition elevates CB1 retrograde synaptic braking"
        ),
        therapeutic_polarity="INHIBITOR (Reversible non-covalent catalytic)",
        primary_pdb_id="9F8D",
        primary_pdb_resolution_angstrom=1.14,
        conformational_state_verified=True,
        gpcrdb_or_structural_state=(
            "1.14 A human non-covalent reversible catalytic inhibited (9F8D)"
        ),
        clinical_evidence_grade="Phase 2/3 Failed in TS",
        audit_status="VERIFIED_WITH_CAVEAT",
        audit_notes=(
            "9F8D (1.14 A) is structurally pristine and strictly non-covalent. CLINICAL CAVEAT: "
            "Irreversible covalent MAGL inhibitor Lu AG06466 (elcubragistat) failed Phase 2a in "
            "adult TS (attributed partly to CB1 receptor desensitization); reversible MAGL remains "
            "an unproven translational hypothesis."
        ),
    ),
    "PDE1B": TouretteCircuitAuditEntry(
        target_id="PDE1B",
        gene_symbol="PDE1B",
        uniprot_id="Q01064",
        chembl_id="CHEMBL4425",
        cstc_node="Direct D1-MSN 'Go'",
        circuit_role="Ca2+/calmodulin-activated striatal D1-MSN & PFC cAMP/cGMP modulator",
        therapeutic_polarity="INHIBITOR (Catalytic Zn2+/Mg2+ cleft)",
        primary_pdb_id="5UOY",
        primary_pdb_resolution_angstrom=1.82,
        conformational_state_verified=True,
        gpcrdb_or_structural_state="1.82 A human Q01064 catalytic inhibited (5UOY, Zn2+/Mg2+)",
        clinical_evidence_grade="Preclinical",
        audit_status="VERIFIED_WITH_CAVEAT",
        audit_notes=(
            "High-resolution human pocket (5UOY 1.82 A); however, PDE1B inhibition elevates cAMP "
            "in D1-MSNs (potentially enhancing D1 signaling) and lacks clinical TS efficacy data "
            "(primary rationale is CIAS/cognition and PDE10A selectivity profiling)."
        ),
    ),
    "GPR52": TouretteCircuitAuditEntry(
        target_id="GPR52",
        gene_symbol="GPR52",
        uniprot_id="Q9Y2T5",
        chembl_id="CHEMBL3297639",
        cstc_node="Indirect D2-MSN 'No-Go'",
        circuit_role="Gs-coupled striatal D2-MSN orphan GPCR opposing Gi-coupled DRD2 signaling",
        therapeutic_polarity="AGONIST (Active Gs-coupled side-pocket)",
        primary_pdb_id="8HMP",
        primary_pdb_resolution_angstrom=2.77,
        conformational_state_verified=True,
        gpcrdb_or_structural_state="GPCRdb=Active (8HMP Gs-coupled c17; avoids inactive 6LI0)",
        clinical_evidence_grade="Preclinical",
        audit_status="VERIFIED",
        audit_notes=(
            "8HMP (2.77 A) correctly uses active Gs-coupled conformation (avoiding inactive X-ray "
            "6LI0). Strong mechanistic fit for non-D2 striatal No-Go brake (HTL0048149 Ph1/2 SCZ)."
        ),
    ),
    # --- 4 Newly Curated GABAergic Impulse-Braking & Basal Ganglia Signal-Filtering Targets ---
    "GABRA5": TouretteCircuitAuditEntry(
        target_id="GABRA5",
        gene_symbol="GABRA5",
        uniprot_id="P31644",
        chembl_id="CHEMBL5112",
        secondary_chembl_ids=("CHEMBL2094122",),
        cstc_node="Striatal GABAergic FSI / Tonic Brake",
        circuit_role=(
            "Heteropentameric GABA-A alpha5+/gamma2- Cl- channel mediating tonic GABAergic "
            "impulse braking without alpha1 full-agonist sedation"
        ),
        therapeutic_polarity="PAM (Partial positive allosteric modulator at alpha5+/gamma2-)",
        primary_pdb_id="8BHG",
        primary_pdb_resolution_angstrom=2.39,
        conformational_state_verified=True,
        gpcrdb_or_structural_state=(
            "2.39 A X-ray multi-chain alpha5+(C)/gamma2-(B) Bretazenil partial PAM (0 pocket muts)"
        ),
        clinical_evidence_grade="Clinical Tool / Off-Label TS",
        audit_status="VERIFIED",
        audit_notes=(
            "Directly addresses the ~50% striatal PV+ GABAergic interneuron deficit in TS "
            "(Kalanithi PNAS 2005; Kataoka J. Neurosci. 2010) and clinical clonazepam efficacy. "
            "Multi-chain extractor isolates canonical B:QMU502 interface (and 6X3X D:DZP404 vs "
            "TMD anesthetic copies). Human GABRA6 (Q16445, DK-I-56-1) has 0 experimental PDBs."
        ),
        is_new_expansion=True,
        multichain_interface=("C", "B"),
    ),
    "GAT1": TouretteCircuitAuditEntry(
        target_id="GAT1",
        gene_symbol="SLC6A1",
        uniprot_id="P30531",
        chembl_id="CHEMBL1903",
        cstc_node="Striatal GABAergic FSI / Tonic Brake",
        circuit_role=(
            "Presynaptic/astrocytic synaptic GABA clearance transporter; blockade prolongs "
            "inhibitory GABAergic impulse braking in striatum, GPe/GPi, and motor cortex"
        ),
        therapeutic_polarity="INHIBITOR (Inward-open Tiagabine-locked state)",
        primary_pdb_id="7Y7Z",
        primary_pdb_resolution_angstrom=3.20,
        conformational_state_verified=True,
        gpcrdb_or_structural_state="3.20 A cryo-EM 100% WT human SLC6A1 + FDA drug Tiagabine (TGI)",
        clinical_evidence_grade="Clinical Tool / Off-Label TS",
        audit_status="VERIFIED",
        audit_notes=(
            "532/532 residues 100% WT Homo sapiens P30531; directly elevates endogenous synaptic "
            "GABA (the 'acid that slows down impulses'). Preferentially chosen over ABAT (0 human "
            "PDBs; all 9 PDBs are Sus scrofa pig P80147) and complements GABA-A PAM modulation."
        ),
        is_new_expansion=True,
        multichain_interface=("A",),
    ),
    "ADORA2A": TouretteCircuitAuditEntry(
        target_id="ADORA2A",
        gene_symbol="ADORA2A",
        uniprot_id="P29274",
        chembl_id="CHEMBL251",
        cstc_node="Indirect D2-MSN 'No-Go'",
        circuit_role=(
            "Striatopallidal D2-MSN Gs/Golf receptor forming antagonistic A2A-D2 heteromers; "
            "agonism allosterically trans-inhibits DRD2 and strengthens the No-Go motor filter"
        ),
        therapeutic_polarity="AGONIST (Active miniGs-coupled state — OPPOSITE of Parkinson's!)",
        primary_pdb_id="9EE8",
        primary_pdb_resolution_angstrom=2.63,
        conformational_state_verified=True,
        gpcrdb_or_structural_state=(
            "GPCRdb=Active (9EE8 2.63 A Apr 2025 cryo-EM miniGs + Adenosine, 0 pocket mutations)"
        ),
        clinical_evidence_grade="Preclinical / Phase 2 Hyperkinetic",
        audit_status="VERIFIED_WITH_CAVEAT",
        audit_notes=(
            "Ultra-clean 2025 active G-protein-coupled human pocket (avoids Q89A mutation in 2YDV "
            "and excludes inactive Parkinson's antagonists 5NM4/8GNG). TRANSLATIONAL CAVEAT: "
            "systemic full ribose A2A agonists cause hypotension/vasodilation and poor BBB entry "
            "(MPO 3.00), motivating CNS-penetrant non-ribose partial agonists (e.g., LUF5833)."
        ),
        is_new_expansion=True,
        multichain_interface=("A",),
    ),
    "GRM5": TouretteCircuitAuditEntry(
        target_id="GRM5",
        gene_symbol="GRM5",
        uniprot_id="P41594",
        chembl_id="CHEMBL3227",
        cstc_node="Corticostriatal / STN Excitatory Filter",
        circuit_role=(
            "Postsynaptic corticostriatal & subthalamic (STN) Gq/11 glutamate receptor; NAMs "
            "dampen hyper-glutamatergic tic/dyskinetic drive while preserving baseline tone"
        ),
        therapeutic_polarity="NAM (Inactive 7TM allosteric pocket)",
        primary_pdb_id="4OO9",
        primary_pdb_resolution_angstrom=2.60,
        conformational_state_verified=True,
        gpcrdb_or_structural_state=(
            "GPCRdb=Inactive (4OO9 2.60 A X-ray Mavoglurant, canonical P41594 numbering)"
        ),
        clinical_evidence_grade="Preclinical / Phase 2 Hyperkinetic",
        audit_status="VERIFIED_WITH_CAVEAT",
        audit_notes=(
            "4OO9 avoids T4L numbering offsets (+160 in 6FFI, +1000 in 6FFH) and has 4,320 human "
            "ChEMBL 37 activities. CAVEAT: Clinical PD-LID signals (dipraglurant/mavoglurant) "
            "have not yet been validated in TS RCTs; alkyne NAMs require GSH-adduct de-risking."
        ),
        is_new_expansion=True,
        multichain_interface=("A",),
    ),
}

# Alias map for the 4 new CSTC / GABA targets
_TS_CIRCUIT_ALIAS_MAP: dict[str, str] = {}
for _tid, _spec in TOURETTE_CIRCUIT_EXPANSION_REGISTRY.items():
    _TS_CIRCUIT_ALIAS_MAP[_tid.upper()] = _tid
    _TS_CIRCUIT_ALIAS_MAP[_spec.gene_symbol.upper()] = _tid
    _TS_CIRCUIT_ALIAS_MAP[_spec.chembl_id.upper()] = _tid
    _TS_CIRCUIT_ALIAS_MAP[_spec.uniprot_id.upper()] = _tid
_TS_CIRCUIT_ALIAS_MAP.update(
    {
        "GABA-A": "GABRA5",
        "GABAA": "GABRA5",
        "GABA-A5": "GABRA5",
        "CHEMBL2094122": "GABRA5",
        "GAT-1": "GAT1",
        "A2A": "ADORA2A",
        "A2AR": "ADORA2A",
        "MGLUR5": "GRM5",
        "MGLU5": "GRM5",
    }
)


def get_tourette_circuit_target(identifier: str) -> CNSTargetSpec:
    """Resolve one of the 4 verified new GABAergic / Basal-Ganglia CSTC targets."""
    key = identifier.strip().upper()
    resolved = _TS_CIRCUIT_ALIAS_MAP.get(key)
    if resolved is None:
        valid = ", ".join(sorted(TOURETTE_CIRCUIT_EXPANSION_REGISTRY.keys()))
        raise KeyError(f"Unknown Tourette CSTC expansion target {identifier!r}. Available: {valid}")
    return TOURETTE_CIRCUIT_EXPANSION_REGISTRY[resolved]


def list_tourette_circuit_targets(
    *,
    cstc_node: CSTCNode | None = None,
) -> list[CNSTargetSpec]:
    """Return the 4 newly curated GABAergic / Basal-Ganglia CSTC target specifications."""
    specs = list(TOURETTE_CIRCUIT_EXPANSION_REGISTRY.values())
    if cstc_node is None:
        return specs
    return [
        spec for spec in specs if TOURETTE_CSTC_CIRCUIT_AUDIT[spec.target_id].cstc_node == cstc_node
    ]


def get_cstc_circuit_audit(target_id: str | None = None) -> dict[str, TouretteCircuitAuditEntry]:
    """Return the full 13-target Tourette CSTC audit table, or a single target's entry."""
    if target_id is None:
        return dict(TOURETTE_CSTC_CIRCUIT_AUDIT)
    key = target_id.strip().upper()
    resolved = _TS_CIRCUIT_ALIAS_MAP.get(key, key)
    if resolved not in TOURETTE_CSTC_CIRCUIT_AUDIT:
        valid = ", ".join(sorted(TOURETTE_CSTC_CIRCUIT_AUDIT.keys()))
        raise KeyError(f"Unknown Tourette CSTC audit target {target_id!r}. Valid: {valid}")
    return {resolved: TOURETTE_CSTC_CIRCUIT_AUDIT[resolved]}


# ---------------------------------------------------------------------------
# 3. Multi-Chain Interface Pocket Extractor (Heteropentameric GABA-A & More)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiChainPocketExtractionResult:
    """Structured summary of a multi-chain ligand-free pocket extraction."""

    output_pocket_pdb: Path
    ligand_chain: str
    ligand_resname: str
    ligand_resseq: int
    ligand_heavy_atoms: int
    retained_chains: tuple[str, ...]
    residues_per_chain: dict[str, int]
    total_heavy_atoms: int
    interface_contacts_4p5a: dict[str, tuple[str, ...]]
    is_multichain_interface: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize extraction metadata to a JSON-compatible dictionary."""
        return {
            "output_pocket_pdb": str(self.output_pocket_pdb),
            "ligand_chain": self.ligand_chain,
            "ligand_resname": self.ligand_resname,
            "ligand_resseq": self.ligand_resseq,
            "ligand_heavy_atoms": self.ligand_heavy_atoms,
            "retained_chains": list(self.retained_chains),
            "residues_per_chain": dict(self.residues_per_chain),
            "total_heavy_atoms": self.total_heavy_atoms,
            "interface_contacts_4p5a": {
                ch: list(res_list) for ch, res_list in self.interface_contacts_4p5a.items()
            },
            "is_multichain_interface": self.is_multichain_interface,
        }


class _MultiChainInterfaceSelect(Select):
    """Biopython selector retaining only ligand-free polymer + whitelisted cofactors in shell."""

    def __init__(
        self,
        keep_chains: set[str],
        lig_coords: np.ndarray,
        pocket_cutoff: float,
        keep_cofactors: set[str],
        exclude_lig: tuple[str, str, int],
    ) -> None:
        self._keep_chains = keep_chains
        self._lig_coords = lig_coords
        self._cutoff = float(pocket_cutoff)
        self._keep_cofactors = keep_cofactors
        self._exclude_lig = exclude_lig

    def accept_chain(self, chain: Any) -> int:
        return 1 if str(chain.id) in self._keep_chains else 0

    def accept_residue(self, residue: Any) -> int:
        ch_id = str(residue.get_parent().id)
        rname = str(residue.get_resname()).strip().upper()
        rseq = int(residue.id[1])
        if (ch_id, rname, rseq) == self._exclude_lig:
            return 0
        is_polymer = str(residue.id[0]) == " "
        is_cofactor = rname in self._keep_cofactors
        if not (is_polymer or is_cofactor):
            return 0
        pts = [
            atom.get_coord()
            for atom in residue.get_atoms()
            if str(getattr(atom, "element", "") or "").strip().upper() not in ("H", "D")
        ]
        if not pts:
            return 0
        arr = np.asarray(pts, dtype=np.float64)
        dmin = float(np.linalg.norm(arr[:, None, :] - self._lig_coords[None, :, :], axis=-1).min())
        return 1 if dmin <= self._cutoff else 0

    def accept_atom(self, atom: Any) -> int:
        elem = str(getattr(atom, "element", "") or "").strip().upper()
        if elem in ("H", "D"):
            return 0
        altloc = str(atom.get_altloc())
        return 1 if altloc in (" ", "A", "1") else 0


def _load_biopython_structure(structure_path: Path) -> Any:
    """Parse ``.pdb``, ``.cif``, ``.mmcif``, or gzipped ``.cif.gz``/``.pdb.gz`` structure."""
    name_low = structure_path.name.lower()
    is_gz = name_low.endswith(".gz")
    is_cif = ".cif" in name_low or ".mmcif" in name_low
    parser: Any = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    if is_gz:
        with gzip.open(structure_path, "rt", encoding="utf-8") as fh:
            return parser.get_structure(structure_path.stem, fh)
    return parser.get_structure(structure_path.stem, str(structure_path))


def extract_multichain_interface_pocket(
    structure_path: str | Path,
    output_pocket_pdb: str | Path,
    *,
    ligand_resname: str,
    ligand_chain: str | None = None,
    ligand_resseq: int | None = None,
    keep_chains: Sequence[str] | None = None,
    pocket_cutoff: float = 14.0,
    interface_contact_cutoff: float = 4.5,
    keep_cofactors: Sequence[str] = (),
) -> MultiChainPocketExtractionResult:
    """Extract a clean ligand-free multi-chain interface pocket around a specific ligand copy.

    Designed for heteromeric oligomeric channels such as pentameric ``GABA_A`` receptors
    (``α+/γ−`` extracellular benzodiazepine cleft in ``8BHG`` or ``6X3X``), where multiple
    copies of the same small molecule occupy chemically distinct interfaces.

    Args:
        structure_path: Input ``.pdb``, ``.cif``, or ``.cif.gz`` atomic coordinate file.
        output_pocket_pdb: Destination path for the clean ligand-free pocket ``.pdb``.
        ligand_resname: 3-to-5 character CCD residue name of the bound ligand (e.g., ``"QMU"``).
        ligand_chain: Optional chain ID disambiguating multi-copy ligands (e.g., ``"B"``).
        ligand_resseq: Optional integer residue sequence ID disambiguating copies (e.g., ``502``).
        keep_chains: Explicit polymer chains to include (e.g., ``("C", "B")``). If ``None``,
            automatically retains every polymer chain with any heavy atom within
            ``interface_contact_cutoff`` Å of the selected ligand copy.
        pocket_cutoff: Spherical heavy-atom residue shell radius in Ångströms (default 14.0 Å).
        interface_contact_cutoff: Contact threshold in Ångströms for reporting interface residues.
        keep_cofactors: Optional integral cofactor residue names (e.g., ``("FAD",)``) to retain.

    Returns:
        :class:`MultiChainPocketExtractionResult` with per-chain residue and contact counts.
    """
    src_path = Path(structure_path)
    if not src_path.is_file():
        raise FileNotFoundError(f"Structure file not found: {src_path}")

    struct = _load_biopython_structure(src_path)
    model = struct[0]
    target_resname = ligand_resname.strip().upper()

    chosen_lig: tuple[str, int, Any] | None = None
    for chain in model:
        if ligand_chain is not None and str(chain.id) != ligand_chain:
            continue
        for res in chain:
            if str(res.get_resname()).strip().upper() != target_resname:
                continue
            if ligand_resseq is not None and int(res.id[1]) != int(ligand_resseq):
                continue
            chosen_lig = (str(chain.id), int(res.id[1]), res)
            break
        if chosen_lig is not None:
            break

    if chosen_lig is None:
        raise ValueError(
            f"Ligand {target_resname!r} (chain={ligand_chain!r}, resseq={ligand_resseq!r}) "
            f"not found in {src_path}"
        )

    lig_ch, lig_seq, lig_res = chosen_lig
    lig_pts = np.asarray(
        [
            atom.get_coord()
            for atom in lig_res.get_atoms()
            if str(getattr(atom, "element", "") or "").strip().upper() not in ("H", "D")
        ],
        dtype=np.float64,
    )
    if len(lig_pts) == 0:
        raise ValueError(f"Selected ligand {lig_ch}:{target_resname}{lig_seq} has 0 heavy atoms.")

    # Auto-discover interacting polymer chains when keep_chains is omitted
    contacts_by_chain: dict[str, list[tuple[float, str]]] = {}
    for chain in model:
        ch_id = str(chain.id)
        if keep_chains is not None and ch_id not in keep_chains:
            continue
        for res in chain:
            if str(res.id[0]) != " ":
                continue
            rpts = [
                atom.get_coord()
                for atom in res.get_atoms()
                if str(getattr(atom, "element", "") or "").strip().upper() not in ("H", "D")
            ]
            if not rpts:
                continue
            dmin = float(
                np.linalg.norm(
                    np.asarray(rpts, dtype=np.float64)[:, None, :] - lig_pts[None, :, :],
                    axis=-1,
                ).min()
            )
            if dmin <= float(interface_contact_cutoff):
                lbl = f"{str(res.get_resname()).strip().capitalize()}{int(res.id[1])}"
                contacts_by_chain.setdefault(ch_id, []).append((dmin, lbl))

    selected_chains = (
        {str(c) for c in keep_chains} if keep_chains is not None else set(contacts_by_chain.keys())
    )
    if not selected_chains:
        raise ValueError(
            f"No polymer chains found within {interface_contact_cutoff:.1f} A of "
            f"{lig_ch}:{target_resname}{lig_seq} in {src_path}"
        )

    out_path = Path(output_pocket_pdb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selector = _MultiChainInterfaceSelect(
        keep_chains=selected_chains,
        lig_coords=lig_pts,
        pocket_cutoff=pocket_cutoff,
        keep_cofactors={c.strip().upper() for c in keep_cofactors},
        exclude_lig=(lig_ch, target_resname, lig_seq),
    )
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(out_path), selector)

    # Re-parse written pocket PDB to report exact retained atom/residue statistics
    written = PDBParser(QUIET=True).get_structure("pocket", str(out_path))[0]
    res_per_chain: dict[str, int] = {}
    total_atoms = 0
    for chain in written:
        r_count = 0
        for res in chain:
            r_count += 1
            total_atoms += sum(1 for _ in res.get_atoms())
        if r_count > 0:
            res_per_chain[str(chain.id)] = r_count

    sorted_contacts: dict[str, tuple[str, ...]] = {
        ch: tuple(lbl for _, lbl in sorted(items))
        for ch, items in sorted(contacts_by_chain.items())
        if ch in res_per_chain
    }
    retained_sorted = tuple(sorted(res_per_chain.keys()))
    return MultiChainPocketExtractionResult(
        output_pocket_pdb=out_path,
        ligand_chain=lig_ch,
        ligand_resname=target_resname,
        ligand_resseq=lig_seq,
        ligand_heavy_atoms=int(len(lig_pts)),
        retained_chains=retained_sorted,
        residues_per_chain=res_per_chain,
        total_heavy_atoms=total_atoms,
        interface_contacts_4p5a=sorted_contacts,
        is_multichain_interface=len(retained_sorted) >= 2,
    )


def find_local_ts_circuit_pocket_files(
    target_id: str,
    *,
    pdb_id: str | None = None,
    structures_dir: str | Path = DEFAULT_TS_CIRCUIT_STRUCTURES_DIR,
) -> tuple[Path, Path | None] | None:
    """Locate curated ``*_pocket.pdb`` and ``*_lig.sdf`` for a Tourette CSTC expansion target."""
    spec = get_tourette_circuit_target(target_id)
    clean_dir = Path(structures_dir) / "clean_pockets"
    if not clean_dir.exists():
        return None
    candidate_pdbs = (pdb_id.strip().upper(),) if pdb_id else spec.all_pdb_ids
    for pid in candidate_pdbs:
        matches = sorted(clean_dir.glob(f"{pid}_{spec.target_id}_*_pocket.pdb"))
        if matches:
            pocket_path = matches[0]
            sdf_path = Path(str(pocket_path).replace("_pocket.pdb", "_lig.sdf"))
            return pocket_path, (sdf_path if sdf_path.exists() else None)
    return None


def load_tourette_circuit_pocket(
    target_id: str,
    *,
    pdb_id: str | None = None,
    structures_dir: str | Path = DEFAULT_TS_CIRCUIT_STRUCTURES_DIR,
    pocket_cutoff: float = 10.0,
    contact_cutoff: float = 4.5,
    max_atoms: int = 384,
) -> PocketFeatures:
    """Load and featurize a curated Tourette CSTC / GABAergic pocket into ``PocketFeatures``."""
    spec = get_tourette_circuit_target(target_id)
    found = find_local_ts_circuit_pocket_files(
        spec.target_id, pdb_id=pdb_id, structures_dir=structures_dir
    )
    if found is None:
        raise FileNotFoundError(
            f"No curated 3D pocket found for Tourette CSTC target {spec.target_id} under "
            f"{Path(structures_dir)}."
        )
    pocket_path, lig_sdf = found
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
            keep_cofactors=spec.keep_cofactors,
        )
    return featurize_pdb_pocket_inference(
        pocket_path,
        pocket_cutoff=pocket_cutoff,
        max_atoms=max_atoms,
        pdb_id=spec.primary_pdb_id.lower(),
        keep_cofactors=spec.keep_cofactors,
    )


# ---------------------------------------------------------------------------
# 4. Mechanism-Aware Tourette Circuit Scoring Helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TouretteCircuitScore:
    """Mechanism-aware multi-objective score for a candidate on a Tourette CSTC target."""

    smiles: str
    target_id: str
    cstc_node: CSTCNode
    moa_class: str
    valid: bool
    hygiene_gate: float
    cns_mpo_v2: float
    mechanism_polarity_fit: float
    vina_score_norm: float
    admet_safety_gate: float
    circuit_readiness_score: float
    anchor_engaged: bool | None
    posebusters_pass: bool | None
    vina_dg_kcal: float | None
    mechanism_flags: tuple[str, ...]
    triage: MedChemTriageResult | None = None
    mpo: CNSMPOResult | None = None
    admet_v4: PharmaAdmetPrediction | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize score breakdown to a JSON-compatible dictionary."""
        return {
            "smiles": self.smiles,
            "target_id": self.target_id,
            "cstc_node": self.cstc_node,
            "moa_class": self.moa_class,
            "valid": self.valid,
            "hygiene_gate": round(self.hygiene_gate, 4),
            "cns_mpo_v2": round(self.cns_mpo_v2, 4),
            "mechanism_polarity_fit": round(self.mechanism_polarity_fit, 4),
            "vina_score_norm": round(self.vina_score_norm, 4),
            "admet_safety_gate": round(self.admet_safety_gate, 4),
            "circuit_readiness_score": round(self.circuit_readiness_score, 4),
            "anchor_engaged": self.anchor_engaged,
            "posebusters_pass": self.posebusters_pass,
            "vina_dg_kcal": round(self.vina_dg_kcal, 3) if self.vina_dg_kcal is not None else None,
            "mechanism_flags": list(self.mechanism_flags),
        }


def _mechanism_polarity_calibration(
    target_id: str,
    mol: Chem.Mol,
    mpo: CNSMPOResult,
) -> tuple[float, list[str]]:
    """Compute target-mechanism-aware CNS polarity/pharmacophore fit in [0, 1] + notes."""
    flags: list[str] = []
    tid = target_id.upper()
    has_cooh = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1,O-]"))
    has_alkyne = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX2]#[CX2]"))
    has_ribose = mol.HasSubstructMatch(
        Chem.MolFromSmarts("[#8;R]1[#6;R][#6;R]([OX2H1])[#6;R]([OX2H1])[#6;R]1")
    )

    if tid == "GABRA5":
        # Non-sedating BZD-site PAMs strongly favor neutral, moderate-MW (250-450), TPSA 40-90
        fit = min(1.0, mpo.cns_mpo_v2 / 5.5)
        if mpo.most_basic_pka > 8.5:
            flags.append("high_basicity_herg_risk_for_neutral_bzd_cleft")
            fit *= 0.85
        flags.append("requires_alpha1_vs_alpha5_electrophysiology_selectivity_check")
    elif tid == "GAT1":
        # GAT1 inhibitors often carry an acidic/zwitterionic headgroup (e.g., Tiagabine MPO 3.86)
        # or neutral allosteric vestibule scaffolds; avoid over-penalizing valid zwitterions
        base = min(1.0, (mpo.cns_mpo_v2 + (0.75 if has_cooh else 0.0)) / 5.2)
        fit = float(np.clip(base, 0.0, 1.0))
        if has_cooh:
            flags.append("zwitterionic_or_acidic_gaba_mimetic_headgroup")
    elif tid == "ADORA2A":
        # Active A2A agonists need CNS BBB access; non-ribose scaffolds beat ribosides (MPO 3.0)
        fit = min(1.0, mpo.cns_mpo_v2 / 5.0)
        if has_ribose:
            flags.append("polar_ribose_agonist_low_bbb_and_peripheral_hypotension_risk")
        else:
            flags.append("non_ribose_cns_scaffold_requires_functional_gs_agonism_assay")
    elif tid == "GRM5":
        # mGluR5 7TM NAM pocket is deep & hydrophobic -> high MPO neutral scaffolds excel
        fit = min(1.0, mpo.cns_mpo_v2 / 5.5)
        if has_alkyne:
            flags.append("diaryl_alkyne_mglur5_nam_requires_gsh_reactive_metabolite_screen")
    else:
        fit = min(1.0, mpo.cns_mpo_v2 / 5.5)

    return float(np.clip(fit, 0.0, 1.0)), flags


def score_tourette_circuit_candidate(
    smiles: str,
    target_id: str,
    *,
    vina_dg_kcal: float | None = None,
    anchor_engaged: bool | None = None,
    posebusters_pass: bool | None = None,
    plif_tanimoto: float | None = None,
    admet_suite: CNSAdmetSuite | None = None,
) -> TouretteCircuitScore:
    """Compute a multiplicative, hygiene-gated, mechanism-aware Tourette CSTC score.

    Guarantees that PAINS / coupling reagents (DCU, DIU, HOBt, DMAP) receive exact ``0.0``,
    integrates optional 12-endpoint v4.0 Avoid-ome/ADMET safety predictions, and applies
    mechanism-specific pharmacology flags for each Basal Ganglia / GABAergic node.
    """
    spec = get_tourette_circuit_target(target_id)
    audit = TOURETTE_CSTC_CIRCUIT_AUDIT[spec.target_id]
    mol = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) and smiles.strip() else None
    if mol is None:
        return TouretteCircuitScore(
            smiles=str(smiles),
            target_id=spec.target_id,
            cstc_node=audit.cstc_node,
            moa_class=spec.moa_class,
            valid=False,
            hygiene_gate=0.0,
            cns_mpo_v2=0.0,
            mechanism_polarity_fit=0.0,
            vina_score_norm=0.0,
            admet_safety_gate=0.0,
            circuit_readiness_score=0.0,
            anchor_engaged=anchor_engaged,
            posebusters_pass=posebusters_pass,
            vina_dg_kcal=vina_dg_kcal,
            mechanism_flags=("invalid_smiles",),
        )

    can_smi = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    tr = triage_molecule(can_smi)
    mpo = compute_cns_mpo(can_smi)
    mech_fit, flags = _mechanism_polarity_calibration(spec.target_id, mol, mpo)

    # 3D Vina binding energy normalized cleanly into [0, 1] (-5.0 -> 0.0, -11.0 -> 1.0)
    if vina_dg_kcal is not None and math.isfinite(vina_dg_kcal):
        vina_norm = float(np.clip((-float(vina_dg_kcal) - 5.0) / 6.0, 0.0, 1.0))
    else:
        vina_norm = 0.50  # neutral placeholder when docking has not been executed yet

    # Optional 12-endpoint v4.0 ADMET + Avoid-ome safety gate
    admet_pred: PharmaAdmetPrediction | None = None
    if admet_suite is not None and len(admet_suite.fitted_endpoints) > 0:
        admet_pred = admet_suite.predict_v4([can_smi])[0]
        p = admet_pred.probabilities
        bbb = float(p.get("bbb", 0.5))
        herg_safe = 1.0 - float(p.get("herg", 0.3))
        pgp_safe = 1.0 - float(p.get("pgp", p.get("mdr1_efflux", 0.3)))
        htr2b_safe = 1.0 - float(p.get("htr2b_avoid", 0.1))
        hrh1_safe = 1.0 - float(p.get("hrh1_avoid", 0.2))
        admet_gate = float(
            np.clip(
                0.35 * bbb
                + 0.25 * herg_safe
                + 0.15 * pgp_safe
                + 0.15 * htr2b_safe
                + 0.10 * hrh1_safe,
                0.0,
                1.0,
            )
        )
        if p.get("herg", 0.0) >= 0.5:
            flags.append("elevated_herg_risk")
        if p.get("htr2b_avoid", 0.0) >= 0.5:
            flags.append("elevated_5ht2b_valvulopathy_risk")
        if p.get("hrh1_avoid", 0.0) >= 0.5:
            flags.append("elevated_h1_sedation_risk")
    else:
        admet_gate = float(np.clip(mpo.cns_mpo_v2 / 6.0, 0.0, 1.0))

    # Multiplicative structure-first readiness score: hygiene_gate=0 vetoes strictly to 0.0
    anchor_bonus = 1.0
    if anchor_engaged is True:
        anchor_bonus = 1.08
    elif anchor_engaged is False:
        anchor_bonus = 0.88
    if posebusters_pass is False:
        anchor_bonus *= 0.70
        flags.append("posebusters_geometry_warning")
    if plif_tanimoto is not None and math.isfinite(plif_tanimoto):
        anchor_bonus *= 0.90 + 0.20 * float(np.clip(plif_tanimoto, 0.0, 1.0))

    core = (0.40 * vina_norm + 0.35 * mech_fit + 0.25 * admet_gate) * anchor_bonus
    readiness = float(np.clip(tr.hygiene_gate * core, 0.0, 1.0))

    return TouretteCircuitScore(
        smiles=can_smi,
        target_id=spec.target_id,
        cstc_node=audit.cstc_node,
        moa_class=spec.moa_class,
        valid=True,
        hygiene_gate=float(tr.hygiene_gate),
        cns_mpo_v2=float(mpo.cns_mpo_v2),
        mechanism_polarity_fit=mech_fit,
        vina_score_norm=vina_norm,
        admet_safety_gate=admet_gate,
        circuit_readiness_score=readiness,
        anchor_engaged=anchor_engaged,
        posebusters_pass=posebusters_pass,
        vina_dg_kcal=vina_dg_kcal,
        mechanism_flags=tuple(flags),
        triage=tr,
        mpo=mpo,
        admet_v4=admet_pred,
    )


__all__ = [
    "DEFAULT_TS_CIRCUIT_DIR",
    "DEFAULT_TS_CIRCUIT_STRUCTURES_DIR",
    "TOURETTE_CIRCUIT_EXPANSION_REGISTRY",
    "TOURETTE_CSTC_CIRCUIT_AUDIT",
    "CSTCNode",
    "ClinicalEvidenceGrade",
    "MultiChainPocketExtractionResult",
    "TouretteCircuitAuditEntry",
    "TouretteCircuitScore",
    "extract_multichain_interface_pocket",
    "find_local_ts_circuit_pocket_files",
    "get_cstc_circuit_audit",
    "get_tourette_circuit_target",
    "list_tourette_circuit_targets",
    "load_tourette_circuit_pocket",
    "score_tourette_circuit_candidate",
]
