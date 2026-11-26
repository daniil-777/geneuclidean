"""3D protein binding-pocket abstraction for featurization, hotspot audits, and docking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geneuclidean.exceptions import PocketFeaturizationError, StructureNotFoundError
from geneuclidean.sdk.registries import get_target

if TYPE_CHECKING:
    from geneuclidean.cns.targets import CNSTargetSpec
    from geneuclidean.data.featurizer import PocketFeatures
    from geneuclidean.evaluation.docking_pipeline import BoxMode, OrthostericBox


@dataclass
class Pocket:
    """Ergonomic 3D protein binding-pocket handle for featurization, auditing, and design.

    Create pockets either from the curated 28-target human CNS atlas via
    :meth:`Pocket.from_target` or from any local PDB/mmCIF structure via
    :meth:`Pocket.from_pdb`.

    Automatically propagates integral catalytic cofactors (e.g., ``FAD`` for ``MAOB``/``MAOA``,
    ``ADP`` for ``NLRP3``, ``ZN``/``MG`` for ``PDE10A``/``PDE1B``) from ``target_spec``
    (or explicit ``keep_cofactors``) into 3D featurization so catalytically essential
    heteroatoms are never silently dropped.

    Examples:
        >>> import geneuclidean as ge
        >>> pocket = ge.Pocket.from_target("DRD1")
        >>> pocket.pdb_id
        '9LLG'
    """

    pdb_id: str
    target_id: str | None = None
    pocket_pdb: Path | None = None
    ligand_sdf: Path | None = None
    center: tuple[float, float, float] | None = None
    structures_dir: Path | None = None
    pocket_cutoff: float = 10.0
    contact_cutoff: float = 4.5
    max_atoms: int = 384
    keep_cofactors: tuple[str, ...] = ()
    target_spec: CNSTargetSpec | None = None
    _cached_features: PocketFeatures | None = field(default=None, repr=False, compare=False)

    @property
    def effective_cofactors(self) -> tuple[str, ...]:
        """Return deduplicated uppercase cofactor residue codes retained during featurization."""
        merged: list[str] = []
        seen: set[str] = set()
        sources: tuple[Sequence[str], ...] = (
            self.target_spec.keep_cofactors if self.target_spec is not None else (),
            self.keep_cofactors,
        )
        for src in sources:
            for item in src:
                code = str(item).strip().upper()
                if code and code not in seen:
                    seen.add(code)
                    merged.append(code)
        return tuple(merged)

    @property
    def has_local_structure(self) -> bool:
        """Return ``True`` when a readable local PDB/mmCIF file is attached."""
        return self.pocket_pdb is not None and self.pocket_pdb.is_file()

    def clear_cache(self) -> None:
        """Invalidate any cached :class:`~geneuclidean.data.featurizer.PocketFeatures`."""
        self._cached_features = None

    @classmethod
    def from_target(
        cls,
        target_id: str,
        *,
        pdb_id: str | None = None,
        structures_dir: str | Path | None = None,
        allow_fetch: bool = False,
        pocket_cutoff: float = 10.0,
        contact_cutoff: float = 4.5,
        max_atoms: int = 384,
        keep_cofactors: Sequence[str] | None = None,
    ) -> Pocket:
        """Instantiate a :class:`Pocket` for a registered human CNS or custom target.

        Automatically searches v3.1, v4.0 Tri-Disorder, and Tourette CSTC/GABAergic local
        structure directories when ``structures_dir`` is omitted, while respecting strict
        directory isolation whenever a caller supplies an explicit ``structures_dir``.

        Args:
            target_id: Registered target ID, gene symbol, ChEMBL ID, or UniProt accession.
            pdb_id: Optional specific RCSB PDB accession (defaults to primary template).
            structures_dir: Optional directory containing ``clean_pockets/`` or ``raw_cif/``.
            allow_fetch: If ``True`` and no local file exists, download structure from RCSB.
            pocket_cutoff: Inclusion shell radius in Angstroms.
            contact_cutoff: Ligand contact distance threshold in Angstroms.
            max_atoms: Maximum retained heavy pocket atoms.
            keep_cofactors: Optional extra cofactor residue codes (e.g., ``("FAD",)``) in
                addition to ``spec.keep_cofactors``.

        Returns:
            Configured :class:`Pocket` instance.
        """
        from geneuclidean.cns.targets import (
            DEFAULT_CNS_STRUCTURES_DIR,
            DEFAULT_TS_CIRCUIT_STRUCTURES_DIR,
            DEFAULT_V4_STRUCTURES_DIR,
            fetch_rcsb_structure,
            find_local_pocket_files,
            get_cns_target,
        )

        spec = get_cns_target(target_id)
        chosen_pdb = (pdb_id or spec.primary_pdb_id).strip().upper()

        search_dirs: list[Path] = []
        if structures_dir is not None:
            search_dirs.append(Path(structures_dir))
        else:
            # Auto-discover across v3.1, v4.0 expansion, CSTC/GABA circuit, and .nosync mirrors.
            for candidate_dir in (
                DEFAULT_CNS_STRUCTURES_DIR,
                DEFAULT_V4_STRUCTURES_DIR,
                DEFAULT_TS_CIRCUIT_STRUCTURES_DIR,
                Path("data/cns_neuro.nosync/structures"),
                Path("data/cns_neuro.nosync/v4_expansion/structures"),
                Path("data/cns_neuro.nosync/ts_circuit/structures"),
            ):
                if candidate_dir not in search_dirs:
                    search_dirs.append(candidate_dir)

        pocket_path: Path | None = None
        lig_sdf: Path | None = None
        resolved_dir = search_dirs[0]
        for sdir in search_dirs:
            found = find_local_pocket_files(spec.target_id, pdb_id=pdb_id, structures_dir=sdir)
            if found is not None:
                pocket_path, lig_sdf = found
                resolved_dir = sdir
                break

        if pocket_path is None and allow_fetch:
            pocket_path = fetch_rcsb_structure(chosen_pdb, resolved_dir / "fetched")

        extra_cofactors = tuple(c.strip().upper() for c in (keep_cofactors or ()) if c.strip())
        return cls(
            pdb_id=chosen_pdb,
            target_id=spec.target_id,
            pocket_pdb=pocket_path,
            ligand_sdf=lig_sdf,
            structures_dir=resolved_dir,
            pocket_cutoff=float(pocket_cutoff),
            contact_cutoff=float(contact_cutoff),
            max_atoms=int(max_atoms),
            keep_cofactors=extra_cofactors,
            target_spec=spec,
        )

    @classmethod
    def from_pdb(
        cls,
        pdb_path: str | Path,
        *,
        center: Sequence[float] | None = None,
        ligand_sdf: str | Path | None = None,
        pdb_id: str | None = None,
        target_id: str | None = None,
        pocket_cutoff: float = 10.0,
        contact_cutoff: float = 4.5,
        max_atoms: int = 384,
        keep_cofactors: Sequence[str] = (),
    ) -> Pocket:
        """Instantiate a :class:`Pocket` directly from a local PDB or mmCIF file.

        Args:
            pdb_path: Path to a receptor or pre-cut pocket ``.pdb`` / ``.cif`` file.
            center: Optional 3D binding-site centroid ``(x, y, z)`` in Angstroms.
            ligand_sdf: Optional co-crystal 3D ligand ``.sdf`` file for centering & PLIF.
            pdb_id: Optional 4-character PDB identifier (inferred from filename if omitted).
            target_id: Optional registered target ID to attach hotspot & cofactor metadata.
            pocket_cutoff: Inclusion shell radius in Angstroms.
            contact_cutoff: Ligand contact distance threshold in Angstroms.
            max_atoms: Maximum retained heavy pocket atoms.
            keep_cofactors: Optional HETATM cofactor residue codes to retain alongside
                standard amino acids and catalytic Zn/Mg ions.

        Returns:
            Configured :class:`Pocket` instance.

        Raises:
            StructureNotFoundError: If ``pdb_path`` or ``ligand_sdf`` does not exist.
            PocketFeaturizationError: If ``center`` does not contain exactly 3 coordinates.
        """
        path = Path(pdb_path)
        if not path.is_file():
            raise StructureNotFoundError(
                f"Pocket structure file not found: {path}",
                hint=(
                    "Pass an existing .pdb/.cif path or use "
                    "Pocket.from_target(..., allow_fetch=True)."
                ),
            )

        center_tuple: tuple[float, float, float] | None = None
        if center is not None:
            if len(center) != 3:
                raise PocketFeaturizationError(
                    f"center must have exactly 3 coordinates, got {len(center)}"
                )
            center_tuple = (float(center[0]), float(center[1]), float(center[2]))

        sdf_path = Path(ligand_sdf) if ligand_sdf is not None else None
        if sdf_path is not None and not sdf_path.is_file():
            raise StructureNotFoundError(f"Ligand SDF file not found: {sdf_path}")

        spec = get_target(target_id) if target_id is not None else None
        resolved_pdb_id = (pdb_id or path.stem.split("_")[0]).strip().upper()
        extra_cofactors = tuple(c.strip().upper() for c in keep_cofactors if c.strip())
        return cls(
            pdb_id=resolved_pdb_id,
            target_id=spec.target_id if spec is not None else target_id,
            pocket_pdb=path,
            ligand_sdf=sdf_path,
            center=center_tuple,
            pocket_cutoff=float(pocket_cutoff),
            contact_cutoff=float(contact_cutoff),
            max_atoms=int(max_atoms),
            keep_cofactors=extra_cofactors,
            target_spec=spec,
        )

    def featurize(self, *, allow_fetch: bool = False, force: bool = False) -> PocketFeatures:
        """Compute strictly E(3)-invariant 52-dim per-atom 3D pocket features.

        Preserves any target-specific or user-supplied ``keep_cofactors`` (such as
        ``FAD`` in ``MAOB``/``MAOA`` and ``ADP`` in ``NLRP3``) consistently across both
        local-file and registry-loaded pockets.

        Args:
            allow_fetch: If ``True`` and created via :meth:`from_target` without local
                files, fetches the RCSB structure before featurizing.
            force: If ``True``, recompute even if cached features exist.

        Returns:
            :class:`~geneuclidean.data.featurizer.PocketFeatures` with centered ``[N, 3]``
            coordinates and ``[N, 52]`` E(3)-invariant node descriptors.
        """
        if self._cached_features is not None and not force:
            return self._cached_features

        from geneuclidean.data.featurizer import (
            extract_pocket_from_files,
            featurize_pdb_pocket_inference,
        )

        cofactors = self.effective_cofactors

        if self.pocket_pdb is not None and self.pocket_pdb.is_file():
            if self.ligand_sdf is not None and self.ligand_sdf.is_file():
                fallback_smi = (
                    self.target_spec.reference_drugs[0].canonical_smiles
                    if (self.target_spec is not None and self.target_spec.reference_drugs)
                    else None
                )
                feat = extract_pocket_from_files(
                    self.pocket_pdb,
                    ligand_sdf=self.ligand_sdf,
                    fallback_smiles=fallback_smi,
                    pdb_id=self.pdb_id.lower(),
                    pocket_cutoff=self.pocket_cutoff,
                    contact_cutoff=self.contact_cutoff,
                    max_atoms=self.max_atoms,
                    keep_cofactors=cofactors,
                )
            else:
                feat = featurize_pdb_pocket_inference(
                    self.pocket_pdb,
                    center=self.center,
                    pocket_cutoff=self.pocket_cutoff,
                    max_atoms=self.max_atoms,
                    pdb_id=self.pdb_id.lower(),
                    keep_cofactors=cofactors,
                )
            self._cached_features = feat
            return feat

        if self.target_id is not None:
            from geneuclidean.cns.targets import DEFAULT_CNS_STRUCTURES_DIR, load_cns_target_pocket

            feat = load_cns_target_pocket(
                self.target_id,
                pdb_id=self.pdb_id,
                structures_dir=self.structures_dir or DEFAULT_CNS_STRUCTURES_DIR,
                allow_fetch=allow_fetch,
                pocket_cutoff=self.pocket_cutoff,
                contact_cutoff=self.contact_cutoff,
                max_atoms=self.max_atoms,
            )
            self._cached_features = feat
            return feat

        raise StructureNotFoundError(
            f"No 3D structure file available for pocket {self.pdb_id!r}.",
            hint="Pass a valid pdb_path to Pocket.from_pdb(...) or allow_fetch=True.",
        )

    def audit_hotspots(
        self,
        *,
        target_id: str | None = None,
        contact_cutoff: float | None = None,
    ) -> dict[str, Any]:
        """Audit canonical pharmacological hotspot residue presence and 4.5 A contacts.

        Args:
            target_id: Optional target ID override if pocket was loaded via :meth:`from_pdb`.
            contact_cutoff: Distance threshold in Angstroms for ligand-contact verification.

        Returns:
            Dictionary reporting ``present_hotspots``, ``contact_hotspots_4p5A``,
            ``anchor_in_contact``, ``hotspot_coverage``, and ``present_cofactors``.
        """
        from geneuclidean.cns.targets import audit_pocket_hotspots, get_cns_target

        spec = get_cns_target(target_id) if target_id is not None else self.target_spec
        if spec is None:
            raise PocketFeaturizationError(
                "Hotspot auditing requires a registered target spec.",
                hint="Create Pocket via Pocket.from_target(...) or pass target_id='DRD1'.",
            )
        if self.pocket_pdb is None or not self.pocket_pdb.is_file():
            raise StructureNotFoundError(
                f"No local pocket PDB found for {spec.target_id} ({self.pdb_id}).",
                hint="Supply structures_dir or Pocket.from_pdb(...).",
            )
        return audit_pocket_hotspots(
            self.pocket_pdb,
            spec,
            ligand_sdf=self.ligand_sdf,
            contact_cutoff=self.contact_cutoff if contact_cutoff is None else float(contact_cutoff),
        )

    def orthosteric_box(
        self,
        *,
        mode: BoxMode = "tight_per_axis",
        padding: float | None = None,
        min_edge: float | None = None,
        max_edge: float | None = None,
    ) -> OrthostericBox:
        """Compute an AutoDock Vina orthosteric box centered on this pocket's ligand or site."""
        from geneuclidean.evaluation.docking_pipeline import OrthostericBox, compute_orthosteric_box

        if self.ligand_sdf is not None and self.ligand_sdf.is_file():
            return compute_orthosteric_box(
                self.ligand_sdf,
                mode=mode,
                padding=padding,
                min_edge=min_edge,
                max_edge=max_edge,
            )

        if self.center is not None:
            eff_center = self.center
        elif self.pocket_pdb is not None and self.pocket_pdb.is_file():
            from geneuclidean.evaluation.geometry3d import extract_pocket_heavy_atoms

            coords, _ = extract_pocket_heavy_atoms(self.pocket_pdb)
            if len(coords) == 0:
                raise PocketFeaturizationError(f"No heavy protein atoms found in {self.pocket_pdb}")
            c_arr = coords.mean(axis=0)
            eff_center = (float(c_arr[0]), float(c_arr[1]), float(c_arr[2]))
        else:
            raise PocketFeaturizationError(
                "Cannot compute orthosteric box without ligand_sdf, center, or pocket_pdb."
            )

        edge = 20.0 if mode == "cubic_standard" else 18.0
        return OrthostericBox(
            center=eff_center,
            size=(edge, edge, edge),
            native_span_xyz=(0.0, 0.0, 0.0),
            mode=mode,
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of this pocket and its attached metadata."""
        return {
            "target_id": self.target_id,
            "pdb_id": self.pdb_id,
            "pocket_pdb": str(self.pocket_pdb) if self.pocket_pdb is not None else None,
            "ligand_sdf": str(self.ligand_sdf) if self.ligand_sdf is not None else None,
            "center": list(self.center) if self.center is not None else None,
            "cofactors": list(self.effective_cofactors),
            "primary_anchor": (
                self.target_spec.primary_anchor_residue if self.target_spec is not None else None
            ),
            "uniprot_id": self.target_spec.uniprot_id if self.target_spec is not None else None,
            "chembl_id": self.target_spec.chembl_id if self.target_spec is not None else None,
        }

    def dock(self, smiles: str, **kwargs: Any) -> Any:
        """Convenience wrapper delegating to :func:`~geneuclidean.sdk.docking.dock_and_profile`."""
        from geneuclidean.sdk.docking import dock_and_profile

        return dock_and_profile(self, smiles, **kwargs)

    def visualize(
        self,
        ligand_sdf: str | Path | Any,
        *,
        out_html: str | Path | None = None,
        out_png: str | Path | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Render this pocket + ligand via :func:`~geneuclidean.sdk.viz.visualize_complex`.

        Automatically pre-populates ``pdb_id``, ``native_ligand_sdf``, and ``disease``
        from the pocket's attached metadata when not explicitly overridden.
        """
        if self.pocket_pdb is None or not self.pocket_pdb.is_file():
            raise StructureNotFoundError(
                f"Pocket {self.pdb_id!r} has no local PDB file to visualize.",
                hint="Construct via Pocket.from_pdb(...) or provide structures_dir.",
            )
        from geneuclidean.sdk.viz import visualize_complex

        eff_title = title or f"{self.target_id or self.pdb_id} ({self.pdb_id}) 3D Complex"
        if "native_ligand_sdf" not in kwargs and self.ligand_sdf is not None:
            kwargs["native_ligand_sdf"] = self.ligand_sdf
        if "pdb_id" not in kwargs:
            kwargs["pdb_id"] = self.pdb_id
        if (
            "disease" not in kwargs
            and self.target_spec is not None
            and self.target_spec.indications
        ):
            kwargs["disease"] = self.target_spec.indications[0]
        return visualize_complex(
            self.pocket_pdb,
            ligand_sdf,
            out_html=out_html,
            out_png=out_png,
            title=eff_title,
            **kwargs,
        )


__all__ = ["Pocket"]
