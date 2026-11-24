"""Reproducible dataset downloaders and live RCSB PDB complex fetcher."""

from __future__ import annotations

import gzip
import platform
import stat
import struct
import tarfile
import tempfile
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import certifi
import numpy as np
import requests
from Bio.PDB import PDBIO, MMCIFParser, PDBParser, Select
from rdkit import Chem
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PBPP2020_URL = "https://huggingface.co/datasets/photonmz/pdbbindpp-2020/resolve/main/pbpp-2020.zip"
DEEPCHEM_V2019_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbindv2019/"
    "pdbbind_v2019_refined.tar.gz"
)
LP_PDBBIND_URL = "https://raw.githubusercontent.com/THGLab/LP-PDBBind/master/dataset/LP_PDBBind.csv"
CASF2016_ZENODO_URL = (
    "https://zenodo.org/api/records/11059434/files/pdbbind_core_set_2016.tar.gz/content"
)
ZINC250K_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/"
    "zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)
VINA_RELEASE_BASE = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7"
RCSB_FILES_BASE = "https://files.rcsb.org/download"
RCSB_LIGANDS_BASE = "https://files.rcsb.org/ligands/download"
RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"

_PBPP_KEEP_SUFFIXES: tuple[str, ...] = ("_pocket.pdb", "_ligand.sdf", "_ligand.mol2")
_SOLVENT_AND_BUFFER_RESNAMES: frozenset[str] = frozenset(
    {
        "HOH",
        "WAT",
        "DOD",
        "SO4",
        "PO4",
        "GOL",
        "EDO",
        "PEG",
        "PGE",
        "PG4",
        "DMS",
        "DMSO",
        "ACE",
        "MES",
        "EPE",
        "TRS",
        "ACT",
        "FMT",
        "CIT",
        "MPD",
        "BME",
        "CL",
        "NA",
        "K",
        "MG",
        "CA",
        "ZN",
        "FE",
        "MN",
        "CU",
        "NI",
        "CO",
        "BR",
        "IOD",
    }
)


def _make_session(pool_size: int = 16) -> requests.Session:
    """Create a retry-enabled HTTPS session verified with the certifi CA bundle."""
    session = requests.Session()
    session.verify = certifi.where()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("HEAD", "GET", "POST"),
    )
    adapter = HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=retry,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "GenEuclidean/2.0"})
    return session


def download_file(
    url: str,
    dest: str | Path,
    *,
    overwrite: bool = False,
    timeout: int = 60,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Stream-download a URL to disk atomically via a temporary file.

    Args:
        url: Remote HTTP(S) URL.
        dest: Target file path on disk.
        overwrite: Re-download even if ``dest`` already exists.
        timeout: Per-request network timeout in seconds.
        chunk_size: Streaming chunk size in bytes.

    Returns:
        Resolved path to the downloaded file.
    """
    dest_path = Path(dest)
    if dest_path.exists() and not overwrite:
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    session = _make_session(pool_size=4)
    try:
        with session.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)
        tmp_path.replace(dest_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    finally:
        session.close()
    return dest_path


def _parse_mol2_coords(mol2_path: Path) -> np.ndarray:
    """Extract heavy/all atom 3D coordinates directly from a TRIPOS MOL2 file."""
    coords: list[list[float]] = []
    in_atoms = False
    with mol2_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                continue
            if stripped.startswith("@<TRIPOS>"):
                if in_atoms:
                    break
                continue
            if in_atoms and stripped:
                parts = stripped.split()
                if len(parts) >= 5:
                    coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


class _ResidueKeySelect(Select):
    """Biopython selector retaining a specified set of (chain_id, residue_id) keys."""

    def __init__(self, keep_keys: set[tuple[str, Any]]) -> None:
        super().__init__()
        self._keep_keys = keep_keys

    def accept_residue(self, residue: Any) -> int:
        parent = residue.get_parent()
        chain_id = str(parent.id) if parent is not None else ""
        return 1 if (chain_id, residue.id) in self._keep_keys else 0


def _write_pocket_from_structure(
    structure: Any,
    lig_coords: np.ndarray,
    out_pdb: Path,
    pocket_cutoff: float = 10.0,
) -> int:
    """Save all non-water residues within ``pocket_cutoff`` Angstroms of ``lig_coords``."""
    model = structure[0]
    cutoff_sq = float(pocket_cutoff) ** 2
    keep_keys: set[tuple[str, Any]] = set()
    kept_atoms = 0
    for chain in model:
        for residue in chain:
            resname = str(residue.get_resname()).strip().upper()
            if resname in ("HOH", "WAT", "DOD"):
                continue
            # Keep standard polymer residues (res.id[0] == " ") and cofactors/ions not the ligand
            if residue.id[0] != " " and resname not in ("ZN", "MG", "CA", "FE", "MN", "CU", "HEM"):
                continue
            atoms = list(residue.get_atoms())
            if not atoms:
                continue
            rcoords = np.asarray([a.get_coord() for a in atoms], dtype=np.float32)
            diff = rcoords[:, None, :] - lig_coords[None, :, :]
            min_d2 = float(np.sum(diff * diff, axis=-1).min())
            if min_d2 <= cutoff_sq:
                keep_keys.add((str(chain.id), residue.id))
                kept_atoms += len(atoms)

    io_writer = PDBIO()
    io_writer.set_structure(structure)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    io_writer.save(str(out_pdb), _ResidueKeySelect(keep_keys))
    return kept_atoms


def download_pbpp2020(
    out_dir: str | Path = "data/raw/pbpp2020",
    *,
    url: str = PBPP2020_URL,
    use_range_streaming: bool = True,
    max_workers: int = 24,
    range_time_limit_s: float = 60.0,
    overwrite: bool = False,
) -> Path:
    """Download PDBbind v2020 refined pockets and ligands, skipping ``*_protein.pdb``.

    Uses ZIP central directory HTTP Range inspection to selectively extract only
    ``_pocket.pdb``, ``_ligand.sdf``, and ``_ligand.mol2`` (~254 MB uncompressed vs.
    2.83 GB full archive). Automatically falls back to a temporary full download
    followed by selective extraction and immediate cleanup if CDN range latency
    exceeds ``range_time_limit_s``.

    Args:
        out_dir: Destination directory for ``<pdb_id>/`` subdirectories.
        url: Remote ZIP archive URL.
        use_range_streaming: Attempt HTTP Range streaming first.
        max_workers: Concurrency for HTTP Range streaming.
        range_time_limit_s: Projected latency ceiling before switching to single stream.
        overwrite: Re-extract even if ``out_dir`` already contains >= 5,000 complexes.

    Returns:
        Path to ``out_dir``.
    """
    out_path = Path(out_dir)
    if (
        not overwrite
        and out_path.exists()
        and sum(1 for p in out_path.iterdir() if p.is_dir()) >= 5000
    ):
        return out_path

    out_path.mkdir(parents=True, exist_ok=True)
    session = _make_session(pool_size=max(32, max_workers + 4))
    try:
        head_resp = session.head(url, allow_redirects=True, timeout=20)
        head_resp.raise_for_status()
        final_url = str(head_resp.url)
        total_len = int(head_resp.headers["Content-Length"])

        did_range = False
        if use_range_streaming and head_resp.headers.get("Accept-Ranges", "").lower() != "none":
            tail_start = max(0, total_len - 65536)
            tail = session.get(
                final_url,
                headers={"Range": f"bytes={tail_start}-{total_len - 1}"},
                timeout=20,
            ).content
            eocd = tail.rfind(b"PK\x05\x06")
            if eocd != -1:
                _, _, _, _, cd_size, cd_offset, _ = struct.unpack_from("<HHHHIIH", tail, eocd + 4)
                cd = session.get(
                    final_url,
                    headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"},
                    timeout=30,
                ).content
                entries: list[tuple[int, str, int, int, int, int]] = []
                pos = 0
                while pos < len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
                    (
                        comp_method,
                        _,
                        _,
                        crc32,
                        comp_sz,
                        uncomp_sz,
                        fn_len,
                        ex_len,
                        cm_len,
                    ) = struct.unpack_from("<HHHIIIHHH", cd, pos + 10)
                    local_off = int(struct.unpack_from("<I", cd, pos + 42)[0])
                    fn = cd[pos + 46 : pos + 46 + fn_len].decode("utf-8", errors="ignore")
                    pos += 46 + fn_len + ex_len + cm_len
                    if fn.endswith(_PBPP_KEEP_SUFFIXES):
                        entries.append((local_off, fn, comp_method, comp_sz, uncomp_sz, crc32))
                entries.sort(key=lambda item: item[0])

                groups: list[list[Any]] = []
                for entry in entries:
                    est_end = entry[0] + 30 + len(entry[1].encode("utf-8")) + 256 + entry[3]
                    if groups and entry[0] - int(groups[-1][1]) <= 4096:
                        groups[-1][1] = max(int(groups[-1][1]), est_end)
                        groups[-1][2].append(entry)
                    else:
                        groups.append([entry[0], est_end, [entry]])

                def _fetch_group(grp: list[Any]) -> int:
                    start = int(grp[0])
                    end = min(int(grp[1]), total_len - 1)
                    items: list[tuple[int, str, int, int, int, int]] = grp[2]
                    resp = session.get(
                        final_url,
                        headers={"Range": f"bytes={start}-{end}"},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    buf = resp.content
                    count = 0
                    for local_off, fn, comp_method, comp_sz, uncomp_sz, crc32 in items:
                        rel = local_off - start
                        l_fn_len, l_ex_len = struct.unpack_from("<HH", buf, rel + 26)
                        data_start = rel + 30 + l_fn_len + l_ex_len
                        comp_data = buf[data_start : data_start + comp_sz]
                        if len(comp_data) < comp_sz:
                            exact_start = local_off + 30 + l_fn_len + l_ex_len
                            r2 = session.get(
                                final_url,
                                headers={
                                    "Range": f"bytes={exact_start}-{exact_start + comp_sz - 1}"
                                },
                                timeout=30,
                            )
                            r2.raise_for_status()
                            comp_data = r2.content
                        raw = (
                            comp_data
                            if comp_method == 0
                            else zlib.decompress(comp_data, -zlib.MAX_WBITS)
                        )
                        if len(raw) != uncomp_sz or (zlib.crc32(raw) & 0xFFFFFFFF) != crc32:
                            raise ValueError(f"ZIP integrity check failed for {fn}")
                        rel_name = fn.removeprefix("pbpp-2020/")
                        target = out_path / rel_name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(raw)
                        count += 1
                    return count

                probe_n = min(20, len(groups))
                if probe_n > 0:
                    t0 = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=max_workers) as pool:
                        futs = [pool.submit(_fetch_group, g) for g in groups[:probe_n]]
                        for fut in as_completed(futs):
                            fut.result()
                    dt = time.perf_counter() - t0
                    projected = dt * (len(groups) / float(probe_n))
                    if projected <= range_time_limit_s:
                        with ThreadPoolExecutor(max_workers=max_workers) as pool:
                            futs = [pool.submit(_fetch_group, g) for g in groups[probe_n:]]
                            for fut in as_completed(futs):
                                fut.result()
                        did_range = True

        if not did_range:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                tmp_zip = Path(tmp_file.name)
            try:
                with session.get(final_url, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    with tmp_zip.open("wb") as handle:
                        for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                with zipfile.ZipFile(tmp_zip, "r") as zf:
                    for info in zf.infolist():
                        if info.filename.endswith(_PBPP_KEEP_SUFFIXES):
                            rel_name = info.filename.removeprefix("pbpp-2020/")
                            target = out_path / rel_name
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(info) as src:
                                target.write_bytes(src.read())
            finally:
                if tmp_zip.exists():
                    tmp_zip.unlink()
    finally:
        session.close()
    return out_path


def download_lp_pdbbind(
    dest: str | Path = "data/raw/LP_PDBBind.csv",
    *,
    overwrite: bool = False,
) -> Path:
    """Download the LP-PDBBind leak-proof metadata and split table (10.9 MB)."""
    return download_file(LP_PDBBIND_URL, dest, overwrite=overwrite)


def download_casf2016_extra(
    out_dir: str | Path = "data/raw/casf2016_extra",
    *,
    existing_pdb_dir: str | Path | None = "data/raw/pbpp2020",
    pocket_cutoff: float = 10.0,
    overwrite: bool = False,
) -> Path:
    """Download CASF-2016 from Zenodo 11059434 and extract complexes missing from pbpp2020.

    Also generates residue-level ``<pdb_id>_pocket.pdb`` (``pocket_cutoff`` Angstroms) and
    RDKit ``<pdb_id>_ligand.sdf`` from ``<pdb_id>_ligand.mol2`` so all 285 CASF-2016
    complexes share the exact same file schema as ``pbpp2020``.
    """
    out_path = Path(out_dir)
    if (
        not overwrite
        and out_path.exists()
        and sum(1 for p in out_path.iterdir() if p.is_dir()) >= 19
    ):
        return out_path

    existing_ids: set[str] = set()
    if existing_pdb_dir is not None and Path(existing_pdb_dir).exists():
        existing_ids = {p.name.lower() for p in Path(existing_pdb_dir).iterdir() if p.is_dir()}

    out_path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tmp_tar = Path(tmp_file.name)

    try:
        download_file(CASF2016_ZENODO_URL, tmp_tar, overwrite=True, timeout=120)
        missing_ids: list[str] = []
        with tarfile.open(tmp_tar, "r:gz") as tf:
            members = tf.getmembers()
            all_ids: set[str] = set()
            for member in members:
                parts = [p for p in member.name.split("/") if p and p != "."]
                if len(parts) >= 2 and len(parts[1]) == 4:
                    all_ids.add(parts[1].lower())
            missing_ids = sorted(all_ids - existing_ids)
            missing_set = set(missing_ids)

            for member in members:
                if not member.isfile():
                    continue
                parts = [p for p in member.name.split("/") if p and p != "."]
                if len(parts) >= 3 and parts[1].lower() in missing_set:
                    pid = parts[1].lower()
                    fname = parts[-1]
                    if fname.endswith(
                        ("_ligand.mol2", "_ligand.sdf", "_protein.pdb", "_pocket.pdb")
                    ):
                        dst = out_path / pid / fname
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        extracted = tf.extractfile(member)
                        if extracted is not None:
                            dst.write_bytes(extracted.read())

        pdb_parser = PDBParser(QUIET=True)
        for pid in missing_ids:
            cdir = out_path / pid
            mol2_path = cdir / f"{pid}_ligand.mol2"
            prot_path = cdir / f"{pid}_protein.pdb"
            pocket_path = cdir / f"{pid}_pocket.pdb"
            sdf_path = cdir / f"{pid}_ligand.sdf"

            if mol2_path.exists() and prot_path.exists() and not pocket_path.exists():
                lig_coords = _parse_mol2_coords(mol2_path)
                if len(lig_coords) > 0:
                    struct = pdb_parser.get_structure(pid, str(prot_path))
                    _write_pocket_from_structure(struct, lig_coords, pocket_path, pocket_cutoff)

            if mol2_path.exists() and not sdf_path.exists():
                mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
                if mol is None:
                    mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)
                if mol is not None:
                    writer = Chem.SDWriter(str(sdf_path))
                    writer.write(mol)
                    writer.close()
    finally:
        if tmp_tar.exists():
            tmp_tar.unlink()
    return out_path


def download_zinc250k(
    dest: str | Path = "data/raw/zinc250k.csv",
    *,
    overwrite: bool = False,
) -> Path:
    """Download the ZINC-250k drug-like SMILES pretraining corpus (22.6 MB)."""
    return download_file(ZINC250K_URL, dest, overwrite=overwrite)


def download_vina_binary(
    dest: str | Path = "bin/vina",
    *,
    overwrite: bool = False,
) -> Path:
    """Download the official AutoDock Vina v1.2.7 precompiled binary for this host."""
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    if sys_name == "darwin":
        asset = "vina_1.2.7_mac_aarch64" if "arm" in machine else "vina_1.2.7_mac_x86_64"
    elif sys_name == "linux":
        asset = "vina_1.2.7_linux_aarch64" if "aarch64" in machine else "vina_1.2.7_linux_x86_64"
    else:
        raise RuntimeError(f"Unsupported platform for AutoDock Vina binary: {sys_name}/{machine}")

    url = f"{VINA_RELEASE_BASE}/{asset}"
    bin_path = download_file(url, dest, overwrite=overwrite)
    current_mode = bin_path.stat().st_mode
    bin_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_path


def download_all_datasets(
    raw_dir: str | Path = "data/raw",
    bin_dir: str | Path = "bin",
    *,
    include_vina: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download the complete lean GenEuclidean v2 dataset stack (~320 MB total)."""
    raw_root = Path(raw_dir)
    results: dict[str, Path] = {
        "pbpp2020": download_pbpp2020(raw_root / "pbpp2020", overwrite=overwrite),
        "lp_pdbbind": download_lp_pdbbind(raw_root / "LP_PDBBind.csv", overwrite=overwrite),
        "casf2016_extra": download_casf2016_extra(
            raw_root / "casf2016_extra",
            existing_pdb_dir=raw_root / "pbpp2020",
            overwrite=overwrite,
        ),
        "zinc250k": download_zinc250k(raw_root / "zinc250k.csv", overwrite=overwrite),
    }
    if include_vina:
        results["vina"] = download_vina_binary(Path(bin_dir) / "vina", overwrite=overwrite)
    return results


def fetch_rcsb_complex(
    pdb_id: str,
    out_dir: str | Path,
    *,
    ligand_resname: str | None = None,
    pocket_cutoff: float = 10.0,
) -> dict[str, Any]:
    """Fetch any PDB entry from RCSB, identify its bound ligand, and extract the 3D pocket.

    Designed for live CLI inference on arbitrary novel targets (``geneuclidean generate --pdb``).
    Uses ``requests`` + ``certifi`` so GraphQL TLS verification works reliably on macOS.

    Args:
        pdb_id: Four-character RCSB PDB identifier (case-insensitive).
        out_dir: Output directory where ``<pdb_id>.cif``, ``<pdb_id>_pocket.pdb``, and
            optional ligand files are written.
        ligand_resname: Optional 3-letter CCD ligand code to target. If ``None``, the
            largest non-solvent hetero residue is selected automatically.
        pocket_cutoff: Pocket extraction radius in Angstroms from any ligand atom.

    Returns:
        Dictionary containing ``pdb_id``, ``cif_path``, ``pocket_pdb``, ``ligand_sdf``,
        ``ligand_resname``, ``ligand_smiles``, ``ligand_center`` (list of 3 floats or
        ``None``), and ``pocket_atoms``.
    """
    pid_lower = pdb_id.strip().lower()
    pid_upper = pid_lower.upper()
    target_dir = Path(out_dir) / pid_lower
    target_dir.mkdir(parents=True, exist_ok=True)

    session = _make_session(pool_size=4)
    try:
        cif_gz_resp = session.get(f"{RCSB_FILES_BASE}/{pid_upper}.cif.gz", timeout=30)
        cif_gz_resp.raise_for_status()
        cif_path = target_dir / f"{pid_lower}.cif"
        cif_path.write_bytes(gzip.decompress(cif_gz_resp.content))

        # Query RCSB GraphQL for non-polymer ligand metadata and canonical SMILES
        gql_query = """
        query($id: String!) {
          entry(entry_id: $id) {
            rcsb_id
            nonpolymer_entities {
              nonpolymer_comp {
                chem_comp { id name formula_weight }
                rcsb_chem_comp_descriptor { SMILES_stereo }
              }
            }
          }
        }
        """
        smiles_by_ccd: dict[str, str] = {}
        try:
            gql_resp = session.post(
                RCSB_GRAPHQL_URL,
                json={"query": gql_query, "variables": {"id": pid_upper}},
                timeout=20,
            )
            gql_resp.raise_for_status()
            payload = gql_resp.json()
            entry_data = (payload.get("data") or {}).get("entry") or {}
            for entity in entry_data.get("nonpolymer_entities") or []:
                comp = (entity or {}).get("nonpolymer_comp") or {}
                ccd = str((comp.get("chem_comp") or {}).get("id", "")).strip().upper()
                desc = comp.get("rcsb_chem_comp_descriptor") or {}
                smi = desc.get("SMILES_stereo")
                if ccd and isinstance(smi, str) and smi:
                    smiles_by_ccd[ccd] = smi
        except Exception:
            smiles_by_ccd = {}

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pid_upper, str(cif_path))
        model = structure[0]

        requested = ligand_resname.strip().upper() if ligand_resname is not None else None
        candidates: list[Any] = []
        for chain in model:
            for residue in chain:
                if not str(residue.id[0]).startswith("H_"):
                    continue
                resname = str(residue.get_resname()).strip().upper()
                if requested is not None:
                    if resname == requested:
                        candidates.append(residue)
                elif resname not in _SOLVENT_AND_BUFFER_RESNAMES:
                    candidates.append(residue)

        selected_lig = (
            max(candidates, key=lambda r: len(list(r.get_atoms()))) if candidates else None
        )
        pocket_pdb = target_dir / f"{pid_lower}_pocket.pdb"
        ligand_sdf: Path | None = None
        detected_resname: str | None = None
        detected_smiles: str | None = None
        ligand_center: list[float] | None = None
        pocket_atoms = 0

        if selected_lig is not None:
            detected_resname = str(selected_lig.get_resname()).strip().upper()
            detected_smiles = smiles_by_ccd.get(detected_resname)
            lig_atoms = [a for a in selected_lig.get_atoms() if str(a.element).upper() != "H"]
            if not lig_atoms:
                lig_atoms = list(selected_lig.get_atoms())
            lig_coords = np.asarray([a.get_coord() for a in lig_atoms], dtype=np.float32)
            ligand_center = [float(x) for x in lig_coords.mean(axis=0)]
            pocket_atoms = _write_pocket_from_structure(
                structure, lig_coords, pocket_pdb, pocket_cutoff=pocket_cutoff
            )

            # Save crystal ligand coordinates as PDB and fetch ideal SDF template if available
            lig_pdb = target_dir / f"{pid_lower}_ligand.pdb"
            io_writer = PDBIO()
            io_writer.set_structure(structure)
            lig_key = (str(selected_lig.get_parent().id), selected_lig.id)
            io_writer.save(str(lig_pdb), _ResidueKeySelect({lig_key}))

            try:
                ideal_resp = session.get(
                    f"{RCSB_LIGANDS_BASE}/{detected_resname}_ideal.sdf",
                    timeout=15,
                )
                if ideal_resp.status_code == 200:
                    sdf_candidate = target_dir / f"{pid_lower}_ligand_ideal.sdf"
                    sdf_candidate.write_bytes(ideal_resp.content)
                    ligand_sdf = sdf_candidate
            except Exception:
                ligand_sdf = None
        else:
            # No non-polymer ligand found: save full polymer chains as PDB for manual center usage
            io_writer = PDBIO()
            io_writer.set_structure(structure)
            io_writer.save(str(pocket_pdb))
            pocket_atoms = sum(1 for _ in model.get_atoms())

        return {
            "pdb_id": pid_lower,
            "cif_path": cif_path,
            "pocket_pdb": pocket_pdb,
            "ligand_sdf": ligand_sdf,
            "ligand_resname": detected_resname,
            "ligand_smiles": detected_smiles,
            "ligand_center": ligand_center,
            "pocket_atoms": pocket_atoms,
        }
    finally:
        session.close()


__all__ = [
    "CASF2016_ZENODO_URL",
    "DEEPCHEM_V2019_URL",
    "LP_PDBBIND_URL",
    "PBPP2020_URL",
    "RCSB_FILES_BASE",
    "RCSB_GRAPHQL_URL",
    "ZINC250K_URL",
    "download_all_datasets",
    "download_casf2016_extra",
    "download_file",
    "download_lp_pdbbind",
    "download_pbpp2020",
    "download_vina_binary",
    "download_zinc250k",
    "fetch_rcsb_complex",
]
