"""Download PTB-XL v1.0.3 (100 Hz records + metadata only) and verify checksums.

Source: PhysioNet's official open-data mirror on AWS (physionet-open), which
serves the same files as physionet.org but much faster. Every file is checked
against PhysioNet's published SHA256SUMS.txt; a mismatch aborts the run.

PTB-XL is licensed CC BY 4.0. Cite:
  Wagner et al., "PTB-XL, a large publicly available electrocardiography
  dataset", Scientific Data 7, 154 (2020). doi:10.1038/s41597-020-0495-6
  PhysioNet version 1.0.3, doi:10.13026/kfzx-aw45

Run via the CLI:  dp-ecg download
"""

from __future__ import annotations

import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..logging_utils import get_logger

BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
log = get_logger(__name__)
META_FILES = ["ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt"]


def fetch(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
    raise RuntimeError("unreachable")


def download(out: Path, base_url: str = BASE, workers: int = 32) -> int:
    """Fetch metadata + all records100 files; skip files already verified.
    Returns the number of files checked. Raises on any checksum mismatch."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sums = {}
    for line in fetch(base_url + "SHA256SUMS.txt").decode().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name] = digest
    wanted = META_FILES + sorted(n for n in sums if n.startswith("records100/"))

    def get(name: str) -> str | None:
        dest = out / name
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sums[name]:
            return None
        data = fetch(base_url + name)
        if hashlib.sha256(data).hexdigest() != sums[name]:
            return name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return None

    log.info("Checking/fetching %d files into %s", len(wanted), out)
    with ThreadPoolExecutor(workers) as pool:
        bad = [r for r in pool.map(get, wanted) if r]
    if bad:
        raise RuntimeError(f"checksum mismatch for {len(bad)} files, e.g. {bad[:3]}")
    log.info("All %d files present and SHA-256 verified", len(wanted))
    return len(wanted)
