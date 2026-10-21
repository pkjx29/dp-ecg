"""Download PTB-XL v1.0.3 (100 Hz records + metadata only) and verify checksums.

Source: PhysioNet's official open-data mirror on AWS (physionet-open), which
serves the same files as physionet.org but much faster. Every file is checked
against PhysioNet's published SHA256SUMS.txt; a mismatch aborts the run.

PTB-XL is licensed CC BY 4.0. Cite:
  Wagner et al., "PTB-XL, a large publicly available electrocardiography
  dataset", Scientific Data 7, 154 (2020). doi:10.1038/s41597-020-0495-6
  PhysioNet version 1.0.3, doi:10.13026/kfzx-aw45

Usage: python scripts/download_ptbxl.py [--out data/raw/ptb-xl]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/ptb-xl")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sums = {}
    for line in fetch(BASE + "SHA256SUMS.txt").decode().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name] = digest

    wanted = META_FILES + sorted(n for n in sums if n.startswith("records100/"))

    def get(name: str) -> str | None:
        dest = out / name
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sums[name]:
            return None  # already have a verified copy
        data = fetch(BASE + name)
        if hashlib.sha256(data).hexdigest() != sums[name]:
            return name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return None

    print(f"Fetching {len(wanted)} files into {out} ...", flush=True)
    bad = []
    with ThreadPoolExecutor(args.workers) as pool:
        for i, res in enumerate(pool.map(get, wanted), 1):
            if res:
                bad.append(res)
            if i % 5000 == 0:
                print(f"  {i}/{len(wanted)}", flush=True)
    if bad:
        sys.exit(f"Checksum mismatch for {len(bad)} files, e.g. {bad[:3]}")
    print("All files present and SHA-256 verified.")


if __name__ == "__main__":
    main()
