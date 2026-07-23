#!/usr/bin/env python3
"""Fetch the compact, model-relevant subset of Zenodo record 14163786.

The full record is about 154 MB. This script downloads only the 22 BSE-SEM
microstructure fields and the three workbooks containing hardness/mechanical
results (about 20 MB total), verifies every Zenodo MD5, and preserves the API
metadata used to select the files.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

RECORD_ID = "14163786"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
DESTINATION = Path("data/public_in718_godec_2024/raw")
WORKBOOKS = {
    "Vickers hardness HV1.xlsx",
    "Gauss-Ring heat treated mechanical tests.xlsx",
    "Results_heat treatedxlsx.xlsx",
}


def selected(key: str) -> bool:
    return key.startswith("BEI ") and key.lower().endswith(".tif") or key in WORKBOOKS


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "microhard/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def main() -> None:
    metadata = fetch_json(API_URL)
    files = [item for item in metadata["files"] if selected(item["key"])]
    if len(files) != 25:
        raise RuntimeError(f"expected 25 selected files, Zenodo returned {len(files)}")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "zenodo_record.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    for index, item in enumerate(files, start=1):
        path = DESTINATION / item["key"]
        expected = item["checksum"].removeprefix("md5:")
        if path.is_file() and hashlib.md5(path.read_bytes()).hexdigest() == expected:
            print(f"[{index:02d}/{len(files)}] verified {path.name}")
            continue
        encoded_key = urllib.parse.quote(item["key"], safe="")
        url = f"{API_URL}/files/{encoded_key}/content"
        print(f"[{index:02d}/{len(files)}] downloading {path.name}")
        download(url, path)
        actual = hashlib.md5(path.read_bytes()).hexdigest()
        if actual != expected:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"MD5 mismatch for {item['key']}: expected {expected}, got {actual}"
            )

    print(f"Ready: {len(files)} files under {DESTINATION}")


if __name__ == "__main__":
    main()
