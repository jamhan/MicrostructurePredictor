#!/usr/bin/env python3
"""Download and verify the next two public-data priorities.

Downloads are resumable through ``.part`` files and every completed file is
checked against the size and MD5 published by Zenodo. The record API response
is retained beside the archives so file selection and provenance remain
auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RECORDS = {
    "16603134": Path("data/public_in718_carbides_2025/raw"),
    "18800251": Path("data/public_316l_composition_2026/raw"),
}
CHUNK_SIZE = 4 * 1024 * 1024
PROGRESS_INTERVAL = 256 * 1024 * 1024


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "microhard/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def human_bytes(value: int) -> str:
    amount = float(value)
    for suffix in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024.0 or suffix == "TB":
            return f"{amount:.1f} {suffix}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def download_resumable(url: str, destination: Path, expected_size: int) -> None:
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0

    request = urllib.request.Request(url, headers={"User-Agent": "microhard/0.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    with urllib.request.urlopen(request, timeout=180) as response:
        status = getattr(response, "status", response.getcode())
        append = offset > 0 and status == 206
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        transferred = offset
        next_progress = (
            ((transferred // PROGRESS_INTERVAL) + 1) * PROGRESS_INTERVAL
        )
        with partial.open(mode) as handle:
            while chunk := response.read(CHUNK_SIZE):
                handle.write(chunk)
                transferred += len(chunk)
                if transferred >= next_progress or transferred == expected_size:
                    print(
                        f"      {human_bytes(transferred)} / "
                        f"{human_bytes(expected_size)}",
                        flush=True,
                    )
                    next_progress += PROGRESS_INTERVAL

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise IOError(
            f"incomplete transfer for {destination.name}: "
            f"{actual_size} of {expected_size} bytes"
        )
    partial.replace(destination)


def fetch_record(record_id: str, destination: Path) -> None:
    api_url = f"https://zenodo.org/api/records/{record_id}"
    metadata = fetch_json(api_url)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "zenodo_record.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    files = metadata["files"]
    total = sum(item["size"] for item in files)
    free = shutil.disk_usage(destination).free
    missing_bytes = sum(
        item["size"]
        for item in files
        if not (destination / item["key"]).exists()
    )
    if free < missing_bytes + 1024**3:
        raise OSError(
            f"record {record_id} needs up to {human_bytes(missing_bytes)} plus "
            f"headroom, but only {human_bytes(free)} is free"
        )

    print(
        f"\n{record_id}: {metadata['metadata']['title']}\n"
        f"  {len(files)} files, {human_bytes(total)}",
        flush=True,
    )
    for index, item in enumerate(files, start=1):
        path = destination / item["key"]
        expected_md5 = item["checksum"].removeprefix("md5:")
        if (
            path.is_file()
            and path.stat().st_size == item["size"]
            and md5_file(path) == expected_md5
        ):
            print(f"  [{index:02d}/{len(files):02d}] verified {path.name}", flush=True)
            continue

        encoded_key = urllib.parse.quote(item["key"], safe="")
        url = f"{api_url}/files/{encoded_key}/content"
        print(
            f"  [{index:02d}/{len(files):02d}] downloading {path.name} "
            f"({human_bytes(item['size'])})",
            flush=True,
        )
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                download_resumable(url, path, item["size"])
                last_error = None
                break
            except (OSError, urllib.error.URLError) as error:
                last_error = error
                print(f"      attempt {attempt}/3 interrupted: {error}", flush=True)
        if last_error is not None:
            raise last_error

        actual_md5 = md5_file(path)
        if actual_md5 != expected_md5:
            path.rename(path.with_name(path.name + ".bad-checksum"))
            raise IOError(
                f"MD5 mismatch for {item['key']}: expected {expected_md5}, "
                f"got {actual_md5}"
            )
        print(f"      checksum verified", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        action="append",
        choices=sorted(RECORDS),
        help="Fetch one record; repeat for both. Defaults to both records.",
    )
    args = parser.parse_args()
    record_ids = args.record or list(RECORDS)
    for record_id in record_ids:
        fetch_record(record_id, RECORDS[record_id])
    print("\nAll requested Zenodo datasets are complete and verified.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; rerun to resume from .part files.", file=sys.stderr)
        raise SystemExit(130)
