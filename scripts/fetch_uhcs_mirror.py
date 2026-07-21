"""Fetch UHCS data from live mirrors (NIST's materialsdata host is down, 2026-07).

Sources — both legitimate redistributions of the CC-licensed NIST items:
  - metadata + micrographs: Materials Data Facility mirror of handle 11256/940
    https://data.materialsdatafacility.org/legacy/mdr_item_1496_v1/
  - segmentation labels: DeCost's own uhcs-segment repo (source of 11256/964)
    https://github.com/bdecost/uhcs-segment (data/uhcs-tif/labels, int64 TIFFs)

Also does the standard UHCS preprocessing:
  - benchmark label stems uhcsNNNN are renamed micrographN (sqlite primary key)
  - the 38 px SIS instrument banner is cropped from images AND labels
    (the -1 "unlabeled" band in the benchmark; DeCost's cropbar=38)
  - int64 label TIFFs (PIL-unreadable) are converted to uint8 PNGs, classes
    0=matrix 1=network 2=spheroidite 3=widmanstatten

Usage:
    uv run python scripts/fetch_uhcs_mirror.py          # benchmark + gallery subset
    uv run python scripts/fetch_uhcs_mirror.py --all    # all 961 micrographs (~300 MB)
"""

from __future__ import annotations

import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

MDF = "https://data.materialsdatafacility.org/legacy/mdr_item_1496_v1"
GH_API_TREE = "https://api.github.com/repos/bdecost/uhcs-segment/git/trees/master?recursive=1"
GH_RAW_LABELS = "https://raw.githubusercontent.com/bdecost/uhcs-segment/master/data/uhcs-tif/labels"
CROPBAR = 38  # px of SIS banner at the bottom of every 645x522 frame

DATA = Path("data")
MICROGRAPHS = DATA / "micrographs"
LABELS = DATA / "segmentation" / "uhcs" / "labels"
BENCH_IMAGES = DATA / "segmentation" / "uhcs" / "images"


def fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            dest.write_bytes(r.read())
        return True
    except Exception as exc:  # noqa: BLE001 — report and continue
        print(f"[FAIL] {url}: {exc}")
        dest.unlink(missing_ok=True)
        return False


def crop_banner_inplace(path: Path) -> None:
    import numpy as np
    from PIL import Image

    im = np.asarray(Image.open(path))
    if im.shape[0] == 484 + CROPBAR:
        Image.fromarray(im[:-CROPBAR]).save(path)


def convert_label(tif: Path) -> None:
    """int64 TIFF -> banner-cropped uint8 PNG (values 0..3)."""
    import cv2
    import numpy as np
    from PIL import Image

    arr = cv2.imread(str(tif), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"could not read {tif}")
    arr = arr[:-CROPBAR]
    values = set(np.unique(arr).tolist())
    if not values <= {0, 1, 2, 3}:
        raise RuntimeError(f"{tif}: unexpected label values {values}")
    Image.fromarray(arr.astype(np.uint8)).save(tif.with_suffix(".png"))
    tif.unlink()


def benchmark_stems() -> list[str]:
    with urllib.request.urlopen(GH_API_TREE, timeout=60) as r:
        tree = json.load(r)["tree"]
    return [
        t["path"].split("/")[-1][:-4]
        for t in tree
        if t["path"].startswith("data/uhcs-tif/labels/") and t["path"].endswith(".tif")
    ]


def main() -> int:
    everything = "--all" in sys.argv
    ok = True

    ok &= fetch(f"{MDF}/microstructures.sqlite", DATA / "microstructures.sqlite")
    if not (DATA / "microstructures.sqlite").exists():
        print("cannot continue without metadata")
        return 1

    with sqlite3.connect(DATA / "microstructures.sqlite") as con:
        paths = [p for (p,) in con.execute("SELECT path FROM micrograph ORDER BY micrograph_id")]

    # segmentation benchmark: labels from GitHub, renamed uhcsNNNN -> micrographN
    stems = benchmark_stems()
    print(f"benchmark labels: {len(stems)}")
    bench_names = []
    for stem in stems:
        n = int(stem.removeprefix("uhcs"))
        name = f"micrograph{n}.tif"
        bench_names.append(name)
        png = LABELS / f"micrograph{n}.png"
        if not png.exists():
            tif = LABELS / f"micrograph{n}.tif"
            if fetch(f"{GH_RAW_LABELS}/{stem}.tif", tif):
                convert_label(tif)
            else:
                ok = False

    # micrographs: benchmark set always; everything with --all
    wanted = paths if everything else sorted(set(bench_names) & set(paths)) or bench_names
    print(f"micrographs to fetch: {len(wanted)}")
    for i, name in enumerate(wanted):
        dest = MICROGRAPHS / name
        if fetch(f"{MDF}/micrographs/micrographs/{name}", dest):
            crop_banner_inplace(dest)
        else:
            ok = False
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(wanted)}")

    # benchmark image copies next to the labels (handy for inspection)
    BENCH_IMAGES.mkdir(parents=True, exist_ok=True)
    for name in bench_names:
        src = MICROGRAPHS / name
        if src.exists():
            (BENCH_IMAGES / name).write_bytes(src.read_bytes())

    csv = DATA / "hardness_labels.csv"
    if not csv.exists():
        csv.write_text("sample_label,hardness_hv,source_note\n")

    print("done" + ("" if ok else " — WITH FAILURES (rerun to retry; files are skipped once present)"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
