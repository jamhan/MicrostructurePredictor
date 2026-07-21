#!/usr/bin/env bash
# Download the UHCS dataset (metadata, micrographs, segmentation benchmark).
# Safe to re-run: files that already exist are skipped. On any failure the
# manual URL is printed and the script keeps going.
#
# NOTE (2026-07): the canonical NIST host (materialsdata.nist.gov) is DOWN —
# every dspace URL 404s and the handle server 500s. The canonical fetches
# below are kept in case it returns; whatever is still missing afterwards is
# fetched from live mirrors (Materials Data Facility + DeCost's uhcs-segment
# repo) by scripts/fetch_uhcs_mirror.py, which also does the standard UHCS
# preprocessing (banner crop, label conversion). Use
#   uv run python scripts/fetch_uhcs_mirror.py --all
# to pull all 961 micrographs instead of the benchmark/demo subset.
#
# MicroNet encoder weights are NOT downloaded here — they are fetched lazily
# by `microhard train-seg` (with automatic fallback to ImageNet weights).
set -u

DATA_DIR="${DATA_DIR:-data}"
BASE940="https://materialsdata.nist.gov/dspace/xmlui/bitstream/handle/11256/940"
BASE964="https://materialsdata.nist.gov/dspace/xmlui/bitstream/handle/11256/964"
HANDLE940="https://hdl.handle.net/11256/940"
HANDLE964="https://hdl.handle.net/11256/964"

mkdir -p "$DATA_DIR"
FAILED=0

fetch() { # fetch <url> <dest> <manual_url>
  local url="$1" dest="$2" manual="$3"
  if [ -s "$dest" ]; then
    echo "[skip] $dest already exists"
    return 0
  fi
  echo "[get ] $url"
  if curl -fL --retry 3 --connect-timeout 30 -o "$dest.part" "$url"; then
    mv "$dest.part" "$dest"
    return 0
  fi
  rm -f "$dest.part"
  echo "[FAIL] could not download $url"
  echo "       -> download manually from $manual and place the file at: $dest"
  FAILED=1
  return 1
}

# Unzip and flatten all image files into one folder (the sqlite 'path' column
# is matched by basename as a fallback, so flattening is safe).
unzip_flat() { # unzip_flat <zip> <dest_dir>
  local zip="$1" dest="$2" tmp="$2.unzip_tmp"
  [ -d "$dest" ] && { echo "[skip] $dest already extracted"; return 0; }
  rm -rf "$tmp" && mkdir -p "$tmp"
  if ! unzip -q -o "$zip" -d "$tmp"; then
    echo "[FAIL] could not unzip $zip"; rm -rf "$tmp"; FAILED=1; return 1
  fi
  mkdir -p "$dest"  # created only after a successful unzip, so re-runs retry
  find "$tmp" -type f \( -iname '*.tif' -o -iname '*.tiff' -o -iname '*.png' \
    -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) \
    -exec mv {} "$dest/" \;
  rm -rf "$tmp"
  echo "[ ok ] extracted $(ls "$dest" | wc -l | tr -d ' ') images -> $dest"
}

# --- 1. UHCSDB metadata + micrographs (handle 11256/940) ---------------------
fetch "$BASE940/microstructures.sqlite" "$DATA_DIR/microstructures.sqlite" "$HANDLE940"
if fetch "$BASE940/micrographs.zip" "$DATA_DIR/micrographs.zip" "$HANDLE940"; then
  unzip_flat "$DATA_DIR/micrographs.zip" "$DATA_DIR/micrographs"
fi

# --- 2. Segmentation benchmark (handle 11256/964: uhcs/ + particles/) --------
# Bitstream names on the DSpace page vary; try the obvious candidates and
# fall back to a manual download prompt.
SEG_DIR="$DATA_DIR/segmentation"
mkdir -p "$SEG_DIR"
for name in uhcs particles; do
  if [ -d "$SEG_DIR/$name" ]; then
    echo "[skip] $SEG_DIR/$name already extracted"
    continue
  fi
  got=""
  for candidate in "$name.zip" "$name.tar.gz"; do
    if curl -fL --retry 2 --connect-timeout 30 -o "$SEG_DIR/$candidate.part" "$BASE964/$candidate"; then
      mv "$SEG_DIR/$candidate.part" "$SEG_DIR/$candidate"
      got="$SEG_DIR/$candidate"
      break
    fi
    rm -f "$SEG_DIR/$candidate.part"
  done
  if [ -n "$got" ]; then
    case "$got" in
      *.zip)    unzip -q -o "$got" -d "$SEG_DIR" || FAILED=1 ;;
      *.tar.gz) tar -xzf "$got" -C "$SEG_DIR" || FAILED=1 ;;
    esac
  else
    echo "[FAIL] could not download the '$name' segmentation archive"
    echo "       -> download manually from $HANDLE964 and extract so that"
    echo "          images+masks live under $SEG_DIR/$name/"
    FAILED=1
  fi
done

# --- 3. Hardness label template (filled in by hand from the Hecht papers) ----
CSV="$DATA_DIR/hardness_labels.csv"
if [ ! -f "$CSV" ]; then
  printf 'sample_label,hardness_hv,source_note\n' > "$CSV"
  echo "[ ok ] created empty $CSV (transcribe HV values from the Hecht papers)"
fi

echo
if [ "$FAILED" -ne 0 ]; then
  echo "Canonical NIST downloads failed — falling back to live mirrors"
  echo "(Materials Data Facility + github.com/bdecost/uhcs-segment)..."
  if uv run python scripts/fetch_uhcs_mirror.py; then
    echo "Mirror fetch OK. Verify with: microhard download"
    exit 0
  fi
  echo "Mirror fetch also failed. Manual URLs:"
  echo "  metadata/micrographs: $HANDLE940 (mirror: https://data.materialsdatafacility.org/legacy/mdr_item_1496_v1/)"
  echo "  segmentation masks:   $HANDLE964 (mirror: https://github.com/bdecost/uhcs-segment)"
  exit 1
fi
echo "Done. Verify with: microhard download"
