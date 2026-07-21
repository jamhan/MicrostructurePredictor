"""Print the UHCS distant-supervision join-key audit.

Usage:
    uv run python scripts/audit_join_keys.py
    uv run python scripts/audit_join_keys.py path/to/microstructures.sqlite
"""

from __future__ import annotations

import sys
from pathlib import Path

from microhard.join_audit import audit_uhcs_join_keys


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/microstructures.sqlite")
    audit = audit_uhcs_join_keys(path)
    summary = audit.summary
    print("summary")
    for name, value in summary.__dict__.items():
        print(f"{name},{value}")

    print("\ncoarse_key_groups")
    print("alloy_grade,condition,samples,micrographs,distinct_thermal_signatures")
    for group in audit.coarse_groups:
        print(
            f"{group.alloy_grade},{group.condition},{group.samples},"
            f"{group.micrographs},{len(group.thermal_signatures)}"
        )

    print("\nexact_condition_candidates")
    print("status,alloy_grade,thermal_signature,sample_ids,micrographs,flags")
    for group in audit.exact_groups:
        sample_ids = "|".join(str(sample_id) for sample_id in group.sample_ids)
        flags = "|".join(group.flags)
        print(
            f"{group.status},{group.alloy_grade},{group.thermal_signature},"
            f"{sample_ids},{group.micrographs},{flags}"
        )


if __name__ == "__main__":
    main()
