from __future__ import annotations

import sqlite3

import pytest

from microhard.measurement_plan import PLAN_COLUMNS, plan_uhcs_measurements


def _configure_process_space(cfg) -> None:
    treatments = {
        1: ("AC1 800C 90M WQ", 800, 90, "M", "WQ"),
        2: ("AC1 970C 24H WQ", 970, 24, "H", "WQ"),
        3: ("AC1 700C 5M WQ", 700, 5, "M", "WQ"),
        4: ("AC1 970C 3H WQ", 970, 3, "H", "WQ"),
        5: ("AC1 900C 90M WQ", 900, 90, "M", "WQ"),
    }
    with sqlite3.connect(cfg.sqlite_path) as con:
        for sample_id, values in treatments.items():
            label, temperature, hold, unit, cooling = values
            con.execute(
                "UPDATE sample SET label = ?, anneal_temperature = ?, anneal_time = ?, "
                "anneal_time_unit = ?, cool_method = ? WHERE sample_id = ?",
                (label, temperature, hold, unit, cooling, sample_id),
            )
    cfg.hardness_csv.write_text(
        "sample_label,hardness_hv,source_note\n"
        "AC1 800C 90M WQ,876,direct\n"
        "AC1 900C 90M WQ,810,direct\n"
    )


def test_measurement_plan_excludes_measured_and_is_deterministic(synthetic_db) -> None:
    _configure_process_space(synthetic_db)
    first = plan_uhcs_measurements(synthetic_db, limit=3)
    second = plan_uhcs_measurements(synthetic_db, limit=3)
    assert list(first.columns) == PLAN_COLUMNS
    assert first.equals(second)
    assert list(first["rank"]) == [1, 2, 3]
    assert set(first["sample_ids"]) == {"2", "3", "4"}
    assert not set(first["sample_label"]) & {
        "AC1 800C 90M WQ",
        "AC1 900C 90M WQ",
    }
    assert first["exact_condition"].str.startswith("condition/").all()


def test_measurement_plan_can_include_unverified_routes(synthetic_db) -> None:
    _configure_process_space(synthetic_db)
    with sqlite3.connect(synthetic_db.sqlite_path) as con:
        con.execute(
            "UPDATE sample SET label = 'AC1 700C 5M Q', cool_method = 'Q' "
            "WHERE sample_id = 3"
        )
    default = plan_uhcs_measurements(synthetic_db, limit=None)
    expanded = plan_uhcs_measurements(
        synthetic_db, limit=None, include_unverified=True
    )
    assert "3" not in set(default["sample_ids"])
    assert "3" in set(expanded["sample_ids"])


def test_measurement_plan_rejects_nonpositive_limit(synthetic_db) -> None:
    with pytest.raises(ValueError, match="positive"):
        plan_uhcs_measurements(synthetic_db, limit=0)
