import sqlite3

from microhard.join_audit import audit_uhcs_join_keys


def test_audit_separates_candidate_provisional_and_blocked(synthetic_db) -> None:
    with sqlite3.connect(synthetic_db.sqlite_path) as con:
        con.execute(
            "UPDATE sample SET label = 'AC1 800C 5M WQ', cool_method = 'WQ' "
            "WHERE sample_id = 1"
        )
        con.execute(
            "UPDATE sample SET label = 'AC1 800C 5M Q', cool_method = 'Q' "
            "WHERE sample_id = 2"
        )
        con.execute(
            "UPDATE sample SET label = 'AC1 800C 900C 90M WQ', cool_method = 'WQ' "
            "WHERE sample_id = 3"
        )
        con.execute(
            "UPDATE sample SET label = 'AC1 970C 90M FC', cool_method = 'FC', "
            "anneal_temperature = 970, anneal_time = 90 WHERE sample_id = 4"
        )

    audit = audit_uhcs_join_keys(synthetic_db.sqlite_path)
    by_id = {sample.sample_id: sample for sample in audit.samples}
    assert by_id[1].status == "candidate"
    assert by_id[2].status == "blocked"
    assert "unspecified_quench_medium" in by_id[2].flags
    assert by_id[3].status == "blocked"
    assert "multi_step_label" in by_id[3].flags
    assert by_id[4].status == "provisional"
    assert by_id[4].thermal_signature == "t970c_90m_fc"


def test_audit_counts_orphan_micrographs(synthetic_db) -> None:
    with sqlite3.connect(synthetic_db.sqlite_path) as con:
        con.execute(
            "INSERT INTO micrograph VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (999, "orphan.png", 10.0, "um", 100, 500.0, "SE", None, "pearlite"),
        )

    summary = audit_uhcs_join_keys(synthetic_db.sqlite_path).summary
    assert summary.micrographs == 16
    assert summary.linked_micrographs == 15
    assert summary.orphan_micrographs == 1
    assert summary.incomplete_key_micrographs == 16


def test_audit_groups_duplicate_condition_rows(synthetic_db) -> None:
    with sqlite3.connect(synthetic_db.sqlite_path) as con:
        for sample_id in (1, 2):
            con.execute(
                "UPDATE sample SET label = 'AC1 970C 24H WQ', cool_method = 'WQ', "
                "anneal_temperature = 970, anneal_time = 24, anneal_time_unit = 'H' "
                "WHERE sample_id = ?",
                (sample_id,),
            )

    audit = audit_uhcs_join_keys(synthetic_db.sqlite_path)
    (group,) = [
        group for group in audit.exact_groups if group.thermal_signature == "t970c_24h_wq"
    ]
    assert group.sample_ids == (1, 2)
    assert group.micrographs == 6
    assert group.flags == ("duplicate_sample_label",)
