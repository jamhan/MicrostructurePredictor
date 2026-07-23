import sqlite3
from dataclasses import replace

import pytest

from microhard.adapters import available_adapters, enabled_adapters, get_adapter
from microhard.adapters.uhcs import SEG_CLASS_NODES, UHCSAdapter, load_hardness_labels
from microhard.records import MEASURED
from microhard.taxonomy import Taxonomy, UnknownNodeError
from tests.conftest import MICROGRAPHS_PER_SAMPLE, N_SAMPLES


@pytest.fixture()
def taxonomy() -> Taxonomy:
    return Taxonomy.load(None)


def test_registry_lists_adapters() -> None:
    assert "uhcs" in available_adapters()
    assert "micronet_al" in available_adapters()
    assert "literature_steel" in available_adapters()
    assert "experimental_steel" in available_adapters()
    assert "godec_in718" in available_adapters()


def test_unknown_adapter_name(cfg, taxonomy) -> None:
    with pytest.raises(KeyError, match="micronet_al"):
        get_adapter("nope", cfg, taxonomy)


def test_uhcs_records(synthetic_db, taxonomy) -> None:
    records = UHCSAdapter(synthetic_db, taxonomy).validated_records()
    assert len(records) == N_SAMPLES * MICROGRAPHS_PER_SAMPLE
    first = records[0]
    assert first.scale_um_per_px == pytest.approx(10.0 / 100)  # micron_bar / micron_bar_px
    assert first.modality == "SEM"
    assert first.group_id.startswith("uhcs-sample-")
    assert first.taxonomy_labels is not None
    assert all(label.startswith("ferrous/") for label in first.taxonomy_labels)
    assert first.mask_path is None  # no benchmark on disk in this fixture
    assert first.properties == {}  # hardness CSV is empty
    assert first.image_path.exists()


def test_uhcs_combined_label_maps_to_two_nodes(synthetic_db, taxonomy) -> None:
    records = UHCSAdapter(synthetic_db, taxonomy).records()
    multi = [r for r in records if r.taxonomy_labels and len(r.taxonomy_labels) == 2]
    assert multi, "expected some combined primary_microconstituent labels"


def test_uhcs_attaches_benchmark_masks(seg_benchmark, taxonomy) -> None:
    records = UHCSAdapter(seg_benchmark, taxonomy).validated_records()
    masked = [r for r in records if r.mask_path is not None]
    assert len(masked) == 4  # stems micrograph1..4 match the benchmark fixture
    assert all(r.mask_class_nodes == SEG_CLASS_NODES for r in masked)


def test_uhcs_attaches_hardness_properties(synthetic_db, taxonomy) -> None:
    synthetic_db.hardness_csv.write_text(
        "sample_label,hardness_hv,source_note\nS1,310,Hecht2017\nS2,not-a-number,bad\n"
    )
    records = UHCSAdapter(synthetic_db, taxonomy).records()
    s1 = [r for r in records if r.group_id == "uhcs-sample-1"]
    assert all(r.properties == {"hardness_hv": 310.0} for r in s1)
    s2 = [r for r in records if r.group_id == "uhcs-sample-2"]
    assert all(r.properties == {} for r in s2)  # non-numeric row dropped


def test_uhcs_emits_a_join_key(synthetic_db, taxonomy) -> None:
    """Structured treatment metadata becomes an exact condition node."""
    con = sqlite3.connect(synthetic_db.sqlite_path)
    con.execute(
        "UPDATE sample SET label = 'AC1 970C 3H WQ', cool_method = 'WQ', "
        "anneal_temperature = 970, anneal_time = 3, anneal_time_unit = 'H' "
        "WHERE sample_id = 1"
    )
    con.execute("UPDATE sample SET label = 'ET Gyro', cool_method = NULL WHERE sample_id = 2")
    con.commit()
    con.close()

    records = UHCSAdapter(synthetic_db, taxonomy).validated_records()
    s1 = next(r for r in records if r.group_id == "uhcs-sample-1")
    assert s1.join_key == (
        "grade/ferrous/uhcs_ac1",
        "condition/austenitize/water_quench/t970c_3h",
    )

    s2 = next(r for r in records if r.group_id == "uhcs-sample-2")
    assert s2.join_key is None  # unrecognised metadata joins to nothing


@pytest.mark.parametrize(
    "label,cooling,temp,hold,unit",
    [
        ("AC1 970C 3H Q", "Q", 970, 3, "H"),
        ("AC1 970C 3H WQ", "WQ", 800, 3, "H"),
        ("AC1 800C 900C 970C 3H WQ", "WQ", 970, 3, "H"),
        ("AC1 970C 90M WQ-2C", "WQ-2C", 970, 90, "M"),
    ],
)
def test_uhcs_rejects_coarse_or_inconsistent_conditions(
    synthetic_db, taxonomy, label, cooling, temp, hold, unit
) -> None:
    with sqlite3.connect(synthetic_db.sqlite_path) as con:
        con.execute(
            "UPDATE sample SET label = ?, cool_method = ?, anneal_temperature = ?, "
            "anneal_time = ?, anneal_time_unit = ? WHERE sample_id = 1",
            (label, cooling, temp, hold, unit),
        )
    record = next(
        record
        for record in UHCSAdapter(synthetic_db, taxonomy).validated_records()
        if record.group_id == "uhcs-sample-1"
    )
    assert record.alloy_grade == "grade/ferrous/uhcs_ac1"
    assert record.condition is None


def test_uhcs_tags_measured_hardness(synthetic_db, taxonomy) -> None:
    synthetic_db.hardness_csv.write_text("sample_label,hardness_hv,source_note\nS1,310,Hecht2017\n")
    records = UHCSAdapter(synthetic_db, taxonomy).validated_records()
    s1 = next(r for r in records if r.group_id == "uhcs-sample-1")
    assert s1.property_sources == {"hardness_hv": MEASURED}


def test_duplicate_sample_labels_share_a_split_group(synthetic_db, taxonomy) -> None:
    with sqlite3.connect(synthetic_db.sqlite_path) as con:
        con.execute("UPDATE sample SET label = 'AC1 970C 24H WQ' WHERE sample_id IN (1, 2)")
    records = UHCSAdapter(synthetic_db, taxonomy).validated_records()
    duplicate_groups = {
        record.group_id for record in records if record.group_id == "uhcs-samples-1-2"
    }
    assert duplicate_groups == {"uhcs-samples-1-2"}
    assert sum(record.group_id == "uhcs-samples-1-2" for record in records) == 6


def test_validated_records_rejects_a_grade_used_as_a_label(synthetic_db, taxonomy) -> None:
    adapter = UHCSAdapter(synthetic_db, taxonomy)
    bad = [replace(adapter.records()[0], taxonomy_labels=("grade/ferrous/uhcs_ac1",))]
    adapter.records = lambda: bad  # type: ignore[method-assign]
    with pytest.raises(UnknownNodeError, match="expected 'microconstituent'"):
        adapter.validated_records()


def test_hardness_bad_header_rejected(synthetic_db) -> None:
    synthetic_db.hardness_csv.write_text("sample,hv\nS1,300\n")
    with pytest.raises(ValueError, match="hardness_hv"):
        load_hardness_labels(synthetic_db.hardness_csv)


def test_missing_sqlite_message(cfg, taxonomy) -> None:
    with pytest.raises(FileNotFoundError, match="11256/940"):
        UHCSAdapter(cfg, taxonomy).records()


def test_folder_stub_records(aluminum_stub, taxonomy) -> None:
    adapter = get_adapter("micronet_al", aluminum_stub, taxonomy)
    records = adapter.validated_records()
    assert len(records) == 6
    assert adapter.family == "aluminum"
    assert records[0].taxonomy_labels == ("aluminum/matrix", "aluminum/precipitate")
    assert records[0].group_id == "micronet_al-g0"
    assert records[0].mask_path is None
    assert records[0].properties == {}
    assert records[0].scale_um_per_px is None


def test_enabled_adapters_follow_config(aluminum_stub, taxonomy) -> None:
    aluminum_stub.adapters = ["micronet_al"]
    adapters = enabled_adapters(aluminum_stub, taxonomy)
    assert [a.name for a in adapters] == ["micronet_al"]
