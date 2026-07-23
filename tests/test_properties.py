from pathlib import Path

import pytest

from microhard.properties import (
    PROPERTY_LOOKUP_COLUMNS,
    join_properties,
    load_property_lookup,
    lookup_index,
    measured_properties,
)
from microhard.records import DISTANT, MEASURED, CanonicalRecord
from microhard.taxonomy import Taxonomy

HEADER = ",".join(PROPERTY_LOOKUP_COLUMNS)

REPO_ROOT = Path(__file__).resolve().parents[1]

GRADE = "grade/ferrous/aisi_1045"
CONDITION = "condition/anneal/full"
# citation, url, note — a stand-in, so the fixtures never look like real data
CITED = "Example Handbook (test fixture),https://example.invalid/doi,"


def write_lookup(path: Path, *rows: str) -> Path:
    path.write_text("\n".join([HEADER, *rows]) + "\n")
    return path


def record(record_id: str = "r1", **kwargs) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=record_id,
        image_path=Path(f"{record_id}.png"),
        scale_um_per_px=None,
        modality="SEM",
        **kwargs,
    )


def test_shipped_template_is_loadable_and_empty() -> None:
    """The table ships as a header so the schema is reviewable before any data.

    Also pins the header against PROPERTY_LOOKUP_COLUMNS, so a column renamed
    in code cannot leave the committed template behind.
    """
    path = REPO_ROOT / "data" / "property_lookup.csv"
    assert path.read_text().splitlines() == [HEADER]
    assert load_property_lookup(path, Taxonomy.load(None)) == []


def test_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_property_lookup(tmp_path / "nope.csv") == []


def test_loads_a_row(tmp_path: Path) -> None:
    path = write_lookup(
        tmp_path / "lookup.csv",
        f"{GRADE},{CONDITION},hardness_hv,170,HV,8,sd,4,high,{CITED}",
    )
    (entry,) = load_property_lookup(path, Taxonomy.load(None))
    assert entry.key == (GRADE, CONDITION)
    assert entry.value == 170.0
    assert entry.scatter == 8.0
    assert entry.n_measurements == 4
    assert entry.join_confidence == "high"


def test_blank_scatter_and_count_are_allowed(tmp_path: Path) -> None:
    path = write_lookup(
        tmp_path / "lookup.csv",
        f"{GRADE},{CONDITION},hardness_hv,170,HV,,unreported,,low,{CITED}",
    )
    (entry,) = load_property_lookup(path)
    assert entry.scatter is None
    assert entry.n_measurements is None


def test_missing_columns_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lookup.csv"
    path.write_text("alloy_grade,condition,value\nx,y,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_property_lookup(path)


def test_uncited_row_rejected(tmp_path: Path) -> None:
    """The one rule the table exists to enforce."""
    path = write_lookup(
        tmp_path / "lookup.csv",
        f"{GRADE},{CONDITION},hardness_hv,170,HV,,unreported,,high,,,",
    )
    with pytest.raises(ValueError, match="source_citation is empty"):
        load_property_lookup(path)


def test_wrong_unit_rejected(tmp_path: Path) -> None:
    path = write_lookup(
        tmp_path / "lookup.csv",
        f"{GRADE},{CONDITION},hardness_hv,170,HB,,unreported,,high,{CITED}",
    )
    with pytest.raises(ValueError, match="recorded in HV"):
        load_property_lookup(path)


def test_unknown_property_rejected(tmp_path: Path) -> None:
    path = write_lookup(
        tmp_path / "lookup.csv",
        f"{GRADE},{CONDITION},fatigue_life,170,HV,,unreported,,high,{CITED}",
    )
    with pytest.raises(ValueError, match="unknown property_name"):
        load_property_lookup(path)


def test_bad_confidence_rejected(tmp_path: Path) -> None:
    path = write_lookup(
        tmp_path / "lookup.csv",
        f"{GRADE},{CONDITION},hardness_hv,170,HV,,unreported,,pretty sure,{CITED}",
    )
    with pytest.raises(ValueError, match="join_confidence"):
        load_property_lookup(path)


def test_unregistered_key_rejected(tmp_path: Path) -> None:
    path = write_lookup(
        tmp_path / "lookup.csv",
        f"grade/ferrous/unobtainium,{CONDITION},hardness_hv,170,HV,,unreported,,medium,{CITED}",
    )
    with pytest.raises(KeyError, match="unobtainium"):
        load_property_lookup(path)


def test_duplicate_key_rejected(tmp_path: Path) -> None:
    row = f"{GRADE},{CONDITION},hardness_hv,170,HV,,unreported,,medium,{CITED}"
    path = write_lookup(tmp_path / "lookup.csv", row, row.replace(",170,", ",210,"))
    with pytest.raises(ValueError, match="rows 2 and 3"):
        load_property_lookup(path)


@pytest.mark.parametrize(
    "value,match",
    [
        ("nan", "not finite"),
        ("inf", "not finite"),
        ("0", "must be positive"),
    ],
)
def test_invalid_hardness_rejected(tmp_path: Path, value: str, match: str) -> None:
    path = write_lookup(
        tmp_path / "lookup.csv",
        f"{GRADE},{CONDITION},hardness_hv,{value},HV,,unreported,,medium,{CITED}",
    )
    with pytest.raises(ValueError, match=match):
        load_property_lookup(path)


def test_scatter_metadata_must_be_consistent(tmp_path: Path) -> None:
    path = write_lookup(
        tmp_path / "lookup.csv",
        f"{GRADE},{CONDITION},hardness_hv,170,HV,-1,sd,4,high,{CITED}",
    )
    with pytest.raises(ValueError, match="non-negative"):
        load_property_lookup(path)

    path = write_lookup(
        tmp_path / "lookup.csv",
        f"{GRADE},{CONDITION},hardness_hv,170,HV,8,unreported,4,high,{CITED}",
    )
    with pytest.raises(ValueError, match="exactly when scatter is blank"):
        load_property_lookup(path)


def test_measurement_count_must_be_a_positive_integer(tmp_path: Path) -> None:
    path = write_lookup(
        tmp_path / "lookup.csv",
        f"{GRADE},{CONDITION},hardness_hv,170,HV,,unreported,2.5,medium,{CITED}",
    )
    with pytest.raises(ValueError, match="positive integer"):
        load_property_lookup(path)


def test_high_confidence_requires_uncertainty_evidence(tmp_path: Path) -> None:
    path = write_lookup(
        tmp_path / "lookup.csv",
        f"{GRADE},{CONDITION},hardness_hv,170,HV,,unreported,,high,{CITED}",
    )
    with pytest.raises(ValueError, match="high confidence requires"):
        load_property_lookup(path)


# --- the join --------------------------------------------------------------


@pytest.fixture()
def index(tmp_path: Path):
    path = write_lookup(
        tmp_path / "lookup.csv",
        f"{GRADE},{CONDITION},hardness_hv,170,HV,8,sd,4,high,{CITED}",
        f"grade/ferrous/aisi_1080,{CONDITION},hardness_hv,190,HV,,unreported,,low,{CITED}",
    )
    return lookup_index(load_property_lookup(path, Taxonomy.load(None)))


def test_join_attaches_a_distant_label(index) -> None:
    (joined,) = join_properties([record(alloy_grade=GRADE, condition=CONDITION)], index)
    assert joined.properties == {"hardness_hv": 170.0}
    assert joined.property_sources == {"hardness_hv": DISTANT}
    assert joined.property_weights == {"hardness_hv": 0.85}


def test_join_needs_both_halves_of_the_key(index) -> None:
    partial = record(alloy_grade=GRADE)  # no condition
    (joined,) = join_properties([partial], index)
    assert joined.join_key is None
    assert joined.properties == {}


def test_join_leaves_unknown_keys_alone(index) -> None:
    unknown = record(alloy_grade=GRADE, condition="condition/as_cast")
    (joined,) = join_properties([unknown], index)
    assert joined.properties == {}


def test_measurement_beats_a_distant_label(index) -> None:
    measured = record(
        alloy_grade=GRADE,
        condition=CONDITION,
        properties={"hardness_hv": 310.0},
        property_sources={"hardness_hv": MEASURED},
    )
    (joined,) = join_properties([measured], index)
    assert joined.properties == {"hardness_hv": 310.0}
    assert joined.property_sources == {"hardness_hv": MEASURED}
    assert joined.property_weight("hardness_hv") == 1.0


def test_min_confidence_filters_weak_rows(index) -> None:
    weak = record(alloy_grade="grade/ferrous/aisi_1080", condition=CONDITION)
    assert join_properties([weak], index)[0].properties == {"hardness_hv": 190.0}
    assert join_properties([weak], index, min_confidence="high")[0].properties == {}


def test_join_does_not_mutate_the_input(index) -> None:
    original = record(alloy_grade=GRADE, condition=CONDITION)
    join_properties([original], index)
    assert original.properties == {}


def test_measured_properties_drops_distant_labels(index) -> None:
    (joined,) = join_properties([record(alloy_grade=GRADE, condition=CONDITION)], index)
    assert measured_properties(joined) == {}
