import pytest

from microhard.adapters import available_adapters, enabled_adapters, get_adapter
from microhard.adapters.uhcs import SEG_CLASS_NODES, UHCSAdapter, load_hardness_labels
from microhard.taxonomy import Taxonomy
from tests.conftest import MICROGRAPHS_PER_SAMPLE, N_SAMPLES


@pytest.fixture()
def taxonomy() -> Taxonomy:
    return Taxonomy.load(None)


def test_registry_lists_both_adapters() -> None:
    assert "uhcs" in available_adapters()
    assert "micronet_al" in available_adapters()


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
