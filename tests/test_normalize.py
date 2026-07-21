import pytest

from microhard.normalize import (
    CONDITION_ALIASES,
    GRADE_ALIASES,
    AmbiguousAliasError,
    canonical_tokens,
    check_aliases,
    normalize_condition,
    normalize_grade,
    normalize_join_key,
    normalize_structured_condition,
)
from microhard.taxonomy import CONDITION_AXIS, GRADE_AXIS, Taxonomy


def test_every_alias_target_is_a_registered_node() -> None:
    """The contract that makes the tables safe to extend."""
    check_aliases(Taxonomy.load(None))


def test_alias_targets_are_on_the_right_axis() -> None:
    tax = Taxonomy.load(None)
    assert {tax.axis_of(v) for v in GRADE_ALIASES.values()} == {GRADE_AXIS}
    assert {tax.axis_of(v) for v in CONDITION_ALIASES.values()} == {CONDITION_AXIS}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.45C", ("0.45c",)),
        ("Quenched & Tempered", ("quenched", "tempered")),
        ("quenched and tempered", ("quenched", "tempered")),
        ("normalised low carbon steel", ("normalized", "low", "carbon")),
        ("  WATER-QUENCHED  ", ("water", "quenched")),
    ],
)
def test_canonical_tokens(raw: str, expected: tuple[str, ...]) -> None:
    assert canonical_tokens(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("normalised low carbon steel", "grade/ferrous/low_carbon"),
        ("Mild Steel", "grade/ferrous/low_carbon"),
        ("0.45C quenched & tempered", "grade/ferrous/medium_carbon"),
        ("0.80C normalized", "grade/ferrous/high_carbon"),
        ("ultrahigh carbon steel", "grade/ferrous/ultrahigh_carbon"),
        ("AISI 1045", "grade/ferrous/aisi_1045"),
        ("AC1 970C 90M WQ", "grade/ferrous/uhcs_ac1"),
        ("AC 800C 8H WQ", "grade/ferrous/uhcs_ac"),
        ("AA 6061", "grade/aluminum/aa_6061"),
        ("Midrex DRI", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_grade(raw: str | None, expected: str | None) -> None:
    assert normalize_grade(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("normalised", "condition/austenitize/air_cool"),
        ("Normalizing", "condition/austenitize/air_cool"),
        ("AR", "condition/austenitize/air_cool"),
        ("water quenched", "condition/austenitize/water_quench"),
        ("WQ", "condition/austenitize/water_quench"),
        ("furnace cooled", "condition/austenitize/furnace_cool"),
        ("Q", "condition/austenitize/unspecified_quench"),
        ("0.45C quenched & tempered", "condition/quench_temper"),
        ("annealed", "condition/anneal"),
        ("full anneal", "condition/anneal/full"),
        ("spheroidised", "condition/anneal/spheroidize"),
        ("as-cast", "condition/as_cast"),
        ("650-1H", None),  # UHCSDB code for a route the vocabulary does not cover
        ("WC", None),
        ("", None),
    ],
)
def test_normalize_condition(raw: str, expected: str | None) -> None:
    assert normalize_condition(raw) == expected


def test_longest_alias_wins() -> None:
    """A bare "quenched" means an unrecorded medium; a qualified one does not."""
    assert normalize_condition("quenched") == "condition/austenitize/unspecified_quench"
    assert normalize_condition("water quenched") == "condition/austenitize/water_quench"
    assert normalize_condition("quenched & tempered") == "condition/quench_temper"


def test_join_key_from_one_free_text_field() -> None:
    assert normalize_join_key("0.45C quenched & tempered") == (
        "grade/ferrous/medium_carbon",
        "condition/quench_temper",
    )


def test_join_key_from_separate_fields() -> None:
    assert normalize_join_key("AC1 970C 90M", "WQ") == (
        "grade/ferrous/uhcs_ac1",
        "condition/austenitize/water_quench",
    )


def test_join_key_rejects_conflicting_fields() -> None:
    with pytest.raises(AmbiguousAliasError, match="conflicting condition ids"):
        normalize_join_key("AISI 1080 annealed", "water quenched")


def test_join_key_accepts_repeated_agreement() -> None:
    assert normalize_join_key("AC1 970C 90M WQ", "WQ") == (
        "grade/ferrous/uhcs_ac1",
        "condition/austenitize/water_quench",
    )


def test_unrecognised_string_yields_no_key() -> None:
    """A missing join is the intended outcome, not an error."""
    assert normalize_join_key("ET Gyro") == (None, None)


@pytest.mark.parametrize(
    "hold,unit,expected",
    [
        (90, "M", "condition/austenitize/water_quench/t800c_90m"),
        (1.5, "H", "condition/austenitize/water_quench/t800c_90m"),
        (3, "H", "condition/austenitize/water_quench/t800c_3h"),
    ],
)
def test_normalize_structured_condition(hold, unit, expected) -> None:
    assert normalize_structured_condition("WQ", 800, "C", hold, unit) == expected


def test_structured_condition_never_falls_back_to_a_coarse_node() -> None:
    assert normalize_structured_condition("WQ", 850, "C", 1, "H") is None
    assert normalize_structured_condition("Q", 800, "C", 3, "H") is None
    assert normalize_structured_condition("WQ-2C", 800, "C", 3, "H") is None
    assert normalize_structured_condition("WQ", 800, "F", 3, "H") is None


def test_ambiguous_match_raises() -> None:
    """Two equally good matches are a vocabulary bug, not a coin flip."""
    from microhard.normalize import _match

    table = {("shiny",): "condition/as_cast", ("bright",): "condition/hot_rolled"}
    with pytest.raises(AmbiguousAliasError, match="several condition ids"):
        _match("shiny bright", table, "condition")


def test_colliding_alias_table_is_rejected() -> None:
    from microhard.normalize import _canonical_table

    colliding = {
        "quenched & tempered": "condition/quench_temper",
        "quenched and tempered": "condition/as_cast",
    }
    with pytest.raises(ValueError, match="collides"):
        _canonical_table(colliding, "condition")
