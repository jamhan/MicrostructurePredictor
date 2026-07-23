from microhard.linkage import (
    MaterialState,
    PropertyObservation,
    best_auto_links,
    match_state_to_observation,
)


def image(**kwargs) -> MaterialState:
    values = {
        "record_id": "image-1",
        "source_id": "study-1",
        "alloy": "Inconel 718",
        "process": "laser powder bed fusion",
        "state": "heat treated",
        "temperature_c": 954.0,
        "hold_minutes": 60.0,
        "build_strategy": "Gaussian",
    }
    values.update(kwargs)
    return MaterialState(**values)


def observation(**kwargs) -> PropertyObservation:
    values = {
        "observation_id": "hardness-1",
        "source_id": "study-1",
        "property_name": "hardness_hv",
        "value": 487.1,
        "unit": "HV1",
        "alloy": "IN-718",
        "process": "LPBF",
        "state": "heat-treated",
        "temperature_c": 954.0,
        "hold_minutes": 60.0,
        "build_strategy": "Gaussian",
    }
    values.update(kwargs)
    return PropertyObservation(**values)


def test_same_study_condition_is_weighted_training_not_validation() -> None:
    result = match_state_to_observation(image(), observation())
    assert result.confidence == "high"
    assert result.score >= 0.85
    assert result.training_weight == 0.85
    assert result.auto_attach
    assert not result.validation_eligible


def test_explicit_temperature_conflict_overrides_string_similarity() -> None:
    result = match_state_to_observation(
        image(temperature_c=1154.0), observation(temperature_c=954.0)
    )
    assert result.confidence == "reject"
    assert result.score == 0
    assert "temperature conflict" in result.blockers[0]


def test_orientation_missing_keeps_tensile_values_as_candidates() -> None:
    result = match_state_to_observation(
        image(orientation=None),
        observation(
            observation_id="yield-h",
            property_name="yield_strength_mpa",
            value=1269,
            unit="MPa",
            orientation="H",
        ),
    )
    assert result.confidence == "medium"
    assert not result.auto_attach
    assert not result.validation_eligible
    assert any("orientation is absent" in reason for reason in result.reasons)


def test_exact_physical_specimen_can_be_validation_data() -> None:
    result = match_state_to_observation(
        image(physical_specimen_id="coupon-7"),
        observation(physical_specimen_id="coupon-7"),
    )
    assert result.confidence == "exact"
    assert result.auto_attach
    assert result.validation_eligible
    assert result.training_weight == 1.0


def test_tied_auto_links_are_not_selected() -> None:
    first = match_state_to_observation(image(), observation(observation_id="h1"))
    second = match_state_to_observation(
        image(), observation(observation_id="h2", value=490.0)
    )
    assert best_auto_links([first, second]) == {}
