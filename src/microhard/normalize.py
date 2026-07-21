"""Messy source strings -> controlled-vocabulary node ids.

Source metadata is written by people. The same heat treatment arrives as
"normalised low carbon steel", "0.45C quenched & tempered", or the bare
UHCSDB cool_method code "WQ". Distant supervision joins on
(alloy_grade, condition) node ids, so each of those has to land on a
registered id or on nothing at all.

Nothing here infers metallurgy. The alias tables are curated by hand, one
entry per phrase a source actually uses, and a string with no matching alias
returns None. A missing join costs one record; a wrong join silently teaches
the model a false hardness for a real microstructure, so None is the right
answer whenever the string is not recognised.

The parser handles the messiness, not the chemistry: case, punctuation,
British spellings, filler words, and extra text around the phrase.

    >>> normalize_join_key("normalised low carbon steel")
    ('grade/ferrous/low_carbon', 'condition/austenitize/air_cool')
"""

from __future__ import annotations

import re

from .taxonomy import CONDITION_AXIS, GRADE_AXIS, Taxonomy

# Words that carry no vocabulary information. Dropping them means an alias
# written as "low carbon steel" also matches "low carbon", and "quenched &
# tempered" also matches "quenched and tempered".
_STOPWORDS = frozenset(
    {"a", "an", "the", "and", "then", "steel", "steels", "alloy", "condition", "sample"}
)

# Token-level rewrites: British spellings and inflections onto one form.
_SPELLING = {
    "normalised": "normalized",
    "normalising": "normalized",
    "normalise": "normalized",
    "normalize": "normalized",
    "normalizing": "normalized",
    "spheroidised": "spheroidized",
    "spheroidise": "spheroidized",
    "spheroidize": "spheroidized",
    "spheroidizing": "spheroidized",
    "anneal": "annealed",
    "annealing": "annealed",
    "quench": "quenched",
    "quenching": "quenched",
    "temper": "tempered",
    "tempering": "tempered",
    "cool": "cooled",
    "cooling": "cooled",
    "roll": "rolled",
    "rolling": "rolled",
    "draw": "drawn",
    "drawing": "drawn",
}

# Anything that is not alphanumeric or a decimal point is a separator, so the
# carbon-content forms ("0.45C") survive tokenisation intact.
_SEPARATORS = re.compile(r"[^a-z0-9.]+")

# --- alias tables ----------------------------------------------------------
# Keys are written the way sources write them; both keys and inputs go through
# the same canonicalisation, so spelling variants do not need their own row.
# Every value must be a registered node id on the matching axis, which
# test_normalize.py checks against the bundled taxonomy.

GRADE_ALIASES: dict[str, str] = {
    # UHCSDB sample labels start with the casting id.
    "AC": "grade/ferrous/uhcs_ac",
    "AC1": "grade/ferrous/uhcs_ac1",
    # Standard designations, with and without the AISI prefix.
    "AISI 1018": "grade/ferrous/aisi_1018",
    "1018": "grade/ferrous/aisi_1018",
    "AISI 1045": "grade/ferrous/aisi_1045",
    "1045": "grade/ferrous/aisi_1045",
    # A carbon percentage is a composition family, not a unique grade. Several
    # alloy steels contain about 0.45% C, so promoting this to AISI 1045 would
    # create a more specific join key than the source supports.
    "0.45C": "grade/ferrous/medium_carbon",
    "AISI 1080": "grade/ferrous/aisi_1080",
    "1080": "grade/ferrous/aisi_1080",
    "0.80C": "grade/ferrous/high_carbon",
    "AISI 4140": "grade/ferrous/aisi_4140",
    "4140": "grade/ferrous/aisi_4140",
    # Coarse classes, for sources that report no grade at all.
    "low carbon steel": "grade/ferrous/low_carbon",
    "mild steel": "grade/ferrous/low_carbon",
    "medium carbon steel": "grade/ferrous/medium_carbon",
    "high carbon steel": "grade/ferrous/high_carbon",
    "ultrahigh carbon steel": "grade/ferrous/ultrahigh_carbon",
    "AA 6061": "grade/aluminum/aa_6061",
    "6061": "grade/aluminum/aa_6061",
    "AA 7075": "grade/aluminum/aa_7075",
    "7075": "grade/aluminum/aa_7075",
}

CONDITION_ALIASES: dict[str, str] = {
    "as cast": "condition/as_cast",
    "hot rolled": "condition/hot_rolled",
    "cold drawn": "condition/cold_drawn",
    # Generic annealing does not imply a full anneal. A source that says only
    # "annealed" gets the coarser registered node and cannot silently join a
    # full-anneal row.
    "annealed": "condition/anneal",
    "full anneal": "condition/anneal/full",
    "spheroidized": "condition/anneal/spheroidize",
    "spheroidized anneal": "condition/anneal/spheroidize",
    "subcritical anneal": "condition/anneal/subcritical",
    "process anneal": "condition/anneal/subcritical",
    # Normalizing is austenitizing plus a still-air cool, so it is the same
    # physical route as air cooling and shares its node.
    "normalized": "condition/austenitize/air_cool",
    "air cooled": "condition/austenitize/air_cool",
    "AR": "condition/austenitize/air_cool",  # UHCSDB cool_method code
    "WQ": "condition/austenitize/water_quench",
    "water quenched": "condition/austenitize/water_quench",
    "OQ": "condition/austenitize/oil_quench",
    "oil quenched": "condition/austenitize/oil_quench",
    "FC": "condition/austenitize/furnace_cool",
    "furnace cooled": "condition/austenitize/furnace_cool",
    "Q": "condition/austenitize/unspecified_quench",
    "quenched": "condition/austenitize/unspecified_quench",
    "quenched & tempered": "condition/quench_temper",
    "hardened & tempered": "condition/quench_temper",
}


class AmbiguousAliasError(ValueError):
    """A raw string matches two different node ids equally well."""


def canonical_tokens(raw: str) -> tuple[str, ...]:
    """Lowercase, split on punctuation, unify spellings, drop filler words."""
    text = _SEPARATORS.sub(" ", str(raw).lower())
    tokens = (t.strip(".") for t in text.split())
    tokens = (_SPELLING.get(t, t) for t in tokens if t)
    return tuple(t for t in tokens if t and t not in _STOPWORDS)


def _canonical_table(aliases: dict[str, str], kind: str) -> dict[tuple[str, ...], str]:
    out: dict[tuple[str, ...], str] = {}
    for phrase, node_id in aliases.items():
        key = canonical_tokens(phrase)
        if not key:
            raise ValueError(f"{kind} alias {phrase!r} canonicalises to nothing")
        if out.get(key, node_id) != node_id:
            raise ValueError(
                f"{kind} alias {phrase!r} collides with an existing entry: "
                f"{out[key]!r} vs {node_id!r}"
            )
        out[key] = node_id
    return out


_GRADE_TABLE = _canonical_table(GRADE_ALIASES, "grade")
_CONDITION_TABLE = _canonical_table(CONDITION_ALIASES, "condition")


def _contains(tokens: tuple[str, ...], key: tuple[str, ...]) -> bool:
    span = len(key)
    return any(tokens[i : i + span] == key for i in range(len(tokens) - span + 1))


def _match(raw: str, table: dict[tuple[str, ...], str], kind: str) -> str | None:
    """The id of the longest alias contained in ``raw``, or None.

    Longest wins so that "water quenched" beats "quenched". A tie between two
    different ids is a vocabulary bug, not something to resolve by guessing.
    """
    tokens = canonical_tokens(raw)
    best_len, best_ids = 0, set()
    for key, node_id in table.items():
        if len(key) < best_len or not _contains(tokens, key):
            continue
        if len(key) > best_len:
            best_len, best_ids = len(key), {node_id}
        else:
            best_ids.add(node_id)
    if not best_ids:
        return None
    if len(best_ids) > 1:
        raise AmbiguousAliasError(f"{raw!r} matches several {kind} ids: {sorted(best_ids)}")
    return best_ids.pop()


def normalize_grade(raw: str | None) -> str | None:
    """alloy_grade node id for a raw grade string, or None if unrecognised."""
    return None if raw is None else _match(raw, _GRADE_TABLE, "grade")


def normalize_condition(raw: str | None) -> str | None:
    """condition node id for a raw condition string, or None if unrecognised."""
    return None if raw is None else _match(raw, _CONDITION_TABLE, "condition")


def _resolve_matches(matches: set[str], raw: tuple[str | None, ...], kind: str) -> str | None:
    if not matches:
        return None
    if len(matches) > 1:
        supplied = [text for text in raw if text is not None]
        raise AmbiguousAliasError(
            f"{supplied!r} resolve to conflicting {kind} ids: {sorted(matches)}"
        )
    return matches.pop()


def normalize_join_key(*raw: str | None) -> tuple[str | None, str | None]:
    """(alloy_grade, condition) from one or more raw strings.

    Sources vary in how they split the information: one free-text field
    holding both ("0.45C quenched & tempered"), or separate grade and
    treatment columns. Pass whatever fields exist. Repeated matches may agree,
    but conflicting fields raise ``AmbiguousAliasError`` rather than choosing
    one according to argument order.
    """
    grade_matches = {match for text in raw if (match := normalize_grade(text)) is not None}
    condition_matches = {
        match for text in raw if (match := normalize_condition(text)) is not None
    }
    return (
        _resolve_matches(grade_matches, raw, "grade"),
        _resolve_matches(condition_matches, raw, "condition"),
    )


def check_aliases(taxonomy: Taxonomy) -> None:
    """Assert every alias target is a registered node on the right axis.

    Called by the tests, and worth calling from any script that extends the
    tables, since an alias pointing at a node id that was renamed produces a
    join key nothing else in the system recognises.
    """
    taxonomy.require(sorted(set(GRADE_ALIASES.values())), axis=GRADE_AXIS)
    taxonomy.require(sorted(set(CONDITION_ALIASES.values())), axis=CONDITION_AXIS)
