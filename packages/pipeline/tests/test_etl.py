"""Tests for ETL champion-name canonicalization.

Regression coverage for the bug where ``str.title()`` corrupted irregular
champion names ("Jarvan IV" -> "Jarvan Iv", "LeBlanc" -> "Leblanc"), which
split a champion's game/synergy/matchup data under a mangled key while the
correctly-spelled roster entry got zero data (and both showed up as separate
selectable champions on the site).
"""

from __future__ import annotations

import pandas as pd

from compstrength_pipeline.champions import STATIC_CHAMPION_ROLES, get_full_champion_roster
from compstrength_pipeline.etl import _standardize_champion_name


def _std(names):
    return list(_standardize_champion_name(pd.Series(names)))


def test_irregular_names_canonicalize_to_riot_spelling():
    # These are exactly the names str.title() used to mangle. They must map to
    # the roster's canonical spelling regardless of the input casing.
    assert _std(["Jarvan IV"]) == ["Jarvan IV"]
    assert _std(["jarvan iv"]) == ["Jarvan IV"]
    assert _std(["JARVAN IV"]) == ["Jarvan IV"]
    assert _std(["LeBlanc"]) == ["LeBlanc"]
    assert _std(["leblanc"]) == ["LeBlanc"]
    assert _std(["LEBLANC"]) == ["LeBlanc"]


def test_apostrophe_and_spaced_names_canonicalize():
    assert _std(["kai'sa"]) == ["Kai'Sa"]
    assert _std(["KOG'MAW"]) == ["Kog'Maw"]
    assert _std(["dr. mundo"]) == ["Dr. Mundo"]
    assert _std(["nunu & willump"]) == ["Nunu & Willump"]


def test_whitespace_is_trimmed_before_canonicalizing():
    assert _std(["  LeBlanc  "]) == ["LeBlanc"]


def test_unknown_champion_falls_back_to_titlecase_with_apostrophe_fixup():
    # A name the roster doesn't know still gets consistent casing (old
    # behavior preserved), including the apostrophe special-case.
    assert _std(["totallynew champion"]) == ["Totallynew Champion"]
    assert _std(["zz'rot the fake"]) == ["Zz'Rot The Fake"]


def test_non_string_values_are_left_missing_not_canonicalized():
    # NaN/None champions (empty ban slots in real OE data) must not crash and
    # must remain missing (they're dropped downstream by build_raw_tables),
    # while a real name alongside them still canonicalizes.
    out = _standardize_champion_name(pd.Series([float("nan"), None, "LeBlanc"]))
    assert bool(pd.isna(out.iloc[0]))
    assert bool(pd.isna(out.iloc[1]))
    assert out.iloc[2] == "LeBlanc"


def test_roster_has_no_casefold_duplicate_entries():
    # The duplicate-variant bug ("Jarvan Iv" + "Jarvan IV", "Leblanc" +
    # "LeBlanc") must not creep back into the static roster, or the full-roster
    # union would emit two selectable entries for one champion again.
    names = list(STATIC_CHAMPION_ROLES)
    casefolded = [n.casefold() for n in names]
    assert len(set(casefolded)) == len(names), (
        "static roster has case-variant duplicates: "
        f"{sorted({c for c in casefolded if casefolded.count(c) > 1})}"
    )


def test_full_roster_has_no_casefold_duplicate_entries():
    names = list(get_full_champion_roster())
    casefolded = [n.casefold() for n in names]
    assert len(set(casefolded)) == len(names)
