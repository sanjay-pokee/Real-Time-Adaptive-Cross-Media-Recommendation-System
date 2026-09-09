import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.audience import (
    AudienceContext,
    explain_ineligibility,
    is_eligible,
    item_min_age,
    item_risk_tier,
)
from backend.domains import get_registry, load_registry


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_declares_the_shipped_content_types():
    registry = get_registry()

    assert set(registry.content_type_names()) == {
        "movie",
        "book",
        "music",
        "industrial",
        "health",
        "finance",
    }
    assert set(registry.domain_names()) == {
        "entertainment",
        "health",
        "industry",
        "finance",
    }


def test_regulated_domains_are_exactly_health_and_finance():
    """Both carry a legal advisory and gate below their content type default."""
    registry = get_registry()

    regulated = {
        name for name, spec in registry.domains.items() if spec.is_regulated
    }
    assert regulated == {"health", "finance"}
    for name in regulated:
        assert registry.domains[name].advisory, f"{name} must carry an advisory"


def test_aliases_resolve_to_canonical_names():
    registry = get_registry()

    assert registry.normalize_content_type("Films") == "movie"
    assert registry.normalize_content_type("tracks") == "music"
    assert registry.normalize_content_type("medical") == "health"
    assert registry.normalize_content_type("b2b") == "industrial"
    assert registry.normalize_content_type(None) is None


def test_unknown_content_type_lists_the_valid_options():
    with pytest.raises(ValueError, match="Unknown content type"):
        get_registry().normalize_content_type("podcast")


def test_domain_grouping_round_trips():
    registry = get_registry()

    assert registry.domain_of("movie").name == "entertainment"
    assert registry.domain_of("industrial").name == "industry"
    assert set(registry.content_types_for_domain("entertainment")) == {
        "movie",
        "book",
        "music",
    }
    # No domain means the whole catalogue is in scope.
    assert set(registry.content_types_for_domain(None)) == set(registry.content_type_names())


def test_regulated_domain_carries_an_advisory():
    registry = get_registry()

    assert registry.domain("health").is_regulated
    assert "not medical advice" in registry.domain("health").advisory.lower()
    assert not registry.domain("entertainment").is_regulated
    assert registry.domain("entertainment").advisory is None


def test_registry_rejects_a_content_type_in_an_undeclared_domain(tmp_path):
    broken = tmp_path / "domains.yaml"
    broken.write_text(
        "maturity_levels:\n"
        "  all_ages: 0\n"
        "domains:\n"
        "  entertainment:\n"
        "    content_types: [movie]\n"
        "content_types:\n"
        "  movie:\n"
        "    domain: entertainment\n"
        "    default_maturity: all_ages\n"
        "  ghost:\n"
        "    domain: nowhere\n"
        "    default_maturity: all_ages\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="undeclared domains"):
        load_registry(broken)


def test_registry_rejects_a_duplicate_alias(tmp_path):
    broken = tmp_path / "domains.yaml"
    broken.write_text(
        "maturity_levels:\n"
        "  all_ages: 0\n"
        "domains:\n"
        "  entertainment:\n"
        "    content_types: [movie, book]\n"
        "content_types:\n"
        "  movie:\n"
        "    domain: entertainment\n"
        "    default_maturity: all_ages\n"
        "    aliases: [flick]\n"
        "  book:\n"
        "    domain: entertainment\n"
        "    default_maturity: all_ages\n"
        "    aliases: [flick]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="claimed by both"):
        load_registry(broken)


# ---------------------------------------------------------------------------
# Item audience metadata
# ---------------------------------------------------------------------------

def test_explicit_min_age_wins_over_maturity_label():
    assert item_min_age({"content_type": "movie", "maturity": "all_ages", "audience_min_age": 18}) == 18


def test_maturity_label_wins_over_the_content_type_default():
    # movie defaults to teen (13); the row says all_ages.
    assert item_min_age({"content_type": "movie", "maturity": "all_ages"}) == 0


def test_bare_row_falls_back_to_its_content_type_default():
    # Absent metadata must not read as unrestricted.
    assert item_min_age({"content_type": "industrial"}) == 18
    assert item_min_age({"content_type": "movie"}) == 13


def test_risk_tier_falls_back_to_the_domain():
    assert item_risk_tier({"content_type": "health"}) == 1
    assert item_risk_tier({"content_type": "movie"}) == 0
    assert item_risk_tier({"content_type": "movie", "risk_tier": 2}) == 2


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

CHILD_MOVIE = {"content_type": "movie", "maturity": "all_ages"}
ADULT_MOVIE = {"content_type": "movie", "maturity": "adult"}
LAB_EQUIPMENT = {"content_type": "industrial"}


def test_a_child_sees_only_all_ages_content():
    context = AudienceContext(age=10)

    assert is_eligible(CHILD_MOVIE, context)
    assert not is_eligible(ADULT_MOVIE, context)
    assert not is_eligible(LAB_EQUIPMENT, context)


def test_an_adult_sees_everything_in_tier():
    context = AudienceContext(age=25)

    assert is_eligible(CHILD_MOVIE, context)
    assert is_eligible(ADULT_MOVIE, context)
    assert is_eligible(LAB_EQUIPMENT, context)


def test_a_missing_age_is_not_treated_as_adult():
    context = AudienceContext(age=None)

    assert context.effective_age == 13
    assert is_eligible(CHILD_MOVIE, context)
    assert not is_eligible(ADULT_MOVIE, context)
    assert "no age was supplied" in explain_ineligibility(ADULT_MOVIE, context)


def test_safe_mode_overrides_a_real_adult_age():
    context = AudienceContext(age=35, safe_mode=True)

    assert context.effective_age == 7
    assert is_eligible(CHILD_MOVIE, context)
    assert not is_eligible(ADULT_MOVIE, context)
    assert "safe-mode ceiling" in explain_ineligibility(ADULT_MOVIE, context)


def test_domain_scoping_excludes_other_verticals():
    context = AudienceContext(age=30, domain="entertainment")

    assert is_eligible(ADULT_MOVIE, context)
    assert not is_eligible(LAB_EQUIPMENT, context)
    assert "outside domain" in explain_ineligibility(LAB_EQUIPMENT, context)


def test_risk_tier_ceiling_withholds_professional_advice_items():
    item = {"content_type": "health", "maturity": "all_ages", "risk_tier": 2}
    permissive = AudienceContext(age=40, max_risk_tier=2)
    default = AudienceContext(age=40)

    assert is_eligible(item, permissive)
    assert not is_eligible(item, default)
    assert "risk tier 2" in explain_ineligibility(item, default)


def test_context_rejects_an_impossible_age():
    with pytest.raises(ValueError, match="between 0 and 120"):
        AudienceContext(age=999)


def test_context_rejects_an_unknown_domain():
    with pytest.raises(ValueError, match="Unknown domain"):
        AudienceContext(domain="astrology")


def test_advisory_is_surfaced_only_for_regulated_domains():
    assert AudienceContext(domain="health").advisory() is not None
    assert AudienceContext(domain="entertainment").advisory() is None
    assert AudienceContext().advisory() is None
