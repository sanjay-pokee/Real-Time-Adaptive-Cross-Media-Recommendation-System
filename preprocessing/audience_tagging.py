"""Derive audience metadata for every catalogue row.

The constraint layer in ``backend.audience`` needs four columns on each item:
``domain``, ``maturity``, ``audience_min_age`` and ``risk_tier``. None of the
source datasets ship them, so they are derived here, in one place, after the
per-dataset normalizers have run. That keeps the normalizers domain-shaped and
means a new vertical inherits audience tagging for free.

**These are heuristics, not certified ratings.** The signal is a keyword match
over the catalogue's own category text plus the content type's declared default
from ``config/domains.yaml``. That is honest enough for ranking and for the
constraint evaluation, and it fails *closed*: a row whose categories say nothing
falls back to its content type's default rather than to "unrestricted". Anywhere
a real certification is available (a BBFC/CBFC rating, an explicit-lyrics flag,
an Rx-only marker) it should replace this, and ``item_min_age`` already prefers
an explicit ``audience_min_age`` when one is present.
"""

from __future__ import annotations

import re

import pandas as pd

from backend.audience import AUDIENCE_COLUMNS
from backend.domains import ContentTypeSpec, DomainRegistry, get_registry

# Ordered most-restrictive first: the first rule that matches a row's category
# text wins, so "young adult horror" is tagged adult rather than teen.
MATURITY_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "restricted",
        (
            "adults only", "adult only", "pornograph", "x-rated",
            # Hardcore terms, added after Last.fm year tags pulled 24 porn
            # scene listings into the music catalogue - every one of them rated
            # `teen`, because music rows carry no genre text for these rules to
            # match and so fell through to the content type's default.
            #
            # These are stems with no false positives against all 106k
            # catalogue titles. They are NOT sufficient on their own: of those
            # 24 listings, 10 contained no explicit word at all ("Stepmoms
            # Protein Supplements", "Stepson Involved"). Keywords cannot
            # classify those, so ingestion filtering is the real defence and
            # this is the backstop.
            "gangbang", "gang bang", "pornstar", "porn star",
            "blowjob", "blow job", "creampie", "cream pie",
            "cumshot", "cum shot", "deepthroat", "deep throat",
            "bukkake", "fisting", "hentai",
            "double penetrat", "double vaginal",
        ),
    ),
    (
        "adult",
        (
            "erotic",
            "explicit",
            "horror",
            "mature content",
            "true crime",
            "gore",
            "gangster",
            "murder",
            "slasher",
        ),
    ),
    (
        "teen",
        (
            "young adult",
            "teen",
            "juvenile nonfiction",
            # Violence/crime signals. Without these a row reading
            # "Animation, Science Fiction, Thriller" fell through every
            # restrictive rule and matched "animation" as all_ages.
            "thriller",
            "crime",
            "violence",
            "violent",
            "suspense",
            "noir",
            "mystery",
        ),
    ),
    (
        "all_ages",
        (
            "animation",
            "children",
            "juvenile fiction",
            "kids",
            "family",
            "picture book",
            "nursery",
            "fairy tale",
            "early reader",
        ),
    ),
]


# Keywords that must match as a whole word, not as a stem. Kept separate from
# MATURITY_CATEGORY_RULES because the two need different boundary handling and
# using the wrong one is silently destructive in opposite directions: a stem
# match on these fires on ordinary words, while a whole-word match on
# "pornograph" would stop matching "pornography".
#
# Every candidate here was measured against all 106k catalogue titles first.
# Rejected on that evidence, with their false-positive counts:
#   "anal" as a stem  - 547 hits: "Final Analysis", "Analyze This", "El analfabeto"
#   "xxx"  as a stem  -  31 hits: the film "xXx" and its sequels
#   "ass"  whole word -  25 hits: "Kick-Ass", "Bad Ass", "A Pain in the Ass"
#   "dap"  whole word -  17 hits: DAP-brand caulk in the health catalogue
#   "bbc"  whole word -   6 hits: "BBC Children In Need", "Ex-Factor - BBC Live"
#   "cock" whole word -  23 hits: "A Cock and Bull Story", "Cock the Hammer"
#   "xxx"  whole word -  14 hits: the "xXx" films, "XXX. FEAT. U2.", and -
#                        the one nobody predicts - "XXX-Large" clothing sizes
#                        in the health catalogue.
# Do not add any of those without re-measuring.
MATURITY_WHOLE_WORD_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("restricted", ("slut", "sluts", "milf", "milfs", "dped")),
    # `adult`, not `restricted`: these are films *about* the industry, not
    # hardcore material. All 10 catalogue hits are adult-themed and 7 of them
    # were rated `teen` - a 13-year-old was served "Bikini Porn" and
    # "After Porn Ends 2". The stem "pornograph" above is unaffected, because a
    # whole-word "porn" cannot match inside "pornography".
    ("adult", ("porn", "porno", "porns")),
]


LEFT_BOUNDARY = "(?<![a-z])"
RIGHT_BOUNDARY = "(?![a-z])"


def _compile(
    rules: list[tuple[str, tuple[str, ...]]],
    whole_word_rules: list[tuple[str, tuple[str, ...]]] | None = None,
) -> list[tuple[str, re.Pattern[str]]]:
    # Match only at a left word boundary, so "teen" no longer fires on
    # "canteen"/"sixteen" and "war" would not fire on "award". A symmetric
    # word-boundary assertion would break stem keywords such as "pornograph"
    # (it has to still match "pornography"), so this is a negative lookbehind
    # on the left only. The haystack is lowercased before matching.
    whole_words = dict(whole_word_rules or [])

    compiled = []
    for maturity, keywords in rules:
        alternatives = [LEFT_BOUNDARY + re.escape(k) for k in keywords]
        alternatives += [
            LEFT_BOUNDARY + re.escape(k) + RIGHT_BOUNDARY
            for k in whole_words.get(maturity, ())
        ]
        compiled.append((maturity, re.compile("|".join(alternatives))))
    return compiled


_COMPILED_RULES = _compile(MATURITY_CATEGORY_RULES, MATURITY_WHOLE_WORD_RULES)


def maturity_for_row(
    content_type: str,
    category_text: str,
    registry: DomainRegistry | None = None,
) -> str:
    """Best-effort maturity label for one row.

    Falls back to the content type's declared default when the category text
    carries no signal, which is the fail-closed behaviour the constraint layer
    relies on.
    """
    registry = registry or get_registry()
    try:
        spec = registry.content_type(content_type)
    except (ValueError, KeyError):
        return "adult"  # An unrecognised type is gated, not waved through.

    if spec.maturity_source != "fixed":
        haystack = str(category_text or "").lower()
        if haystack:
            for maturity, pattern in _COMPILED_RULES:
                if pattern.search(haystack):
                    return _restrict_only(maturity, spec, registry)

    return spec.default_maturity


def _restrict_only(
    derived: str,
    spec: ContentTypeSpec,
    registry: DomainRegistry,
) -> str:
    """Where a keyword is a poor substitute for a rating, it may only restrict.

    Two cases, both of which produced items rated safe for children that were
    plainly not:

    *Regulated domains.* Health content types default to ``adult``. Letting a
    single loose category word demote an item below that default is how an iron
    supplement, a peppermint pesticide and a "Sexual Wellness, Bondage Gear" eye
    mask were all tagged ``all_ages``: their category text merely happened to
    contain a word from the all-ages rule.

    *Types rated by a board.* A content type declaring
    ``maturity_source: certification`` has a real rating available, so a genre is
    never the better authority. "Animation" is a production technique, not an
    audience: it matched the all-ages rule and relaxed Akira, Heavy Metal,
    Grave of the Fireflies and Waltz with Bashir - adult war and science-fiction
    films - from the movie default of ``teen`` down to ``all_ages``, where an
    eight-year-old was served them. The certification from
    ``scripts.fetch_tmdb_certifications`` is applied before this runs and is
    preserved by :func:`annotate_audience`, so this path is reached only by
    titles no board rated, and those now hold at the default instead of being
    relaxed by their genre.

    Escalation is unaffected in both cases - a keyword can still push an item up
    to ``adult`` or ``restricted``, which is what keeps an unrated slasher out of
    a child's results.
    """
    domain = registry.domains.get(spec.domain)
    is_regulated = domain is not None and domain.is_regulated
    # `explicit_flag` is the third case, and it is the weakest of them: the
    # source ships no explicit-lyrics column, so the flag never fires and music
    # has no authoritative rating at all - only a genre, which says nothing
    # about audience. That let a title word relax 48 tracks from the `teen`
    # default down to `all_ages`: "Kids in America", "Sour Patch Kids", and -
    # the ones that matter - Drake's "Family Matters" and Baby Keem's "family
    # ties", both explicit rap, served to a child because their titles contain
    # "family". Escalation still works, so an explicit genre can still push a
    # track up.
    has_no_real_rating = spec.maturity_source in ("certification", "explicit_flag")
    if not is_regulated and not has_no_real_rating:
        return derived
    if registry.minimum_age(derived) < registry.minimum_age(spec.default_maturity):
        return spec.default_maturity
    return derived


def annotate_audience(
    catalog: pd.DataFrame,
    registry: DomainRegistry | None = None,
) -> pd.DataFrame:
    """Add the audience columns to a catalogue frame.

    Existing non-empty values are preserved, so a dataset that *does* carry a
    real rating keeps it and only the gaps are filled.
    """
    registry = registry or get_registry()
    annotated = catalog.copy()

    content_types = annotated["content_type"].astype(str).str.strip().str.lower()

    # The category text a maturity rule matches against: categories plus title,
    # since some datasets put "(Children's Edition)" only in the title.
    category_text = (
        annotated.get("categories", pd.Series("", index=annotated.index)).astype(str)
        + " "
        + annotated.get("title", pd.Series("", index=annotated.index)).astype(str)
    )

    # One rule evaluation per distinct (content_type, category_text) pair rather
    # than per row: catalogues repeat category strings heavily.
    pairs = pd.DataFrame({"content_type": content_types, "category_text": category_text})
    unique_pairs = pairs.drop_duplicates()
    resolved = {
        (row.content_type, row.category_text): maturity_for_row(
            row.content_type, row.category_text, registry
        )
        for row in unique_pairs.itertuples(index=False)
    }
    derived_maturity = pd.Series(
        [resolved[(ct, text)] for ct, text in zip(pairs["content_type"], pairs["category_text"])],
        index=annotated.index,
    )

    if "maturity" in annotated.columns:
        # fillna before astype(str): a missing value stringifies to "nan", which
        # is not empty, so it survived as the row's maturity and then resolved to
        # a minimum age of 0 - every unrated row rated safe for any age.
        existing = annotated["maturity"].fillna("").astype(str).str.strip()
        annotated["maturity"] = existing.where(existing != "", derived_maturity)
    else:
        annotated["maturity"] = derived_maturity

    annotated["domain"] = [
        _domain_for(content_type, registry) for content_type in content_types
    ]
    annotated["audience_min_age"] = [
        registry.minimum_age(maturity) for maturity in annotated["maturity"]
    ]
    annotated["risk_tier"] = [
        _risk_tier_for(content_type, registry) for content_type in content_types
    ]
    return annotated


def _domain_for(content_type: str, registry: DomainRegistry) -> str:
    try:
        return registry.domain_of(content_type).name
    except (ValueError, KeyError):
        return ""


def _risk_tier_for(content_type: str, registry: DomainRegistry) -> int:
    try:
        return registry.domain_of(content_type).risk_tier
    except (ValueError, KeyError):
        # Unknown provenance is treated as consequential, not as harmless.
        return 1


def audience_summary(catalog: pd.DataFrame) -> pd.DataFrame:
    """Row counts per domain and maturity, for the build log and the report."""
    missing = [column for column in AUDIENCE_COLUMNS if column not in catalog.columns]
    if missing:
        raise ValueError(f"Catalog is missing audience columns: {', '.join(missing)}")
    return (
        catalog.groupby(["domain", "content_type", "maturity"], dropna=False)
        .size()
        .reset_index(name="items")
        .sort_values(["domain", "content_type", "maturity"])
        .reset_index(drop=True)
    )
