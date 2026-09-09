"""Audience eligibility: the constraint layer that runs before relevance.

Recommendation here is two-stage. Stage one decides what a given viewer is
*allowed* to see; stage two ranks whatever survives. Keeping them separate is
what lets the engine be pointed at a regulated catalogue at all: the age and
risk constraints are compiled into the vector-store query, so ineligible items
are never retrieved, never scored and cannot be reranked back into the results
by a personalization signal.

``eligibility_conditions`` builds that pre-filter. ``is_eligible`` re-checks the
same policy in plain Python, which is how ``scripts.evaluate_constraints``
measures a violation rate: the two are independent implementations of one rule,
so agreement is evidence and disagreement is a bug worth failing on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.domains import DomainRegistry, get_registry

# Catalogue columns the constraint layer reads. Written by the catalog build and
# uploaded verbatim into the Qdrant payload.
AUDIENCE_COLUMNS = [
    "domain",
    "maturity",
    "audience_min_age",
    "risk_tier",
]

# A request with no age states nothing about the viewer, so it cannot be treated
# as an adult. Restricted material stays hidden until an age is supplied.
UNKNOWN_AGE_CEILING = "teen"

# Safe mode caps what a request may see regardless of the viewer's real age, so
# an adult can deliberately ask for family-appropriate results.
SAFE_MODE_CEILING = "child"


@dataclass(frozen=True)
class AudienceContext:
    """Who is asking, and what they may be shown."""

    age: int | None = None
    domain: str | None = None
    safe_mode: bool = False
    max_risk_tier: int = 1
    registry: DomainRegistry = field(default_factory=get_registry, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.age is not None and not 0 <= int(self.age) <= 120:
            raise ValueError(f"age must be between 0 and 120, got {self.age}.")
        if self.domain is not None:
            # Raises on an unknown domain rather than silently filtering nothing.
            object.__setattr__(self, "domain", self.registry.normalize_domain(self.domain))

    @property
    def effective_age(self) -> int:
        """The age the policy reasons with.

        An absent age is not 'adult' and not 'zero': it is capped at the
        unknown-age ceiling, which keeps general content available while
        withholding anything gated above it.
        """
        if self.age is None:
            base = self.registry.minimum_age(UNKNOWN_AGE_CEILING)
        else:
            base = int(self.age)
        if self.safe_mode:
            return min(base, self.registry.minimum_age(SAFE_MODE_CEILING))
        return base

    @property
    def allowed_content_types(self) -> tuple[str, ...]:
        return self.registry.content_types_for_domain(self.domain)

    def advisory(self) -> str | None:
        """The advisory to surface for this request's domain, if any."""
        if self.domain is None:
            return None
        return self.registry.domain(self.domain).advisory


def item_min_age(item: dict[str, Any], registry: DomainRegistry | None = None) -> int:
    """Minimum viewer age for a catalogue item.

    Prefers the explicit ``audience_min_age`` column; falls back to the maturity
    label, then to the content type's declared default. A row that carries none
    of these is treated as its content type's default rather than as unrestricted.
    """
    registry = registry or get_registry()

    raw_age = item.get("audience_min_age", "")
    if raw_age not in ("", None):
        try:
            return max(0, int(float(raw_age)))
        except (TypeError, ValueError):
            pass

    maturity = item.get("maturity", "")
    if maturity:
        return registry.minimum_age(maturity)

    content_type = item.get("content_type", "")
    try:
        spec = registry.content_type(content_type)
    except (ValueError, KeyError):
        return 0
    return registry.minimum_age(spec.default_maturity)


def item_risk_tier(item: dict[str, Any], registry: DomainRegistry | None = None) -> int:
    registry = registry or get_registry()
    raw_tier = item.get("risk_tier", "")
    if raw_tier not in ("", None):
        try:
            return int(float(raw_tier))
        except (TypeError, ValueError):
            pass
    try:
        return registry.domain_of(item.get("content_type", "")).risk_tier
    except (ValueError, KeyError):
        return 0


def is_eligible(
    item: dict[str, Any],
    context: AudienceContext,
    registry: DomainRegistry | None = None,
) -> bool:
    """Whether one catalogue item may be shown to this viewer."""
    registry = registry or context.registry

    if context.domain is not None:
        content_type = str(item.get("content_type", "")).strip().lower()
        if content_type not in context.allowed_content_types:
            return False

    if item_min_age(item, registry) > context.effective_age:
        return False

    return item_risk_tier(item, registry) <= context.max_risk_tier


def explain_ineligibility(
    item: dict[str, Any],
    context: AudienceContext,
    registry: DomainRegistry | None = None,
) -> str | None:
    """Why an item was withheld, or None when it is eligible."""
    registry = registry or context.registry

    if context.domain is not None:
        content_type = str(item.get("content_type", "")).strip().lower()
        if content_type not in context.allowed_content_types:
            return f"content type '{content_type}' is outside domain '{context.domain}'"

    minimum = item_min_age(item, registry)
    if minimum > context.effective_age:
        if context.safe_mode:
            return f"requires age {minimum}+, above the safe-mode ceiling of {context.effective_age}"
        if context.age is None:
            return f"requires age {minimum}+ and no age was supplied"
        return f"requires age {minimum}+, viewer is {context.age}"

    tier = item_risk_tier(item, registry)
    if tier > context.max_risk_tier:
        return f"risk tier {tier} exceeds the permitted {context.max_risk_tier}"

    return None


def eligibility_conditions(
    context: AudienceContext,
    content_type: str | None = None,
) -> list[Any]:
    """Compile the context into Qdrant payload conditions.

    Returned as a list so the caller can drop it straight into ``Filter(must=...)``.
    An empty list means "no constraint", which is a valid outcome for an
    unfiltered adult request.
    """
    try:
        from qdrant_client.models import FieldCondition, MatchAny, MatchValue, Range
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "qdrant-client is required. Install with: pip install qdrant-client"
        ) from exc

    registry = context.registry
    conditions: list[Any] = []

    normalized_type = registry.normalize_content_type(content_type)
    if normalized_type is not None:
        conditions.append(
            FieldCondition(key="content_type", match=MatchValue(value=normalized_type))
        )
    elif context.domain is not None:
        conditions.append(
            FieldCondition(
                key="content_type",
                match=MatchAny(any=list(context.allowed_content_types)),
            )
        )

    # The age gate. Range(lte=...) on the item's own floor means an ineligible
    # point is never returned by the search, rather than filtered afterwards.
    conditions.append(
        FieldCondition(key="audience_min_age", range=Range(lte=float(context.effective_age)))
    )
    conditions.append(
        FieldCondition(key="risk_tier", range=Range(lte=float(context.max_risk_tier)))
    )
    return conditions


def build_eligibility_filter(
    context: AudienceContext,
    content_type: str | None = None,
) -> Any | None:
    """The full Qdrant filter for a request, or None when nothing is constrained."""
    conditions = eligibility_conditions(context, content_type)
    if not conditions:
        return None
    from qdrant_client.models import Filter

    return Filter(must=conditions)
