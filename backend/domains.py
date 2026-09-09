"""Domain registry: the taxonomy the engine is generic over.

The retrieval stack (SBERT -> Qdrant -> rerank) knows nothing about films or
lab equipment; it only ever sees text and a ``content_type`` payload field.
Everything that *is* vertical-specific -- which content types exist, which
domain they roll up to, how mature their catalogue skews, what advisory a
regulated domain needs -- is declared in ``config/domains.yaml`` and read here.

Adding a vertical is therefore a config change. This module is the single place
that answers "what content types exist?", so nothing downstream should hardcode
a media list.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "config" / "domains.yaml"


@dataclass(frozen=True)
class ContentTypeSpec:
    """One catalogue content type, e.g. ``movie`` or ``industrial``."""

    name: str
    domain: str
    label: str
    aliases: tuple[str, ...]
    default_maturity: str
    maturity_source: str


@dataclass(frozen=True)
class DomainSpec:
    """One vertical the engine serves, grouping one or more content types."""

    name: str
    label: str
    description: str
    risk_tier: int
    advisory: str | None
    content_types: tuple[str, ...]

    @property
    def is_regulated(self) -> bool:
        return self.risk_tier >= 1


@dataclass(frozen=True, eq=False)
class DomainRegistry:
    domains: dict[str, DomainSpec]
    content_types: dict[str, ContentTypeSpec]
    maturity_levels: dict[str, int]
    _alias_index: dict[str, str]

    # -- content types ----------------------------------------------------
    def normalize_content_type(self, content_type: str | None) -> str | None:
        """Resolve an alias ("films") to its canonical name ("movie")."""
        if content_type is None:
            return None
        key = str(content_type).strip().lower()
        if not key:
            return None
        resolved = self._alias_index.get(key)
        if resolved is None:
            allowed = ", ".join(sorted(self._alias_index))
            raise ValueError(
                f"Unknown content type '{content_type}'. Use one of: {allowed}."
            )
        return resolved

    def content_type(self, name: str) -> ContentTypeSpec:
        canonical = self.normalize_content_type(name)
        if canonical is None:
            raise ValueError("content_type cannot be empty.")
        return self.content_types[canonical]

    def content_type_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.content_types))

    def all_aliases(self) -> tuple[str, ...]:
        return tuple(sorted(self._alias_index))

    # -- domains ----------------------------------------------------------
    def normalize_domain(self, domain: str | None) -> str | None:
        if domain is None:
            return None
        key = str(domain).strip().lower()
        if not key:
            return None
        if key not in self.domains:
            allowed = ", ".join(sorted(self.domains))
            raise ValueError(f"Unknown domain '{domain}'. Use one of: {allowed}.")
        return key

    def domain(self, name: str) -> DomainSpec:
        canonical = self.normalize_domain(name)
        if canonical is None:
            raise ValueError("domain cannot be empty.")
        return self.domains[canonical]

    def domain_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.domains))

    def domain_of(self, content_type: str) -> DomainSpec:
        return self.domains[self.content_type(content_type).domain]

    def content_types_for_domain(self, domain: str | None) -> tuple[str, ...]:
        """Content types in a domain, or every known type when domain is None."""
        canonical = self.normalize_domain(domain)
        if canonical is None:
            return self.content_type_names()
        return self.domains[canonical].content_types

    # -- maturity ---------------------------------------------------------
    def minimum_age(self, maturity: str | None) -> int:
        """Minimum viewer age for a maturity level; unknown levels are permissive."""
        if maturity is None:
            return 0
        return self.maturity_levels.get(str(maturity).strip().lower(), 0)

    def maturity_at_or_below(self, age: int) -> tuple[str, ...]:
        return tuple(
            level
            for level, minimum in sorted(self.maturity_levels.items(), key=lambda kv: kv[1])
            if minimum <= age
        )


def load_registry(path: Path | str = DEFAULT_REGISTRY_PATH) -> DomainRegistry:
    """Read and validate ``config/domains.yaml``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Domain registry not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    maturity_levels = {
        str(level).lower(): int(minimum)
        for level, minimum in (raw.get("maturity_levels") or {}).items()
    }
    if not maturity_levels:
        raise ValueError(f"{path} declares no maturity_levels.")

    content_types: dict[str, ContentTypeSpec] = {}
    alias_index: dict[str, str] = {}
    for name, spec in (raw.get("content_types") or {}).items():
        canonical = str(name).strip().lower()
        aliases = tuple(
            dict.fromkeys([canonical, *(str(a).strip().lower() for a in spec.get("aliases", []))])
        )
        default_maturity = str(spec.get("default_maturity", "all_ages")).lower()
        if default_maturity not in maturity_levels:
            raise ValueError(
                f"Content type '{canonical}' has default_maturity '{default_maturity}', "
                f"which is not one of: {', '.join(sorted(maturity_levels))}."
            )
        content_types[canonical] = ContentTypeSpec(
            name=canonical,
            domain=str(spec.get("domain", "")).strip().lower(),
            label=str(spec.get("label", canonical.title())),
            aliases=aliases,
            default_maturity=default_maturity,
            maturity_source=str(spec.get("maturity_source", "fixed")).lower(),
        )
        for alias in aliases:
            if alias in alias_index and alias_index[alias] != canonical:
                raise ValueError(
                    f"Alias '{alias}' is claimed by both "
                    f"'{alias_index[alias]}' and '{canonical}'."
                )
            alias_index[alias] = canonical

    domains: dict[str, DomainSpec] = {}
    for name, spec in (raw.get("domains") or {}).items():
        canonical = str(name).strip().lower()
        declared = tuple(str(c).strip().lower() for c in spec.get("content_types", []))
        unknown = [c for c in declared if c not in content_types]
        if unknown:
            raise ValueError(
                f"Domain '{canonical}' lists unknown content types: {', '.join(unknown)}."
            )
        advisory = spec.get("advisory")
        domains[canonical] = DomainSpec(
            name=canonical,
            label=str(spec.get("label", canonical.title())),
            description=str(spec.get("description", "")).strip(),
            risk_tier=int(spec.get("risk_tier", 0)),
            advisory=str(advisory).strip() if advisory else None,
            content_types=declared,
        )

    orphans = sorted(
        name for name, spec in content_types.items() if spec.domain not in domains
    )
    if orphans:
        raise ValueError(
            f"Content types point at undeclared domains: {', '.join(orphans)}."
        )

    return DomainRegistry(
        domains=domains,
        content_types=content_types,
        maturity_levels=maturity_levels,
        _alias_index=alias_index,
    )


@lru_cache(maxsize=1)
def get_registry() -> DomainRegistry:
    """Process-wide registry. Cached: the file is read once per process."""
    return load_registry()


def normalize_content_type(content_type: str | None) -> str | None:
    """Module-level shim so callers need not thread the registry through."""
    return get_registry().normalize_content_type(content_type)
