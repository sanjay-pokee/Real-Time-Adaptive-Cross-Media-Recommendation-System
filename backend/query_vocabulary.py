"""What the catalogue's own words say about a query: typos and keyboard mash.

SBERT has no notion of a word it does not know. A misspelling or a mashed key
row is cut into word pieces and embedded like any other text, and its nearest
vectors come back with ordinary scores: "qwertyuiop" scored 0.53 against an item
titled "Qwerty", above real searches such as "dog" (0.46) and "f1" (0.48).
Across 53 real queries and 25 mashed ones the best-match scores overlap from
0.45 to 0.54, so no cosine floor drops the mash without dropping real searches.

The words themselves do separate them. A real query is made of words that
appear somewhere in 120,747 titles, genres, descriptions and creator names;
"qzxv", "blorf" and "asdfghjkl" appear nowhere. This module counts how many
catalogue rows use each word and answers two questions from that:

- has_known_word: does the query contain any word the catalogue uses? If not,
  vector search is skipped, since the nearest neighbours of noise are noise.
- correct: which unknown words are near-misses of a known one, and what were
  they meant to be? "intersteller" -> "interstellar".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product

import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein

# Function words name no topic, so they cannot vouch for a query: "the qzxv" is
# as much noise as "qzxv".
STOPWORDS = frozenset(
    "a an the of and or but to in on at for with by from as is are was were be "
    "it its this that these those i me my we our you your he she his her they "
    "them their what which who how about into than then so not no".split()
)

# Rows a word must appear in to count as real. One is not enough: listings carry
# typos too, and "intersteller" appears in exactly one of them.
KNOWN_ROWS = 2

# Rows a word must appear in to be offered as a correction. Title and creator
# words qualify regardless, because a title is worth reaching even when one
# item uses it: "shawshank" appears in two rows.
TARGET_ROWS = 3

# A word the catalogue has seen once is corrected only toward a word at least
# this many times more common; otherwise the rare word is taken as meant.
RARE_WORD_MARGIN = 10

MIN_CORRECTABLE_LENGTH = 4
CANDIDATES_PER_WORD = 3
# Up to this many misspelt words are corrected jointly, choosing the combination
# that forms a real title phrase. Past it, each takes its own best candidate.
MAX_JOINT_WORDS = 3


@dataclass(frozen=True)
class QueryVocabulary:
    rows: Counter
    names: frozenset[str]
    # hash((word, next_word)) for adjacent words in a title, so "hary poter"
    # resolves to the pair that forms one - "harry potter" - rather than to the
    # two individually commonest neighbours, "hard power".
    title_bigrams: frozenset[int]
    targets_by_length: dict[int, list[str]]
    catalog_rows: int

    @classmethod
    def build(
        cls,
        fulltext: pd.Series,
        titles: pd.Series,
        creators: pd.Series,
    ) -> QueryVocabulary:
        """Count words per row from the recommender's lookup haystacks.

        The haystacks are already tokenised the way a query is (lowercase, no
        accents, punctuation as spaces), so a query word and a catalogue word
        compare directly. Counting rows rather than occurrences keeps one
        listing that repeats a word forty times from making it common.
        """
        rows: Counter = Counter()
        names: set[str] = set()
        bigrams: set[int] = set()
        for text, title, people in zip(fulltext.values, titles.values, creators.values):
            title_words = title.split()
            name_words = title_words + people.split()
            words = set(text.split())
            words.update(name_words)
            rows.update(words)
            names.update(name_words)
            bigrams.update(hash(pair) for pair in zip(title_words, title_words[1:]))

        targets: dict[int, list[str]] = {}
        for word, count in rows.items():
            if word.isalpha() and len(word) >= 3 and (count >= TARGET_ROWS or word in names):
                targets.setdefault(len(word), []).append(word)

        return cls(
            rows=rows,
            names=frozenset(names),
            title_bigrams=frozenset(bigrams),
            targets_by_length=targets,
            catalog_rows=len(fulltext),
        )

    def is_known(self, word: str) -> bool:
        return self.rows.get(word, 0) >= KNOWN_ROWS

    def has_known_word(self, words: list[str]) -> bool:
        """True when any word that carries meaning is one the catalogue uses."""
        return any(
            self.is_known(word)
            for word in words
            if len(word) > 1 and word not in STOPWORDS
        )

    def correct(self, words: list[str]) -> list[str] | None:
        """The words with near-miss typos replaced, or None if none were found."""
        options = [self._candidates(word) or [word] for word in words]
        misspelt = [i for i, word in enumerate(words) if options[i] != [word]]
        if not misspelt:
            return None

        if len(misspelt) > MAX_JOINT_WORDS:
            return [choices[0] for choices in options]

        def fit(combo: tuple[str, ...]) -> tuple[int, int]:
            phrases = sum(hash(pair) in self.title_bigrams for pair in zip(combo, combo[1:]))
            rank = sum(choices.index(word) for word, choices in zip(combo, options))
            return phrases, -rank

        return list(max(product(*options), key=fit))

    def _candidates(self, word: str) -> list[str]:
        """Best corrections for one word, best first; empty if it needs none."""
        if len(word) < MIN_CORRECTABLE_LENGTH or not word.isalpha():
            return []
        seen = self.rows.get(word, 0)
        if seen >= KNOWN_ROWS:
            return []

        # Elasticsearch's AUTO fuzziness allows two edits from six letters; that
        # turned too many short mashed words into real ones, so two edits wait
        # until eight letters here.
        limit = 1 if len(word) <= 7 else 2
        least_rows = max(1, RARE_WORD_MARGIN * seen)
        matches: list[tuple[str, int]] = []
        for length in range(len(word) - limit, len(word) + limit + 1):
            pool = self.targets_by_length.get(length)
            if not pool:
                continue
            for candidate, distance, _ in process.extract(
                word,
                pool,
                scorer=DamerauLevenshtein.distance,
                score_cutoff=limit,
                limit=None,
            ):
                if candidate != word and self.rows[candidate] >= least_rows:
                    matches.append((candidate, int(distance)))

        # Fewest edits first. Among equals, the usual typing slips, then a
        # matching first letter (people rarely get that one wrong), then the
        # more common word.
        matches.sort(
            key=lambda match: (
                match[1],
                not _is_common_slip(word, match[0]),
                match[0][0] != word[0],
                -self.rows[match[0]],
            )
        )
        return [candidate for candidate, _ in matches[:CANDIDATES_PER_WORD]]


def _is_common_slip(typo: str, word: str) -> bool:
    """A doubled letter typed once, or two neighbouring letters swapped.

    Frequency alone gets both wrong: "recieve" is one edit from "relieve" and
    from "receive", and "relieve" is the commoner word in this catalogue.
    """
    if len(word) == len(typo) + 1:
        return any(
            word[i] == word[i + 1] and word[:i] + word[i + 1:] == typo
            for i in range(len(word) - 1)
        )
    if len(word) == len(typo):
        diff = [i for i in range(len(word)) if word[i] != typo[i]]
        return (
            len(diff) == 2
            and diff[1] == diff[0] + 1
            and word[diff[0]] == typo[diff[1]]
            and word[diff[1]] == typo[diff[0]]
        )
    return False
