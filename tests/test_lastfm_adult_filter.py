"""Ingestion filter for the Last.fm fetcher.

Every title below is real: the bad ones are listings a year-tag fetch actually
pulled into the music catalogue, the good ones are titles from the Spotify
export and the cleaned Last.fm set that the filter must not touch.

The filter matters more than the maturity keywords do. A music row carries no
genre text, so the rules in preprocessing/audience_tagging.py have almost
nothing to match and these land rated `teen`. Keywords catch 14 of the 25;
this catches all 25.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fetch_lastfm_tracks import LONG_TITLE_CHARS, looks_like_adult_video


class TestAdultListingsAreCaught:
    @pytest.mark.parametrize("title", [
        "Private Gangbangs Vol. 5 (Scene 3)",
        "7on1 MEGA GANGBANG! Sweet Slut Sofia Smith gets double dicked",
        "Colombian pornstars Madison Stil and OliviaQueen getting used",
        "Beautiful teen Gabi Manhaes undergoes hardcore double penetration",
        "Priscila Belini - Double Vaginal Fantasy Of a Busty Beauty",
        "Anal-Crazy Hottie Gets DPed",
        "Cock-Hungry Hotties Have Intense Anal Foursome",
        "Stepmoms Protein Supplements",
        "Stepson Involved",
        "Coach Makes Sure My Balls Are Ready For The Game - S2:E10",
        "BBC-Crazy Hottie Goes Airtight",
        "Whats Shared Is Always Sweeter - S1:E8",
        "Jenifer Lol - DAP! Destruction of the beauty's holes: DVP and more",
    ])
    def test_caught(self, title):
        assert looks_like_adult_video(title)

    @pytest.mark.parametrize("title", [
        # No explicit term in any of these; only the length rule catches them.
        "HELENA STAR always loves to keep two BBCs in her ass, anal, DAP, deep balls",
        "HELENA STAR returns to get two huge cocks in her ass at the same time. DAP. Anal. Deep Balls.",
        "The beautiful SELENA PAIGE receives a welcome of 4 huge big cocks for her hot ass. Anal, deep ball.",
    ])
    def test_long_titles_with_weak_terms_are_caught(self, title):
        assert len(title) > LONG_TITLE_CHARS
        assert looks_like_adult_video(title)


class TestRealMusicSurvives:
    @pytest.mark.parametrize("title", [
        "Last One Standing (feat. Polo G, Mozzy & Eminem) - From Venom: Let There Be Carnage",
        "Lift Me Up - From Black Panther: Wakanda Forever - Music From the Motion Picture",
        "Dreamers [Music from the FIFA World Cup Qatar 2022 Official Soundtrack]",
        "I DID IT (feat. Post Malone, Megan Thee Stallion, Lil Baby & DaBaby)",
        "Don't You (Forget About Me) (feat. Tyler Connolly of Theory Of A Deadman)",
        "Way 2 Sexy (with Future & Young Thug)",
        "naked freestyle",
        "Rich Baby Daddy (feat. Sexyy Red & SZA)",
        "Sex = Money",
        "PerkySex",
        "Chemical Whore",
        "The Way (With XXXTENTACION)",
        "Reverberotic",
        "Kick-Ass",
        "Bad Ass",
        "Cock the Hammer",
    ])
    def test_not_flagged(self, title):
        assert not looks_like_adult_video(title)

    def test_a_long_title_alone_is_not_enough(self):
        # Length is never sufficient on its own - the 83-character soundtrack
        # titles above would all be lost if it were.
        title = "A" * 120
        assert len(title) > LONG_TITLE_CHARS
        assert not looks_like_adult_video(title)

    def test_a_weak_term_alone_is_not_enough(self):
        # "ass" and "huge" are ordinary words in a short title.
        assert not looks_like_adult_video("Bad Ass")
        assert not looks_like_adult_video("Huge")


class TestEdges:
    def test_empty_and_none_are_safe(self):
        assert not looks_like_adult_video("")
        assert not looks_like_adult_video(None)

    def test_matching_is_case_insensitive(self):
        assert looks_like_adult_video("PRIVATE GANGBANGS VOL. 5 (SCENE 3)")
