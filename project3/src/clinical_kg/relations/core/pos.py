#!/usr/bin/env python3
"""Coarse part-of-speech tagging: a hand-built lexicon plus suffix backoff.

POS tagging is not a goal of this project -- it is the *backoff* that makes the
bigram relation features generalize. Measured on this corpus, 87% of the
bigrams occurring between two entities are seen exactly once (2,156 of 2,489
distinct types), and only 130 occur three or more times. So a purely lexical
bigram model memorizes ~130 strings and is blind to everything else:
"presented by" occurs once in the entire corpus and "reported with" occurs
zero times, yet both are VERB+PREP and both signal a relation.

The lexicon (rather than suffix rules alone) exists to fix one specific error
class: suffix rules cannot tell a plural noun from a verb, so they wrongly
admit "areas of", "episodes of", "months of", "doses of" as VERB+PREP. Every
such word is frequent, so a few hundred hand-tagged entries remove the whole
class.

Tags: VERB NOUN ADJ ADV PREP CONJ DET PRON NUM PUNCT (plus ENT for tokens
covered by an entity span, which are never trigger candidates).

Usage:
    clinical-kg pos-lexicon            # dump the lexicon + a coverage report
"""

import argparse
import csv
import re
from pathlib import Path

from clinical_kg.paths import DEFAULT_POS_LEXICON, DEFAULT_TOKEN_FREQ

DEFAULT_OUT = DEFAULT_POS_LEXICON
DEFAULT_FREQ = DEFAULT_TOKEN_FREQ

# --- Closed classes: enumerated exhaustively, since they genuinely are closed.
DET = """the a an this that these those any all both each every no some
    another either neither such"""
PRON = """she he it we they i you her his its their our my your him them us me
    himself herself itself themselves who whom whose which what one"""
PREP = """of with to in for on at from by during without due into through over
    under after before within per than about since against between among
    across upon along toward towards near via until despite regarding
    following including according out up down off around behind beyond
    beside besides throughout onto off inside outside"""
CONJ = """and or but however while whereas although though because therefore
    also then thus moreover furthermore nor so if whether as when where
    once unless whenever"""
# Auxiliaries and modals are verbs, but never relation triggers on their own;
# the trigger scorer subtracts them explicitly.
AUX = """was were is are be been being am had has have having did does do
    will would can could may might must should shall"""

# --- Open-class hand tagging, restricted to what suffix rules get wrong.
# Plural nouns ending in -s: the entire reason the lexicon exists.
NOUN_S = """areas months days weeks years doses episodes features levels
    results tests cells values changes findings signs symptoms lesions masses
    patients images tubes abnormalities cultures isolates criteria antibodies
    fistulae tissues nodes limits complaints vitals units cycles sessions
    hours minutes times sides margins borders walls vessels bones joints
    lungs kidneys eyes ears legs arms hands feet ribs teeth bowels
    metastases series studies biopsies therapies stents drains sutures
    medications antibiotics symptoms tumors cysts nodules abscesses"""
# -ed forms that are adjectival far more often than verbal here.
ADJ_ED = """elevated enlarged increased decreased computed mixed related
    advanced prolonged sustained marked distended dilated thickened impaired
    reduced diffused localized generalized aged combined preserved
    unremarkable affected involved known used"""
# -ing forms that are nouns (nominalizations), not verbs.
NOUN_ING = """bleeding swelling imaging testing monitoring sequencing opening
    screening thickening scarring dressing bruising vomiting breathing
    training setting reading finding building morning evening"""
# -ing forms that really are verbs.
VERB_ING = """showing revealing resulting using measuring having requiring
    involving considering taking causing producing yielding confirming
    demonstrating indicating presenting suggesting containing extending
    undergoing receiving developing worsening improving"""
# High-frequency content verbs. These double as the seed for the trigger
# lexicon, but POS tagging keeps them separate from relation semantics.
VERB = """showed reveal revealed reveals show shows performed perform performs
    presented present presents underwent undergo undergoes developed develop
    started start improve improved discharged confirmed confirm diagnosed
    diagnose observed initiated followed demonstrated identified placed
    reported report admitted indicated obtained remained compared associate
    treated treat denied denies deny measured continued transferred referred
    occurred administered repeated found note noted made given seen done
    removed maintained controlled revealed detected detect noted underwent
    received receive require required suggest suggested consist consisted
    involve involved cause caused lead led resolve resolved persist persisted
    complain complained experience experienced undergo tolerate tolerated
    prescribe prescribed discontinue discontinued initiate begin began
    complete completed schedule scheduled plan planned attempt attempted"""
ADV = """not very well also only further later subsequently initially
    previously spontaneously respectively approximately significantly
    markedly mildly severely again still yet now already almost about
    however therefore thus otherwise instead rather quite too"""

_GROUPS = [("DET", DET), ("PRON", PRON), ("PREP", PREP), ("CONJ", CONJ),
           ("AUX", AUX), ("NOUN", NOUN_S), ("ADJ", ADJ_ED), ("NOUN", NOUN_ING),
           ("VERB", VERB_ING), ("VERB", VERB), ("ADV", ADV)]


def _build() -> dict[str, str]:
    """Earlier groups win, so closed classes beat the open-class lists."""
    lex: dict[str, str] = {}
    for tag, words in _GROUPS:
        for word in words.split():
            lex.setdefault(word, tag)
    return lex


LEXICON = _build()

# Suffix backoff for everything not in the lexicon, tried longest first.
# "-ing" is deliberately absent: on this corpus its top hits are "during",
# "following", "using", "including", "according", "being" plus symptom
# nominalizations ("bleeding", "swelling", "vomiting"), so a blanket
# "-ing implies verb" rule costs more than it earns. Same for "-ify", which
# never fires here at all.
SUFFIX_RULES = [
    ("ously", "ADV"), ("ally", "ADV"), ("ely", "ADV"), ("ily", "ADV"),
    ("itis", "NOUN"), ("osis", "NOUN"), ("emia", "NOUN"), ("oma", "NOUN"),
    ("pathy", "NOUN"), ("ectomy", "NOUN"), ("otomy", "NOUN"), ("scopy", "NOUN"),
    ("plasty", "NOUN"), ("gram", "NOUN"), ("graphy", "NOUN"), ("ment", "NOUN"),
    ("tion", "NOUN"), ("sion", "NOUN"), ("ness", "NOUN"), ("ity", "NOUN"),
    ("ance", "NOUN"), ("ence", "NOUN"), ("ism", "NOUN"), ("ist", "NOUN"),
    ("ous", "ADJ"), ("ary", "ADJ"), ("ive", "ADJ"), ("able", "ADJ"),
    ("ible", "ADJ"), ("ical", "ADJ"), ("al", "ADJ"), ("ic", "ADJ"),
    ("ar", "ADJ"), ("ile", "ADJ"), ("less", "ADJ"), ("ful", "ADJ"),
    ("ed", "VERB"),   # the one productive verb suffix on this corpus
]

NUM_RE = re.compile(r"^\d+(?:[.,]\d+)*$")


def tag_token(token: str, prev_tag: str = "") -> str:
    """Coarse tag for one token. `prev_tag` disambiguates -ing forms."""
    low = token.lower()
    if NUM_RE.match(token):
        return "NUM"
    if low in LEXICON:
        return LEXICON[low]
    if low.endswith("ing"):
        # An -ing word is a verb only under an auxiliary ("was showing");
        # standalone it is far likelier to be a nominalization here.
        return "VERB" if prev_tag == "AUX" else "NOUN"
    for suffix, tag in SUFFIX_RULES:
        if low.endswith(suffix) and len(low) > len(suffix) + 1:
            return tag
    return "NOUN"


def tag_sequence(tokens: list[str]) -> list[str]:
    tags, prev = [], ""
    for token in tokens:
        prev = tag_token(token, prev)
        tags.append(prev)
    return tags


def main():
    ap = argparse.ArgumentParser(
        description="Dump the POS lexicon and report its corpus coverage.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--freq", type=Path, default=DEFAULT_FREQ)
    args = ap.parse_args()

    lex_rows = sorted(LEXICON.items())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "tag", "source"])
        for token, tag in lex_rows:
            writer.writerow([token, tag, "hand"])
    print(f"lexicon entries : {len(lex_rows)}")
    print(f"written to      : {args.out}")

    if args.freq.exists():
        with open(args.freq, newline="", encoding="utf-8") as f:
            freq = [(r["token"], int(r["frequency"])) for r in csv.DictReader(f)]
        total = sum(n for _, n in freq)
        covered = sum(n for t, n in freq if t in LEXICON)
        print(f"\ntoken coverage  : {covered:,} / {total:,} ({covered / total:.1%}) "
              f"of corpus tokens tagged from the lexicon, rest by suffix backoff")
        print("\n  Tag distribution over the 400 most frequent tokens:")
        counts: dict[str, int] = {}
        for token, _ in freq[:400]:
            counts[tag_token(token)] = counts.get(tag_token(token), 0) + 1
        for tag, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {tag:<6} {count:>4}")


if __name__ == "__main__":
    main()
