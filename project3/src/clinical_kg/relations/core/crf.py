#!/usr/bin/env python3
"""The linear-chain CRF: emission potentials, transitions and Viterbi decoding.

The model tags each sentence's token sequence with O / B-TRIG / I-TRIG / NEG to
locate relation *triggers*. Its transition potentials are the bigram model:
B->I keeps a multi-token trigger together ("was treated with"), B->B forbids two
adjacent independent triggers, and O->I is impossible by BIO validity.

Potentials are set by hand in `features.WEIGHTS` rather than learned, because
this project has no labeled relation data. Hand-setting them keeps Viterbi
inference -- and with it the sequence-level constraints a per-pair score cannot
express -- while giving up any claim that the values are optimal.

This module knows nothing about entities or relations; it only finds triggers.
Attachment, typing and assertion scoping live in `extract.py`.
"""

import math

from . import features as rf

TAGS = ("O", "B-TRIG", "I-TRIG", "NEG")
NEG_INF = float("-inf")


# --------------------------------------------------------------------------
# CRF
# --------------------------------------------------------------------------
def emissions(tokens, tags_pos, in_entity, freq):
    """Per-token emission potential for each tag, plus the weights that fired.

    Returns (scores, fired) where scores[i][tag] is a float and fired[i][tag]
    is the list of weight names contributing to it.
    """
    W = rf.WEIGHTS
    scores, fired = [], []
    for i, token in enumerate(tokens):
        low = token.lower()
        pos = tags_pos[i]
        row, names = {}, {}

        # A relation trigger phrase must be headed by a verb. Without this
        # hard constraint the lattice pulls a noun into B just to collect the
        # B->I bonus from a following preposition ("department with").
        verbal = (not in_entity[i]
                  and (low in rf.TRIGGER_LEXICON
                       or (pos == "VERB" and not rf.ABLATE["pos_backoff"])
                       or (low.endswith("ed") and pos not in ("NOUN", "ADJ"))))
        trig, hit = 0.0, []
        if in_entity[i]:
            trig += W["trig.in_entity"]; hit.append("trig.in_entity")
        elif not verbal:
            trig, hit = NEG_INF, ["trig.not_verbal"]
        else:
            if low in rf.TRIGGER_LEXICON:
                trig += W["trig.lexicon_hit"]; hit.append("trig.lexicon_hit")
            elif pos == "VERB" and not rf.ABLATE["pos_backoff"]:
                trig += W["trig.pos_verb"]; hit.append("trig.pos_verb")
                if low.endswith("ed"):
                    trig += W["trig.suffix_ed"]; hit.append("trig.suffix_ed")
            if low in rf.STOP_VERBS:
                trig += W["trig.stop_verb"]; hit.append("trig.stop_verb")
            if pos == "PREP" or low in rf.TRIGGER_PREPS:
                # Only rewarded as a continuation; the B-tag never wants a bare
                # preposition, which is why this lands on I-TRIG below.
                trig += W["trig.prep_alone"]; hit.append("trig.prep_alone")
            if pos in ("DET", "CONJ", "PRON"):
                trig += W["trig.pos_det_conj"]; hit.append("trig.pos_det_conj")
            if pos == "NOUN":
                trig += W["trig.pos_noun"]; hit.append("trig.pos_noun")
            if pos == "ADJ":
                trig += W["trig.pos_adj"]; hit.append("trig.pos_adj")
            count = freq.get(low, 1)
            trig += W["trig.freq_bucket"] * math.log10(max(count, 1) + 1)
            hit.append("trig.freq_bucket")

        row["B-TRIG"], names["B-TRIG"] = trig, list(hit)

        # I-TRIG is scored as a *continuation*, not as a second trigger. In
        # English the continuation of a trigger phrase is a preposition or
        # particle ("presented WITH", "followed BY", "started ON"); a content
        # verb landing here should open its own trigger instead.
        if in_entity[i]:
            cont, cont_hit = W["trig.in_entity"], ["trig.in_entity"]
        elif pos == "PREP" or low in rf.TRIGGER_PREPS:
            cont, cont_hit = W["trig.prep_after_trig"], ["trig.prep_after_trig"]
        elif low in rf.TRIGGER_LEXICON:
            cont, cont_hit = W["trig.i_content_verb"], ["trig.i_content_verb"]
        elif pos == "VERB":
            cont, cont_hit = 0.0, ["trig.pos_verb"]
        else:
            cont, cont_hit = W["trig.i_other"], ["trig.i_other"]
        row["I-TRIG"], names["I-TRIG"] = cont, cont_hit

        is_cue = low in rf.NEGATION_CUES and not in_entity[i]
        row["NEG"] = W["neg.cue_hit"] if is_cue else W["neg.other"]
        names["NEG"] = ["neg.cue_hit"] if is_cue else ["neg.other"]

        row["O"], names["O"] = 0.0, []
        scores.append(row)
        fired.append(names)
    return scores, fired


def transition(prev_tag: str, tag: str, run_len: int) -> float:
    """Transition potential. `run_len` is the current I-TRIG run length."""
    W = rf.WEIGHTS
    if tag == "I-TRIG":
        if prev_tag not in ("B-TRIG", "I-TRIG"):
            return NEG_INF                      # BIO validity: no O -> I
        if run_len >= W["trans.max_trigger_len"]:
            return NEG_INF                      # hard cap on trigger length
        return W["trans.B_I"] if prev_tag == "B-TRIG" else W["trans.I_I"]
    if rf.ABLATE["transitions"]:
        return 0.0
    key = {("O", "O"): "trans.O_O", ("O", "B-TRIG"): "trans.O_B",
           ("O", "NEG"): "trans.O_NEG", ("B-TRIG", "B-TRIG"): "trans.B_B",
           ("B-TRIG", "O"): "trans.B_O", ("I-TRIG", "O"): "trans.I_O",
           ("NEG", "O"): "trans.NEG_O", ("NEG", "B-TRIG"): "trans.NEG_B"}.get(
        (prev_tag, tag))
    return W.get(key, 0.0) if key else 0.0


def viterbi(emit) -> list[str]:
    """Decode the highest-scoring valid tag sequence.

    State is (tag, run_length) so the trigger-length cap is enforceable inside
    the lattice rather than by post-hoc trimming.
    """
    if not emit:
        return []
    cap = rf.WEIGHTS["trans.max_trigger_len"]
    # state: (tag, run) ; run only meaningful for B/I-TRIG
    start = [(t, 1 if t in ("B-TRIG", "I-TRIG") else 0) for t in TAGS
             if t != "I-TRIG"]
    best = {s: emit[0][s[0]] for s in start}
    back = [{}]
    for i in range(1, len(emit)):
        new, ptr = {}, {}
        for (ptag, prun), pscore in best.items():
            if pscore == NEG_INF:
                continue
            for tag in TAGS:
                trans = transition(ptag, tag, prun)
                if trans == NEG_INF:
                    continue
                run = prun + 1 if tag == "I-TRIG" else (1 if tag == "B-TRIG" else 0)
                if run > cap:
                    continue
                state = (tag, run)
                score = pscore + trans + emit[i][tag]
                if score > new.get(state, NEG_INF):
                    new[state] = score
                    ptr[state] = (ptag, prun)
        best, _ = new, ptr
        back.append(ptr)
    end = max(best, key=best.get)
    path = [end]
    for i in range(len(emit) - 1, 0, -1):
        end = back[i][end]
        path.append(end)
    return [tag for tag, _ in reversed(path)]


def decode_triggers(tokens, spans, tags):
    """Group B/I-TRIG runs into trigger phrases."""
    out, i = [], 0
    while i < len(tags):
        if tags[i] != "B-TRIG":
            i += 1
            continue
        j = i + 1
        while j < len(tags) and tags[j] == "I-TRIG":
            j += 1
        words = [tokens[k].lower() for k in range(i, j)]
        category = next((rf.TRIGGER_LEXICON[w] for w in words
                         if w in rf.TRIGGER_LEXICON), "")
        out.append({"i": i, "j": j, "start": spans[i][1], "end": spans[j - 1][2],
                    "words": words, "category": category})
        i = j
    return out


