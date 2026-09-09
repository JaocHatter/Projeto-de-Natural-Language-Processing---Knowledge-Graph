#!/usr/bin/env python3
"""Features, trigger lexicon and the hand-set weight table for relation scoring.

There is no labeled relation data in this project, so these weights are set by
hand rather than learned. That makes two things non-negotiable:

  1. Every weight lives in this one module, in WEIGHTS, so the whole model can
     be read and re-tuned in one place.
  2. Every scoring decision reports which weights fired (`rule_path` in the
     output CSV), because an explanation is the only defence a hand-weighted
     model has.

The weights are log-space potentials of a linear-chain CRF. Setting them by
hand instead of fitting them keeps the CRF's inference -- Viterbi decoding over
the tag sequence -- while giving up any claim that the values are optimal. If a
labeled set ever exists, these become the initialization and the same feature
code fits properly.
"""

# --------------------------------------------------------------------------
# Trigger lexicon: the closed set of relation-bearing verbs and phrases, mined
# from this corpus's own frequent inter-entity contexts. The category chosen
# here decides the relation type emitted downstream.
# --------------------------------------------------------------------------
TRIGGER_CATEGORIES = {
    "REVEAL": """showed show shows revealed reveal reveals revealing showing
        demonstrated demonstrate demonstrates demonstrating indicated indicate
        indicates identified identify detected detect found note noted notes
        observed observe confirmed confirm confirms measured yielded yield
        disclosed visualized visualised documented""",
    "TREAT": """treated treat treats underwent undergo undergoes undergoing
        received receive receives administered administer given start started
        starting initiated initiate prescribed prescribe performed perform
        performs performing placed place implanted resected removed excised
        continued discontinued managed manage undergone injected infused
        transfused operated repaired replaced drained""",
    "DIAGNOSE": """diagnosed diagnose diagnoses consistent suggestive
        compatible confirmed established indicative diagnostic""",
    "PRESENT": """presented present presents presenting complained complain
        complains reported report reports developed develop develops
        developing experienced experience exhibited manifested denied deny
        denies had has have admitted referred""",
    "CAUSE": """caused cause causes causing resulted resulting result led
        leading lead secondary due attributed attributable provoked triggered
        precipitated""",
    "LOCATE": """involving involved involves located locating arising
        extending extended originating affecting affected""",
}
TRIGGER_LEXICON = {word: category
                   for category, words in TRIGGER_CATEGORIES.items()
                   for word in words.split()}

# Prepositions that continue a trigger ("presented *with*", "accompanied *by*").
# This is the pattern that makes the bigram feature generalize: "presented by"
# occurs once in this corpus and "reported with" occurs zero times, yet both
# are VERB+PREP and both signal a relation.
TRIGGER_PREPS = frozenset("""with by for to of in on from due after into
    through as at within over against""".split())

# Locative prepositions, which turn a BodyPart tail into LOCATED_IN.
LOCATIVE_PREPS = frozenset("in of on at within into over under along near".split())

# Auxiliaries and light verbs: verbs, but not triggers on their own.
STOP_VERBS = frozenset("""was were is are be been being am had has have having
    did does do will would can could may might must should shall""".split())

# --------------------------------------------------------------------------
# Assertion cues. 293 negation-cue tokens occur in this corpus; without them
# "revealed abdominal pain ... but no rebound tenderness" yields an affirmed
# edge for a symptom the text explicitly denies.
# --------------------------------------------------------------------------
NEGATION_CUES = frozenset("""no not without denied denies deny negative
    absence absent free lack lacking none neither nor never ruled excluded
    unremarkable failed non""".split())
# Phrases checked as bigrams before the single-token cues.
NEGATION_PHRASES = (("no", "evidence"), ("negative", "for"), ("free", "of"),
                    ("ruled", "out"), ("absence", "of"), ("without", "any"))
HEDGE_CUES = frozenset("""possible possibly probable probably likely unlikely
    suspected suspicious suggestive consistent may might could presumed
    presumptive apparent apparently questionable concerning differential
    potential potentially""".split())
HISTORICAL_CUES = frozenset("""history previously previous past prior formerly
    earlier remote childhood chronic""".split())
FAMILY_CUES = frozenset("""family familial mother father sister brother parents
    parent son daughter maternal paternal grandmother grandfather sibling
    relatives hereditary""".split())
# A negation scope stops here rather than running to the end of the sentence.
SCOPE_TERMINATORS = frozenset("""but however although though except aside
    apart otherwise nevertheless whereas while""".split())

# Words indicating the clause subject is the patient, anchoring to the Person.
PATIENT_CUES = frozenset("""patient patient's he she his her him herself
    himself man woman male female boy girl child infant newborn baby""".split())

# --------------------------------------------------------------------------
# THE WEIGHT TABLE. Everything the model believes, in one place.
# --------------------------------------------------------------------------
WEIGHTS = {
    # -- CRF emission potentials for the B-TRIG / I-TRIG tags ---------------
    "trig.lexicon_hit":        3.0,   # token is a curated relation trigger
    "trig.pos_verb":           1.2,   # unseen verb: the POS backoff earning its keep
    "trig.stop_verb":         -2.5,   # auxiliary/light verb alone is not a trigger
    "trig.prep_after_trig":    1.8,   # "presented WITH", "accompanied BY"
    "trig.prep_alone":        -1.0,   # a preposition with no verb before it
    # I-TRIG means *continuation*. A content verb in continuation position
    # should start its own trigger instead, or a noun gets pulled into B just
    # to collect the B->I bonus ("examination revealed" as one trigger).
    "trig.i_content_verb":    -2.0,
    "trig.i_other":           -2.5,
    "trig.pos_det_conj":      -3.0,   # determiners/conjunctions are never triggers
    "trig.in_entity":         -6.0,   # tokens inside an entity span are arguments
    "trig.pos_noun":          -1.2,
    "trig.pos_adj":           -1.5,
    "trig.suffix_ed":          0.6,   # backoff for unseen -ed verbs
    "trig.freq_bucket":        0.15,  # x log10(corpus frequency); rare words rarely trigger
    # -- CRF emission potentials for the NEG tag ---------------------------
    "neg.cue_hit":             4.0,
    "neg.other":              -4.0,
    # -- CRF transition potentials: this is the bigram model ---------------
    "trans.O_B":               0.0,
    "trans.O_O":               0.3,
    "trans.B_I":               1.5,   # keep multi-token triggers ("was treated with")
    "trans.I_I":               0.4,
    "trans.B_B":              -3.0,   # no two adjacent independent triggers
    "trans.B_O":               0.0,
    "trans.I_O":               0.0,
    "trans.O_NEG":             0.0,
    "trans.NEG_O":             0.0,
    "trans.NEG_B":             0.5,   # "no evidence of" often precedes a trigger
    "trans.max_trigger_len":   3,     # hard cap; I->I beyond this is -inf
    # -- Pair-level potentials, added to the trigger score -----------------
    "pair.dist_0":             0.8,   # entities adjacent (142 pairs in corpus)
    "pair.dist_1_3":           1.0,   # 1,078 pairs
    "pair.dist_4_10":          0.2,   # 1,133 pairs
    "pair.dist_gt_10":        -1.5,   # 731 pairs; mostly unrelated
    "pair.same_clause":        1.2,   # the window that replaces "same paragraph"
    "pair.cross_clause":      -0.8,
    "pair.per_intervening":   -0.45,  # each entity in between dilutes the link
    "pair.max_intervening":    3,     # hard prune
    "pair.max_token_gap":      20,    # hard prune
    "pair.type_prior":         1.0,   # x ENTITY_PAIR_PRIOR below
    "pair.no_trigger":        -1.2,   # adjacency alone, no trigger found
    "pair.coordinate":         2.0,   # sibling in a coordinated list
    "pair.patient_anchor":     1.5,   # attaches to the Person root
    # -- Combining trigger and pair evidence -------------------------------
    "score.trigger_weight":    0.6,   # x mean trigger emission potential
    # -- Decision threshold ------------------------------------------------
    "decide.threshold":        1.5,
}

# Entity-type pair priors. Derived from which pairs are clinically meaningful,
# not from raw co-occurrence counts -- BodyPart+BodyPart co-occurs constantly
# (180 adjacent pairs) but is usually just anatomical modification.
ENTITY_PAIR_PRIOR = {
    ("Exam", "Finding"): 1.0, ("Exam", "Diagnosis"): 1.0,
    ("Exam", "Symptom"): 0.6, ("Exam", "BodyPart"): 0.4,
    ("Diagnosis", "Treatment"): 1.0, ("Symptom", "Treatment"): 0.7,
    ("Finding", "Treatment"): 0.7, ("Diagnosis", "Symptom"): 0.8,
    ("Diagnosis", "Finding"): 0.8, ("Symptom", "Diagnosis"): 0.8,
    ("Finding", "Diagnosis"): 0.9, ("Treatment", "Diagnosis"): 0.6,
    ("Diagnosis", "BodyPart"): 0.7, ("Symptom", "BodyPart"): 0.7,
    ("Finding", "BodyPart"): 0.7, ("Treatment", "BodyPart"): 0.6,
    ("Exam", "Exam"): 0.1, ("BodyPart", "BodyPart"): -0.3,
    ("Treatment", "Treatment"): 0.2, ("Finding", "Finding"): 0.2,
    ("Diagnosis", "Diagnosis"): 0.2, ("Symptom", "Symptom"): 0.2,
}
DEFAULT_PAIR_PRIOR = 0.0

# Trigger category x argument types -> relation. Falls back to TYPE_FALLBACK.
RELATION_BY_CATEGORY = {
    ("REVEAL", "Finding"): "REVEALED_BY", ("REVEAL", "Diagnosis"): "REVEALED_BY",
    ("REVEAL", "Symptom"): "REVEALED_BY", ("REVEAL", "BodyPart"): "LOCATED_IN",
    ("TREAT", "Treatment"): "TREATED_WITH", ("TREAT", "Exam"): "TREATED_WITH",
    ("DIAGNOSE", "Diagnosis"): "HAS_DIAGNOSIS", ("DIAGNOSE", "Finding"): "HAS_DIAGNOSIS",
    ("PRESENT", "Symptom"): "HAS_SYMPTOM", ("PRESENT", "Finding"): "HAS_FINDING",
    ("PRESENT", "Diagnosis"): "HAS_DIAGNOSIS", ("PRESENT", "Treatment"): "TREATED_WITH",
    ("CAUSE", "Diagnosis"): "CAUSED_BY", ("CAUSE", "Finding"): "CAUSED_BY",
    ("CAUSE", "Symptom"): "CAUSED_BY", ("CAUSE", "Treatment"): "CAUSED_BY",
    ("LOCATE", "BodyPart"): "LOCATED_IN",
}
TYPE_FALLBACK = {"Symptom": "HAS_SYMPTOM", "Diagnosis": "HAS_DIAGNOSIS",
                 "Finding": "HAS_FINDING", "Treatment": "TREATED_WITH",
                 "Exam": "REVEALED_BY", "BodyPart": "LOCATED_IN"}

RELATIONS = ("HAS_SYMPTOM", "HAS_DIAGNOSIS", "HAS_FINDING", "TREATED_WITH",
             "REVEALED_BY", "LOCATED_IN", "CAUSED_BY", "COORDINATE_WITH")
ASSERTIONS = ("affirmed", "negated", "hedged", "historical", "family")


# Ablation switches, flipped by evaluate_relations.py. A hand-weighted model
# cannot claim its weights are optimal, so the defensible claim is a component
# one: this is what each piece of the design actually buys.
ABLATE = {"pos_backoff": False, "coordination": False,
          "negation": False, "transitions": False}


def pair_prior(head_type: str, tail_type: str) -> float:
    return ENTITY_PAIR_PRIOR.get((head_type, tail_type), DEFAULT_PAIR_PRIOR)


# Which argument type belongs in the TAIL slot of each relation. An edge whose
# tail violates this but whose head satisfies it is pointing backwards:
# "right ovary --REVEALED_BY--> laparoscopic exploration" is the exploration
# revealing the ovary, not the reverse.
TAIL_TYPES = {
    "REVEALED_BY": {"Finding", "Diagnosis", "Symptom", "BodyPart"},
    "TREATED_WITH": {"Treatment"},
    "LOCATED_IN": {"BodyPart"},
    "HAS_SYMPTOM": {"Symptom"},
    "HAS_DIAGNOSIS": {"Diagnosis"},
    "HAS_FINDING": {"Finding"},
}
HEAD_TYPES = {"REVEALED_BY": {"Exam"}, "TREATED_WITH": {"Diagnosis", "Finding",
              "Symptom", "Person"}, "LOCATED_IN": {"Diagnosis", "Finding",
              "Symptom", "Treatment", "Exam", "BodyPart"}}


def orient(head, tail, relation):
    """Return (head, tail, relation, swapped) with the edge pointing the way
    the relation's type signature requires."""
    if head is None or relation not in TAIL_TYPES:
        return head, tail, relation, False
    head_type, tail_type = head["entity_type"], tail["entity_type"]
    if tail_type in TAIL_TYPES[relation]:
        return head, tail, relation, False
    if head_type in TAIL_TYPES[relation]:
        # Swapping only helps if the new head is admissible too.
        allowed = HEAD_TYPES.get(relation)
        if allowed is None or tail_type in allowed:
            return tail, head, relation, True
    return head, tail, relation, False


def relation_for(category: str, tail_type: str) -> str:
    if (category, tail_type) in RELATION_BY_CATEGORY:
        return RELATION_BY_CATEGORY[(category, tail_type)]
    return TYPE_FALLBACK.get(tail_type, "HAS_FINDING")
