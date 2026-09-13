#!/usr/bin/env python3
"""Cross-sentence patient coreference: rule-based, no ML.

Every case report in this corpus describes exactly one patient, so
coreference here never needs to disambiguate *among* entities -- only to
decide whether an elided or pronominal subject still refers to the patient.
`extract.py::patient_subject` already answers that *within* one clause using
`PATIENT_CUES` / `FAMILY_CUES`; what was missing is carrying that answer
*across* sentence boundaries, which is what a trigger with no left-hand
entity needs when its own clause has no explicit subject cue at all
("Diagnosed with pneumonia. Subsequently treated with ceftriaxone.").

The rule is recency, not general entity coreference: track one bit of state,
`patient_active`, updated at each sentence boundary from that sentence's own
cues, and inherited by the next sentence when it has no cue of its own.
"""

from . import features as rf


class CorefState:
    """One bit of state threaded across the sentences of a single case.

    Starts True: a case report's first sentence is conventionally about the
    patient even before any pronoun appears ("A 44-year-old woman presented
    with...").
    """

    def __init__(self):
        self.patient_active = True

    def resolve(self, tokens, upto):
        """Is the clause subject (up to token index `upto`) the patient?

        Checks this sentence's own cues first; only falls back to the
        inherited state when neither a patient nor a family/other-subject cue
        appears locally, which is what lets an elided subject ("Subsequently
        treated with...") continue referring to whoever the last sentence was
        about.
        """
        window = [t.lower() for t in tokens[:upto]]
        if any(w in rf.FAMILY_CUES for w in window):
            return False
        if any(w in rf.PATIENT_CUES for w in window):
            return True
        return self.patient_active

    def advance(self, tokens):
        """Update the inherited state from this sentence's own subject cues."""
        window = [t.lower() for t in tokens]
        if any(w in rf.FAMILY_CUES for w in window):
            self.patient_active = False
        elif any(w in rf.PATIENT_CUES for w in window):
            self.patient_active = True
        # No cue at all: state carries over unchanged (elided subject).
