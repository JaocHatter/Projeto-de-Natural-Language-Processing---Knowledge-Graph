# Every target runs from a plain checkout with no install: PYTHONPATH=src is all
# the package needs. `make install` is optional and only adds the `clinical-kg`
# console script.
PY      ?= python3
export PYTHONPATH := src

.PHONY: help install test clean-cases gazetteer entities measurements \
        relations annotate evaluate pos-lexicon graph app all

help:
	@echo "make test          run the test suite"
	@echo "make all           clean-cases -> entities -> measurements -> relations"
	@echo "make relations     extract relations into data/processed/relations.csv"
	@echo "make evaluate      score relations against the gold set"
	@echo "make app           launch the Streamlit viewer"
	@echo "make install       pip install -e . (adds the clinical-kg command)"

install:
	$(PY) -m pip install -e .

test:
	$(PY) -m unittest discover -s tests

clean-cases:      ; $(PY) -m clinical_kg clean-cases
gazetteer:        ; $(PY) -m clinical_kg build-gazetteer
entities:         ; $(PY) -m clinical_kg extract-entities
measurements:     ; $(PY) -m clinical_kg extract-measurements
relations:        ; $(PY) -m clinical_kg extract-relations
annotate:         ; $(PY) -m clinical_kg annotate-relations
evaluate:         ; $(PY) -m clinical_kg evaluate-relations
pos-lexicon:      ; $(PY) -m clinical_kg pos-lexicon
graph:            ; $(PY) -m clinical_kg export-graph

all: clean-cases entities measurements relations

app:
	streamlit run src/clinical_kg/app/main.py
