# Every target runs from a plain checkout with no install: PYTHONPATH=src is all
# the package needs. `make install` is optional and only adds the `clinical-kg`
# console script.
PY      ?= python3
export PYTHONPATH := src

.PHONY: help install test clean-cases gazetteer entities measurements \
        relations umls-relations annotate evaluate pos-lexicon graph app all \
        p2 p2-tabelas p2-treinar p2-extrair p2-comparar app-p2

help:
	@echo "make test          run the test suite"
	@echo "make all           clean-cases -> entities -> measurements -> relations"
	@echo "make relations     extract relations into data/processed/relations.csv"
	@echo "make umls-relations  filter UMLS MRREL.RRF to this corpus's CUIs (needs the UMLS zip)"
	@echo "make evaluate      score relations against the gold set"
	@echo "make app           launch the Streamlit viewer"
	@echo "make p2            Projeto 2: tabelas -> treinar -> extrair -> comparar"
	@echo "make app-p2        Streamlit viewer for P1 x P2"
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
umls-relations:   ; $(PY) -m clinical_kg build-umls-relations
annotate:         ; $(PY) -m clinical_kg annotate-relations
evaluate:         ; $(PY) -m clinical_kg evaluate-relations
pos-lexicon:      ; $(PY) -m clinical_kg pos-lexicon
graph:            ; $(PY) -m clinical_kg export-graph

all: clean-cases entities measurements relations

# Projeto 2 (n-gramas, PMI, regressão logística): `make p2` roda tudo na ordem certa;
# o mesmo que `bash run_all.sh`, para quem não tem bash. Não precisa do gazetteer.
p2: p2-tabelas p2-treinar p2-extrair p2-comparar

p2-tabelas:	; $(PY) -m clinical_kg.learning.pipeline tabelas
p2-treinar:	; $(PY) -m clinical_kg.learning.pipeline treinar
p2-extrair:	; $(PY) -m clinical_kg.learning.pipeline extrair
p2-comparar:	; $(PY) -m clinical_kg.learning.pipeline comparar

app:
	streamlit run src/clinical_kg/app/main.py

# Projeto 2: visualizador P1 x P2 (não precisa do gazetteer UMLS)
app-p2:
	streamlit run src/clinical_kg/app/projeto2.py
