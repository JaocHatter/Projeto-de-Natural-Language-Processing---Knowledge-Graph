# Every target runs from a plain checkout with no install: PYTHONPATH=src is all
# the package needs. `make install` is optional and only adds the `clinical-kg`
# console script.
PY      ?= python3
export PYTHONPATH := src

.PHONY: help install test clean-cases gazetteer entities measurements \
        relations umls-relations annotate evaluate pos-lexicon graph app all \
        p2 p2-tabelas p2-treinar p2-extrair p2-comparar app-p2 \
        p3 p3-maccrobat p3-externo p3-etapas p3-treinar p3-extrair p3-corpus p3-ida-volta \
        p3-figuras p3-llm-estimar p3-llm-avaliar app-p3

help:
	@echo "make test          run the test suite"
	@echo "make all           clean-cases -> entities -> measurements -> relations"
	@echo "make relations     extract relations into data/processed/extracao/relations.csv"
	@echo "make umls-relations  filter UMLS MRREL.RRF to this corpus's CUIs (needs the UMLS zip)"
	@echo "make evaluate      score relations against the gold set"
	@echo "make app           launch the Streamlit viewer"
	@echo "make p2            Projeto 2: tabelas -> treinar -> extrair -> comparar"
	@echo "make app-p2        Streamlit viewer for P1 x P2"
	@echo "make p3            Projeto 3: MACCROBAT -> externo -> treinar -> extrair -> figuras"
	@echo "make p3-corpus     corpora do Paciente Virtual (precisa do gazetteer UMLS)"
	@echo "make p3-ida-volta  ida e volta dos pacientes virtuais (precisa do gazetteer UMLS)"
	@echo "make app-p3        Paciente Virtual (Streamlit; não precisa do gazetteer nem de LLM)"
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

# Projeto 3. `make p3` não precisa do gazetteer nem de LLM; os alvos marcados com (UMLS)
# precisam do gazetteer completo (CLINICAL_KG_GAZETTEER aponta para ele, ver paths.py).
p3: p3-maccrobat p3-externo p3-treinar p3-extrair p3-figuras

p3-maccrobat:	; $(PY) -m clinical_kg.maccrobat.preparar --baixar-multicare
p3-externo:	; $(PY) -m clinical_kg.hibrido.pipeline externo
p3-etapas:	; $(PY) -m clinical_kg.hibrido.pipeline etapas          # (UMLS) entidades; asserção e medidas não
p3-treinar:	; $(PY) -m clinical_kg.hibrido.pipeline treinar
p3-extrair:	; $(PY) -m clinical_kg.hibrido.pipeline extrair
p3-figuras:	; $(PY) -m clinical_kg.hibrido.pipeline figuras && $(PY) -m clinical_kg.hibrido.modelo_logico
p3-corpus:                                                         # (UMLS) ~12 min
	$(PY) -m clinical_kg.paciente.corpus construir --nome multicare50
	$(PY) -m clinical_kg.paciente.corpus construir --nome digestivo --recorte digestivo --maximo 2000
	$(PY) -m clinical_kg.paciente.corpus construir --nome maccrobat --fonte maccrobat
p3-ida-volta:	; $(PY) -m clinical_kg.paciente.ida_volta                # (UMLS)
p3-llm-estimar:	; $(PY) -m clinical_kg.llm.pipeline estimar
p3-llm-avaliar:	; $(PY) -m clinical_kg.llm.pipeline avaliar          # só o cache, sem rede

app-p3:
	streamlit run src/clinical_kg/app/projeto3.py
