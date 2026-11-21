# Projeto 3 — plano interno da equipe

Este arquivo é só para a equipe: saiu do README para que ele fique pronto para a entrega. Aqui ficam o
estado do trabalho, o que falta e como rodar as variantes com LLM.

**Estado (21/11/2026):** tudo o que não depende de chamar um LLM já está implementado, medido e testado
(E0, avaliação externa, P3-D, P3-G, grafo do P3, Paciente Virtual e ida e volta). As variantes com LLM (P3-A, B, C,
E, F) estão implementadas e testadas com um provedor falso; falta rodá-las com um modelo (ver abaixo).
As seções marcadas *(a preencher)* no README esperam esses números.

## Plano de Trabalho

**Prioridades.** O núcleo garante a entrega; o resto é ganho.

| Nível | Entregas |
|---|---|
| Núcleo | E0 e avaliação externa do P1/P2 no MACCROBAT; E3 com cache e as variantes P3-A, P3-C e P3-F na tabela do P2; paciente virtual v1 (ficha → narrativa → hipótese → feedback); ida e volta com P1, P2 e P3 |
| Extensão | E2 (ligação com SapBERT); E1 (quebra de sentenças); variantes P3-B, P3-D, P3-E e P3-G |
| Ousado | ordem temporal no grafo e linha do tempo do paciente; teste de estresse por paráfrase (o LLM reescreve o paciente com sinônimos e abreviações, e medimos quanto cada extrator perde); pacientes virtuais como dados de treino extra do P2 (ajudam no gold real?) |

**Situação em 21/11/2026:**

| Entrega | Situação |
|---|---|
| E0 (MACCROBAT no esquema do P1) e avaliação externa (relações, entidades, asserção, medidas) | feito |
| P3-D (embeddings) e P3-G (MACCROBAT) na tabela do P2 | feito |
| Grafo do P3 nos 50 casos e comparação P1 × P2 × P3 | feito |
| Asserção com coordenação (achada pela ida e volta, validada no MACCROBAT) | feito |
| Paciente virtual v1 (corpus de 1.920 casos de MultiCaRe + MACCROBAT, ficha, diálogo, feedback, subgrafo) e ida e volta | feito, narrador determinístico |
| E3 + P3-A, B, C, E, F: código, prompts, validação, cache, testes com provedor falso | feito; **falta rodar com um LLM** |
| Narrador LLM e ida e volta com ele (`ida_volta --llm`) | código feito; falta rodar |
| E1 (quebra de sentenças) e E2 (SapBERT) | não começado |
| Caderno `projeto3.ipynb` e figura do modelo lógico | feito |
| Slides | com a outra parte da equipe |

**Para rodar as variantes com LLM** (qualquer membro com acesso a um modelo; sem pagar, com o Ollama):

```bash
ollama pull qwen2.5:7b                                  # ou outro modelo instruído
export CLINICAL_KG_LLM_URL=http://localhost:11434/v1 CLINICAL_KG_LLM_MODELO=qwen2.5:7b
python -m clinical_kg.llm.pipeline anotar --limite 20    # teste rápido
python -m clinical_kg.llm.pipeline anotar --modo zero    # P3-A, C, E (413 chamadas)
python -m clinical_kg.llm.pipeline anotar --modo few     # P3-B
python -m clinical_kg.llm.pipeline anotar --conjunto fracos   # P3-F (2.391 chamadas)
python -m clinical_kg.llm.pipeline avaliar               # tabela, só com o cache
```

Depois, versionar `data/interim/cache/llm_cache.jsonl`: com ele, qualquer um refaz os números sem chamar o modelo.

**Frentes sugeridas** (uma por pessoa):

1. MACCROBAT: leitor brat, mapeamento e avaliação externa (E0).
2. LLM anotador: prompts, cache, autoconsistência e as variantes P3-A a P3-G (E3 e E4).
3. Entidades e embeddings: ligação ao UMLS com SapBERT e quebra de sentenças (E1 e E2).
4. Aplicação Paciente Virtual (E6).
5. Avaliação de ida e volta, figuras, README e slides.

**Decisões pendentes:**

* **Provedor e modelo do LLM** (API ou local): custo, velocidade e reprodutibilidade. Ordem de grandeza: o P3-F com
  autoconsistência (*k* = 5) sobre os 2.391 candidatos dá cerca de 12 mil chamadas; com o cache, cada uma acontece
  uma vez só.
* ~~**Especialidade e tamanho do recorte do MultiCaRe**~~: adotado o recorte digestivo inteiro (1.674 casos),
  configurável em `paciente/corpus.py` (`RECORTES`). Trocar é uma linha e ~12 min de `make p3-corpus`.
* **Estrutura completa do Cookiecutter Data Science**, que o template conta como diferencial.

**Riscos e como os tratamos:**

* **O gold foi construído com apoio de um LLM** (ver README do P2): avaliar LLMs nele pode favorecê-los. O
  MACCROBAT, anotado só por humanos, é a avaliação de controle.
* **Alucinação:** evidência literal obrigatória, tipos restritos pelas categorias das entidades e ancoragem no UMLS.
* **Custo e tempo:** cache, cascata (P3-E) e LLM só onde ele agrega (gold, MACCROBAT e pacientes gerados); os
  extratores baratos escalam para o recorte da aplicação.
* **Grafo esparso:** daí o recorte maior do MultiCaRe para a aplicação (ver E5).
* **Contaminação:** textos do PubMed Central podem estar no pré-treino do LLM; fica como limitação declarada.
