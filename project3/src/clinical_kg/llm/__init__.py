"""LLM no Projeto 3: provedor configurável, cache de respostas e o anotador de relações (E3).

    provedor.py   quem responde: cache (sem rede), endpoint compatível com OpenAI (API ou Ollama
                  local) ou um provedor falso para os testes
    cache.py      toda resposta fica em data/interim/cache/llm_cache.jsonl (chave = hash do pedido)
    anotador.py   prompt por par candidato, validação da resposta por regras e autoconsistência
    fontes.py     os votos do LLM como fonte de atributos do classificador do P2 (P3-C) e as
                  variantes P3-A, P3-E e P3-F

Nada aqui chama a rede por conta própria: sem `CLINICAL_KG_LLM_URL` configurada, o provedor
padrão só lê o cache, e um pedido fora do cache levanta `RespostaAusente`. Assim o caderno e os
testes rodam sem chave de API.
"""
