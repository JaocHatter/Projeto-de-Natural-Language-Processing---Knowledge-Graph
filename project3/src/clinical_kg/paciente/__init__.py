"""E6: o Paciente Virtual, um gerador de cenários de treino preso ao grafo de conhecimento.

    corpus.py      constrói o corpus da aplicação (P1 + modelo final do P3) e o carrega
    perfil.py      perfil de cada diagnóstico: o que os pacientes reais que o têm apresentam
    ficha.py       a ficha de um paciente virtual, amostrada de um perfil (com dificuldade)
    narrador.py    a ficha em texto: modelo determinístico ou LLM restrito à ficha
    dialogo.py     o atendimento: perguntas e exames pedidos respondidos só com a ficha
    feedback.py    hipótese do estudante, diagnóstico diferencial tirado do grafo
    ida_volta.py   o texto gerado volta aos extratores: fidelidade e alucinação
"""
