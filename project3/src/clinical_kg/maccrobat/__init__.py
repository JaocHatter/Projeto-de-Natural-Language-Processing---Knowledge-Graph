"""MACCROBAT2020: a segunda fonte de dados do Projeto 3.

200 relatos de caso do PubMed Central anotados à mão no formato brat (Caufield et al., 2019),
com licença CC BY 4.0.

    brat.py         lê o .txt e o .ann de cada documento
    mapeamento.py   converte entidades e relações para o esquema do Projeto 1
    preparar.py     baixa o zip, mapeia tudo e grava as tabelas em data/interim/maccrobat
"""
