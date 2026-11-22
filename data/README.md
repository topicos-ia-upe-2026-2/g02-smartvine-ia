# Dados

## Origem

| Item | Valor |
|---|---|
| Conjunto | Embrapa Wine Grape Instance Segmentation Dataset (WGISD) |
| Autores | Santos, T. T.; de Souza, L. L.; dos Santos, A. A.; Avila, S. |
| Repositório | https://github.com/thsant/wgisd |
| DOI | [10.5281/zenodo.3361736](https://doi.org/10.5281/zenodo.3361736) |
| Licença | CC BY-NC 4.0 (uso não comercial, com atribuição) |

## Conteúdo desta pasta

| Item | Conteúdo |
|---|---|
| [`imagens/`](imagens/) | só o README: origem das 300 imagens `.jpg` do WGISD (os arquivos não são versionados) |
| [`labels/`](labels/) | só o README: formato e origem das labels YOLO-seg (os arquivos não são versionados) |
| `b1_ajuste.txt` | 23 imagens de `train.txt` usadas só para escolher os parâmetros do baseline B1 |

As imagens e labels não estão no repositório por causa do tamanho e da licença. Para reproduzir, baixe o WGISD:

```bash
git clone --depth 1 https://github.com/thsant/wgisd.git ../wgisd
```

## Limpeza e rotulagem

As imagens e as labels vieram prontas do repositório. Não foi necessária limpeza (o notebook só confere se as 300 imagens listadas existem), e nenhuma anotação foi feita ou alterada pela equipe.

## Partições

As partições originais do repositório, como no artigo:

| Partição | Imagens | Arquivo | Uso |
|---|---|---|---|
| Treino | 242 | `train.txt` | treino do FT-0; 23 delas (`b1_ajuste.txt`) ajustam o B1 |
| Teste | 58 | `test.txt` | avaliação por caixa |
| Teste com máscara | 27 (subconjunto de `test.txt`) | `test_masked.txt` | avaliação por máscara |

Não há partição de validação própria: no treino, o `val` do Ultralytics aponta para o teste e serve só para monitoramento (o checkpoint é a última época). Justificativa no [`README` principal, §1.5](../README.md#15-partições).
