# PR — E2: Dados preparados e primeira adaptação

**Grupo:** G02 | **Integrantes:** Marilio Cavalcanti de Moura, José Antonio

Documento completo: [`README.md`](README.md)

## Checklist da entrega

- [x] Limpeza, rotulagem e pré-processamento documentados (`README.md` §1, `data/README.md`)
- [x] Partições de treino/teste justificadas: 242 / 58, como no original; protocolo de validação explicado (README §1.5)
- [x] Baseline avaliado nas mesmas condições de teste (B0 e B1: `src/evaluate.py`, `resultados/metricas/`)
- [x] Fine-tuning inicial executado: código, configuração, hiperparâmetros e ambiente (`notebooks/uva_testes_bagas.ipynb`, `resultados/treino_ft0/args.yaml`)
- [x] Evidências de execução: logs, curvas e métricas (`resultados/treino_ft0/`)
- [x] Primeiros resultados, limitações e próximo passo até a E3 registrados (README §4)
- [x] Instruções de execução (README §5)
- [x] Pesos entregues (`weights/best.pt`, `best.onnx`; licenças em `weights/README.md`)
- [x] Registro da origem dos dados (`data/README.md`)

## Resumo dos resultados

Medido em 21/11/2026, IoU ≥ 0,5, `conf = 0,25` e 640 px.

| Método | test_masked · máscara F1 | test_masked · máscara mAP50 | test (58) · caixa F1 | Erro de contagem (test) |
|---|---|---|---|---|
| B0: YOLOv8n-seg COCO sem ajuste | 0,043 | 0,117 | 0,032 | 14,33 |
| B1: regra HSV + morfologia | 0,081 | — | 0,073 | 8,21 |
| FT-0: YOLOv8n-seg ajustado | **0,788** | **0,857** | **0,764** | **2,16** |

## Limitações

- Diâmetro das bagas em px; os mm do notebook usam escala fixa sem calibração.
- Sem partição de validação própria (o `val` do treino é o teste, só para monitorar).
- Modelo treinado uma única vez (semente 0).
- Caminhos fixos no notebook (`E:\Projeto_Uva\...`).
- F1 de máscara abaixo da referência da literatura (0,788 contra ≈ 0,91, protocolo não idêntico).

Não fazem parte do escopo da E3; serão tratadas apenas se houver tempo.

## Próximo passo até a E3

- Primeira versão da aplicação web com o `best.onnx` (upload → máscara → tabela CSV).

Detalhes em README §4.4 e §4.5. Relatório de progresso em `RELATORIO_PROGRESSO.md`. As imagens e labels do WGISD não são versionadas (licença e tamanho); a origem está em `data/README.md`.
