# Relatório de progresso — E2 (SmartVine, G02)

**Equipe:** Marilio Cavalcanti de Moura · José Antonio  
**Alinhamento com o professor:** 23/11/2026

Documento completo: [`README.md`](README.md).

## 1. Andamento

| Etapa | Situação |
|---|---|
| Dados (WGISD, 300 imagens, partições 242 treino / 58 teste) | concluído |
| Baselines B0 (COCO sem ajuste) e B1 (regra HSV) | concluído |
| Fine-tuning do YOLOv8n-seg (50 épocas, pesos `.pt` e `.onnx`) | concluído |
| Módulo de bagas | concluído |
| Aplicação web com o `best.onnx` | em andamento, para a E3 (29/11/2026) |

## 2. Resultados (medidos em 21/11/2026)

| Método | F1 máscara (test_masked) | mAP50 máscara (test_masked) | F1 caixa (test) | Erro de contagem (cachos/img) |
|---|---|---|---|---|
| B0: COCO sem ajuste | 0,043 | 0,117 | 0,032 | 14,33 |
| B1: regra HSV | 0,081 | — | 0,073 | 8,21 |
| **FT-0: YOLOv8n-seg ajustado** | **0,788** | **0,857** | **0,764** | **2,16** |

O modelo ajustado supera os dois baselines em todas as métricas e roda em 1 ms por imagem na GPU.

## 3. Próximo passo

Aplicação web com o `best.onnx`: o usuário envia a imagem, recebe a máscara dos cachos e a tabela CSV com as medidas.
