# Pesos do modelo ajustado (E2)

| Arquivo | Tamanho | SHA-256 |
|---|---|---|
| `best.pt` | 6,8 MB | `80b9dc0471ad6f9490ab0d98bd891e0e3777418f8fdce9e0f8af73a681bf1479` |
| `best.onnx` | 13,4 MB (opset 19, eixos dinâmicos) | `3fed83ad7284d9cf11876fff917ae4384c9f092cb6c7ca412137281dd67e0aa5` |

- **Modelo:** YOLOv8n-seg (Ultralytics 8.4), fine-tuning completo a partir de `yolov8n-seg.pt` (COCO), com 1 classe.
- **Origem:** gerado pelo notebook [`../notebooks/uva_testes_bagas.ipynb`](../notebooks/uva_testes_bagas.ipynb) (pasta `wgisd_treino_baga/weights/`). O `best.onnx` foi exportado na última célula do mesmo notebook.
- **Treino:** [`../resultados/treino_ft0/args.yaml`](../resultados/treino_ft0/args.yaml), 50 épocas; checkpoint = época 50.

## Uso

```python
from ultralytics import YOLO
model = YOLO("weights/best.pt")
results = model.predict("imagem.jpg", imgsz=640, conf=0.25, retina_masks=True)
```

## Licenças

- Derivado de `yolov8n-seg.pt`: **AGPL-3.0** (Ultralytics). A redistribuição é permitida desde que o código continue aberto.
- Treinado com o WGISD (**CC BY-NC 4.0**): uso **não comercial**, com atribuição a Santos et al. (2019/2020).


