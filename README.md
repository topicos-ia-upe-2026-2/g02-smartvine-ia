# SmartVine — Segmentação de cachos e análise de bagas de uva

Projeto aplicado da disciplina **Tópicos Avançados em Inteligência Artificial** — 2026.2

| Grupo | Integrantes |
|---|---|
| SmartVine | Marilio Cavalcanti de Moura (@marilio01), José Antonio (@jos3magalha3s) |

## Cliente, problema e beneficiário

**Cliente e beneficiário:** Parreiral da Serra, Sítio Jussaral, Natuba/PB, propriedade de uvas de mesa das variedades Vitória e Verde. O usuário principal é o produtor ou funcionário de campo; o agrônomo é usuário secundário.

**Problema:** acompanhar o diâmetro (calibre) das bagas ao longo do cultivo. Segundo o relato da equipe, a medição atual é manual, com paquímetro em amostras de cachos, e depende de assistência especializada e deslocamento do técnico. Isso limita a frequência de acompanhamento.

**Evidência inicial:** a equipe relata visita de campo entre 15 e 22/07/2026 e registro de quatro imagens RGB com smartphone Samsung Galaxy A71, em iluminação natural e com oclusões. As imagens não integram esta entrega; seu acesso e autorização de uso serão documentados antes da utilização experimental.

**Benefício esperado:** estimar o diâmetro das bagas visíveis a partir de uma foto de celular, permitindo exportar os resultados para acompanhamento e compartilhamento com o especialista. As estimativas serão em pixels; medidas em milímetros dependerão de referência de tamanho conhecido e calibração adequada. O protótipo não substitui o paquímetro nem a orientação agronômica.

## Solução e escopo

**Entrada:** imagem RGB de um cacho ou fileira. Para saída em mm, a foto deverá conter uma referência de tamanho conhecido, aproximadamente no mesmo plano das bagas, com seu tamanho real informado. Sem referência, a saída será em pixels.

**Processamento:** YOLOv8n-seg ajustado no WGISD para segmentação por instância dos cachos. Dentro de cada máscara, um módulo geométrico aplicará CLAHE, suavização gaussiana, Transformada Circular de Hough, filtros de borda/distância e supressão de sobreposição para localizar bagas visíveis e estimar seus diâmetros.

**Saída:** imagem anotada, estimativas de diâmetro por cacho (média, mínimo, máximo e distribuição) e tabela com identificação, confiança e quantidade de bagas analisadas, exportável em CSV/JSON.

**Fluxo de uso:** envio da foto → pré-processamento → segmentação dos cachos → análise geométrica das bagas → visualização e exportação. As medidas retornarão à escala da imagem original após resize/letterbox.

**Será implementado até a apresentação:** preparação dos dados, baselines, fine-tuning, avaliação, módulo geométrico, exportação ONNX e aplicação web com envio e visualização de imagens. Na E3, está previsto teste de tarefa ou feedback de ao menos um representante do público-alvo.

**Não será implementado:** sensores IoT/ESP32, drones, rastreamento 3D, detecção de doenças, previsão de safra, histórico por talhão/safra ou validação física em grande amostra. A conversão para mm será condicionada à disponibilidade de dados calibrados.

## Artigo-base

- **Título:** Grape detection, segmentation, and tracking using deep neural networks and three-dimensional association.
- **Autores:** Thiago T. Santos, Leonardo L. de Souza, Andreza A. dos Santos e Sandra Avila.
- **Ano:** 2020, *Computers and Electronics in Agriculture*, v. 170, artigo 105247.
- **DOI/URL:** https://doi.org/10.1016/j.compag.2020.105247
- **Componentes utilizados/adaptados:** o artigo compara Mask R-CNN para segmentação por instância com YOLOv2/YOLOv3 para detecção, utilizando transfer learning e métricas por limiar de IoU. O projeto utilizará o WGISD, a formulação de segmentação por instância, o ajuste a partir de pesos COCO e a avaliação de caixa/máscara. Mask R-CNN será substituído por YOLOv8n-seg, mantendo tarefa e estratégia de adaptação. O rastreamento por associação 3D fica fora do escopo; o módulo geométrico de bagas é contribuição própria.

A reutilização deste artigo, publicado em 2020, deverá seguir a aprovação do professor; esta proposta não declara aprovação já concedida.

## Modelo pré-treinado

- **Nome e versão:** YOLOv8n-seg, Ultralytics YOLOv8, variante nano de segmentação; biblioteca prevista `ultralytics` 8.4.x. A versão instalada e o identificador dos pesos serão registrados na E2.
- **Origem dos pesos:** `yolov8n-seg.pt`, disponibilizado pela Ultralytics em https://github.com/ultralytics/assets/releases e obtido pela biblioteca. Documentação: https://docs.ultralytics.com/models/yolov8/
- **Licença:** AGPL-3.0 na distribuição aberta. O uso acadêmico não dispensa atribuição nem cumprimento das condições do código e dos pesos. Dados e outros componentes conservam suas próprias licenças.
- **Tarefa original:** segmentação por instância no COCO, com 80 classes e sem classe uva.
- **Relação com o artigo-base:** mesma tarefa de segmentação por instância e transfer learning COCO → WGISD. A variante nano, com cerca de 3,3 milhões de parâmetros, foi escolhida por viabilidade computacional e suporte à exportação ONNX. Tempo de execução e memória serão medidos, não presumidos.

## Conjunto de dados-base

- **Fonte e acesso:** Embrapa Wine Grape Instance Segmentation Dataset (WGISD), público em https://github.com/thsant/wgisd e https://doi.org/10.5281/zenodo.3361736. Obter as imagens, anotações e listas de partição pelos links da distribuição oficial. Dados completos e pesos não serão versionados.
- **Licença/autorização:** CC BY-NC 4.0, com atribuição e uso não comercial. As quatro fotos de Natuba relatadas pela equipe terão acesso e autorização documentados antes de seu uso.
- **Formato e volume:** 300 imagens JPG, 4.432 cachos com caixas em TXT e máscaras de instância em NPZ para 137 imagens/2.020 cachos. O subconjunto mascarado tem 110 imagens/1.612 cachos em `train_masked` e 27 imagens/408 cachos em `test_masked`. Há um subconjunto de anotações pontuais de bagas com 111 imagens/7.349 pontos.
- **Rótulos e adequação ao problema:** uma classe, cacho, com máscaras por instância. O cenário de vinhedo e iluminação natural é adequado para estudar a segmentação, mas uvas de vinho do WGISD diferem das uvas de mesa de Natuba; a avaliação de campo será exploratória. Pontos de bagas permitem avaliar localização/contagem, não diâmetros.

**Preparação e partições:** converter NPZ para o formato de polígonos YOLO-seg e inspecionar amostras. Dividir somente as 110 imagens de treino em aproximadamente 88 para treino e 22 para validação, com semente fixa, listas registradas e inspeção de duplicatas/imagens relacionadas. As 27 imagens do teste oficial permanecerão isoladas de ajustes de parâmetros. As fotos de Natuba ficarão fora do treino.

> Arquivos de dados grandes **não** são versionados neste repositório.
> **Download do WGISD:** clonar ou baixar https://github.com/thsant/wgisd (pasta `data/` com imagens `.jpg`, anotações `.txt`, máscaras `.npz` e listas `train_masked.txt`/`test_masked.txt`); cópia arquivada em https://doi.org/10.5281/zenodo.3361736. Colocar os arquivos localmente em `data/wgisd/`, que não deve ser enviado ao Git.
> **Pesos:** `yolov8n-seg.pt` é baixado automaticamente pela biblioteca `ultralytics` na primeira execução.

## Plano de adaptação e avaliação

- **O que será ajustado:** fine-tuning completo de backbone, neck e cabeça, com uma classe `grape_cluster`. Configuração inicial: resolução 640×640, até 100 épocas, early stopping de 20 épocas, batch 16–32 conforme memória, AdamW com taxa inicial aproximada de 0,002 e weight decay 5e-4. Aplicar aumentos de cor, escala, translação, flip e mosaic, desativando mosaic nas dez épocas finais. A meta é realizar cinco sementes e apresentar média, desvio-padrão e IC95%, conforme recursos disponíveis. Congelamento do backbone será ablação opcional.
- **Recursos computacionais:** GPU prevista NVIDIA RTX 4000 Ada (20 GB), com Colab/T4 como alternativa. Ambiente previsto Python 3.10, PyTorch 2.5, Ultralytics, OpenCV e ONNX Runtime. Tempos e memória serão medidos na E2; inferência da aplicação será avaliada em CPU.
- **Baseline:** B0, YOLOv8n-seg COCO sem ajuste, tratando predições como candidatos independentemente do rótulo, com filtros definidos na validação; B1, limiarização HSV, morfologia e componentes conexos. B0 é uma referência diagnóstica limitada, pois COCO não contém a classe uva; supressão sem distinguir classes não cria novas classes. Todos serão avaliados nas mesmas imagens e condições de teste. Resultados do artigo serão apenas referências de literatura, não comparação direta.
- **Métricas:** precision, recall, F1 em IoU 0,5, mAP@50 e mAP@50–95 para caixas/máscaras; precision, recall e F1 por casamento um a um com pontos anotados e MAE de contagem para bagas. Para diâmetro, erro médio absoluto e viés em pixels contra anotações manuais em subconjunto independente; medidas em mm serão avaliadas separadamente, apenas com referência física/calibração. Registrar latência em CPU, tempo total por imagem e tamanho do modelo. Limiares e parâmetros serão definidos na validação e mantidos fixos no teste.
- **Riscos:** poucos dados e sobreajuste (aumentos, early stopping e repetição); mudança de domínio (teste de campo exploratório); oclusão de bagas (medir apenas bagas visíveis e analisar falhas); perspectiva e escala (referência conhecida ou saída em pixels); indisponibilidade da GPU (Colab); incompatibilidade/licenças (verificar ambiente e cumprir as condições de cada recurso).
- **Divisão das atividades entre os integrantes:** Marilio — preparação e partições, fine-tuning e análise experimental; José Antonio — baselines, módulo geométrico, aplicação e exportação ONNX. Ambos revisarão código, documentação e apresentação. Trata-se de divisão planejada, não declaração de atividades já executadas.

**Marcos:** E1, proposta (11/11/2026); E2, dados e modelo (PR 22/11, alinhamento 23/11); E3, aplicação e teste com público-alvo (PR 29/11, alinhamento 30/11); E4, apresentação (07/12). A E1 não apresenta resultados experimentais próprios.

## Como executar

Nesta E1, a entrega é documental: não há scripts de treinamento ou aplicação implementados. As dependências e os comandos executáveis serão adicionados na E2/E3. O roteiro abaixo é o modelo do template para essas etapas, não um procedimento de execução já validado.

```bash
# 1. Criar ambiente
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (Se aplicável) baixar dados e pesos conforme as seções acima

# 3. Executar
# python src/train.py ...
# python src/app.py ...
```

## Resultados

Não há resultados experimentais próprios nesta E1. Baselines, configurações, curvas, métricas e exemplos de sucesso/falha serão registrados a partir da E2 em [docs/experimentos.md](docs/experimentos.md). O relatório técnico seguirá [docs/relatorio.md](docs/relatorio.md), e mudanças relevantes serão registradas em [docs/decisoes.md](docs/decisoes.md).

## Limitações e uso responsável

O protótipo estima apenas bagas visíveis e pode falhar com oclusão, iluminação, perspectiva e variedades diferentes das usadas no treinamento. Não substitui medição física, diagnóstico ou recomendação agronômica. Não fornecerá medidas físicas confiáveis sem calibração adequada.

Imagens de campo serão utilizadas somente com autorização, evitando dados pessoais e informações sensíveis desnecessárias. Uso dos dados WGISD será acadêmico e não comercial, com atribuição. Código e pesos respeitarão suas respectivas condições de licença. Credenciais, dados completos e pesos não serão adicionados ao Git; recursos externos terão links e instruções de obtenção. Custo, latência e memória serão reportados após medição.

## Referências

SANTOS, T. T.; SOUZA, L. L.; SANTOS, A. A.; AVILA, S. Grape detection, segmentation, and tracking using deep neural networks and three-dimensional association. *Computers and Electronics in Agriculture*, v. 170, 105247, 2020. DOI: https://doi.org/10.1016/j.compag.2020.105247.

WGISD. *Embrapa Wine Grape Instance Segmentation Dataset*. Disponível em: https://github.com/thsant/wgisd. DOI: https://doi.org/10.5281/zenodo.3361736. Acesso em: 13 nov. 2026.

ULTRALYTICS. *YOLOv8: documentação e modelos de segmentação*. Disponível em: https://docs.ultralytics.com/models/yolov8/. Pesos: https://github.com/ultralytics/assets/releases. Acesso em: 13 nov. 2026.
