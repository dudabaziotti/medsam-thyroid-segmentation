# Segmentação de tireoide em ultrassom com MedSAM (assistida e automática)

Código do trabalho **"Comparing Assisted and Fully Automatic Thyroid Segmentation Using Fine-Tuned MedSAM with YOLO Detection"**.

O pipeline segmenta **nódulos** (dataset TN3K) e a **glândula tireoide** (dataset TG3K) em imagens de ultrassom usando o [MedSAM](https://github.com/bowang-lab/MedSAM), em dois modos:

| Modo | Origem da bounding box (prompt) | Segmentação |
|---|---|---|
| **Assistido** | Caixa extraída da máscara do gabarito | MedSAM com decoder ajustado |
| **Automático** | Caixa prevista por um detector YOLOv8n | MedSAM com decoder ajustado |

O fine-tuning é feito **apenas no mask decoder** do MedSAM (image encoder e prompt encoder congelados), com os embeddings das imagens pré-calculados para acelerar o treino. Um decoder é treinado para nódulo e outro para glândula.

---

## Estrutura do repositório

```
.
├── pipeline_medsam_tirads.ipynb   # pipeline completo (só código)
├── requirements.txt
└── README.md
```

O notebook está dividido nas mesmas seções descritas abaixo; cada seção começa com um comentário `# N. Nome` para facilitar a navegação.

---

## Como executar

O notebook foi feito para o **Google Colab com GPU**, lendo dados e pesos do Google Drive.

### 1. Organize o Google Drive

```
MyDrive/IC_TIRADS/
├── checkpoints/
│   └── medsam_vit_b.pth
└── data_public/ThyroidDataset/Thyroid Dataset/
    ├── tn3k/   (trainval-image, trainval-mask, test-image, test-mask)
    └── tg3k/   (thyroid-image, thyroid-mask)
```

- Pesos do MedSAM (`medsam_vit_b.pth`): https://github.com/bowang-lab/MedSAM
- Datasets TN3K e TG3K: https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation

### 2. Ajuste a configuração (seção 2)

As flags principais:

| Variável | Padrão | Efeito |
|---|---|---|
| `TREINAR_MEDSAM` | `False` | `True` treina os decoders do zero; `False` carrega os decoders já salvos em `checkpoints/` |
| `TREINAR_YOLO` | `False` | `True` treina os detectores; `False` carrega `checkpoints/yolo_*_detector/weights/best.pt` |

Na primeira execução (sem checkpoints treinados), coloque as duas como `True`.

### 3. Execute todas as células

As dependências são instaladas na seção 1 (`ultralytics`, `opencv-python-headless` e `segment-anything`). Para rodar fora do Colab, use o `requirements.txt` e ajuste os caminhos em `BASE_DIR`.

---

## Explicação do pipeline

### 1. Setup
Monta o Google Drive, instala as dependências, importa as bibliotecas e seleciona o dispositivo (`cuda` se disponível).

### 2. Configuração
Define caminhos, nomes dos checkpoints, semente (`SEED = 42`) e hiperparâmetros:

| Parâmetro | Valor | Descrição |
|---|---|---|
| `IMG_SIZE` | 1024 | Resolução de entrada do MedSAM |
| `BOX_JITTER` | 10 px | Folga aleatória da caixa no treino (0 na avaliação) |
| `BATCH_SIZE` | 4 | Batch do treino do decoder |
| `NUM_EPOCHS` | 20 | Épocas do treino do decoder |
| `LR` | 1e-4 | Learning rate (AdamW) |
| `LIMIAR` | 0.5 | Limiar de binarização da máscara prevista |
| `YOLO_EPOCHS` | 30 | Épocas do YOLO |
| `YOLO_IMGSZ` | 640 | Resolução de entrada do YOLO |
| `YOLO_BATCH` | 16 | Batch do YOLO |
| `YOLO_CONF` | 0.25 | Confiança mínima da detecção |

Os embeddings e o dataset do YOLO ficam em `/content/work` (disco local do Colab), para não encher o Drive.

### 3. Dados e splits
`indexar_pares` lista os pares imagem/máscara com o mesmo nome de arquivo.

- **TN3K:** usa o split oficial (trainval / test); 15% do trainval vira validação → 2447 treino / 432 val / 614 teste.
- **TG3K:** não tem split oficial; divisão aleatória 70/15/15 por imagem → 2509 treino / 538 val / 538 teste.

### 4. Datasets PyTorch
- Imagem redimensionada para 1024×1024 (bilinear), em RGB, normalizada para [0, 1]; máscara redimensionada por vizinho mais próximo.
- `caixa_da_mascara` gera o prompt `[x0, y0, x1, y1]` a partir da máscara, com folga aleatória de até `BOX_JITTER` px no treino.
- `ThyroidSAMDataset` carrega a imagem (usado na avaliação e no cálculo dos embeddings).
- `ThyroidEmbeddingDataset` carrega o embedding pré-calculado em vez da imagem (usado no treino).

### 5. Modelo MedSAM
Carrega o MedSAM ViT-B, congela image encoder e prompt encoder e deixa só o mask decoder treinável (~4,1 M de ~93,7 M parâmetros, 4,3%). Funções auxiliares:

- `resetar_decoder` — restaura o decoder original do MedSAM.
- `carregar_decoder` — carrega um decoder treinado de `checkpoints/`.
- `prever_mascara` — segmenta uma imagem a partir de uma caixa.

### 6. Treino do decoder
1. `precalcular_embeddings` passa cada imagem de treino pelo image encoder uma única vez e salva o resultado em disco (retoma de onde parou se o Colab cair).
2. `treinar_uma_epoca` aplica prompt encoder + mask decoder sobre os embeddings, com loss **Dice + BCE**.
3. `treinar_decoder` usa AdamW e salva o checkpoint a cada época; o modelo final é o da última época.

O decoder de **nódulo** (TN3K) e o de **glândula** (TG3K) são treinados de forma independente, ambos partindo do decoder original do MedSAM.

### 7. Métricas
Calculadas por imagem, na resolução 1024×1024:

- Dice, IoU, precisão, recall (sensibilidade), especificidade e acurácia por pixel.
- `hd95` — percentil 95 das distâncias simétricas entre os contornos (em pixels).
- `hd_max` — distância de Hausdorff máxima.

Os contornos são obtidos por erosão binária. Se a predição ou o gabarito não tiverem contorno, as distâncias ficam `NaN`. A função `avaliar` usa as caixas do gabarito (modo assistido) ou caixas externas passadas em `boxes` (modo automático); `resumir` imprime média, desvio padrão e mediana.

### 8. Avaliação assistida
Avalia cada decoder no conjunto de teste usando a caixa do gabarito, sem folga. Salva `metrics_nodulo_full.csv` e `metrics_gland_full.csv` e lista os piores casos de HD95 da glândula.

### 9. Detector YOLO
- `mascara_para_bbox_yolo` converte a máscara em caixa no formato YOLO (cx, cy, w, h normalizados).
- `preparar_dataset_yolo` monta a estrutura `images/` + `labels/` e o `dataset.yaml`, reaproveitando os mesmos splits do MedSAM.
- `treinar_yolo` faz fine-tuning do YOLOv8n (pré-treinado no COCO), com um modelo de uma classe para nódulo e outro para glândula.

### 10. Avaliação automática (YOLO + MedSAM)
`detectar_caixas` usa a detecção de maior confiança de cada imagem e reescala a caixa para 1024×1024. Sem detecção, a caixa vira a imagem inteira (*fallback*). `avaliar_automatico` reporta as métricas para todas as imagens, só as detectadas e só as não detectadas, e salva `metrics_nodulo_automatico.csv` e `metrics_gland_automatico.csv`.

### 11. Artefato de anotação no TG3K
Parte das máscaras do TG3K inclui, por engano, o texto exibido no canto superior esquerdo da tela do ultrassom. `tem_artefato` marca uma máscara quando há um componente conectado pequeno (10–2000 px), diferente do maior componente, inteiramente dentro dos 100×100 px do canto superior esquerdo. Em seguida, alguns casos marcados são exibidos para inspeção visual.

---

## Referências

- MedSAM: https://github.com/bowang-lab/MedSAM
- Segment Anything: https://github.com/facebookresearch/segment-anything
- TN3K / TG3K (TRFE-Net): https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics