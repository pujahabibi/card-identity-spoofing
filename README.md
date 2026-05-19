# Card Identity Spoofing

Image-classification pipeline for detecting spoofed Indonesian identity cards (KTP). The model is fine-tuned to distinguish genuine card crops from spoofed/forged ones, and is served behind a FastAPI app for inference.

## Overview

- **Task:** binary/multi-class image classification on cropped ID-card images.
- **Base model:** `ktp-crop-clip` (a CLIP-style backbone pre-trained on KTP crops), fine-tuned via Hugging Face `transformers`.
- **Dataset format:** `imagefolder` layout under `archive/crop_data/` with `train/` and `validation/` splits; class labels are inferred from sub-folder names.
- **Output:** a fine-tuned image classifier pushed to the Hugging Face Hub and served via the FastAPI endpoint in `main.py`.

## Project structure

```
.
├── main.py     # FastAPI entry point (uvicorn server on port 5000)
├── train.py    # Fine-tuning script for the image classifier
└── archive/
    └── crop_data/
        ├── train/<class_name>/*.jpg
        └── validation/<class_name>/*.jpg
```

## Training

`train.py` performs the following:

1. Loads the `imagefolder` dataset from `archive/crop_data`.
2. Applies image preprocessing (resize to 224×224, tensor conversion, normalization using the base processor's mean/std).
3. Loads `AutoImageProcessor` and `AutoModelForImageClassification` from the `ktp-crop-clip` checkpoint, with label mappings derived from the dataset.
4. Fine-tunes with the Hugging Face `Trainer`:
   - 250 epochs, batch size 16, gradient accumulation 4
   - Learning rate 5e-5, warmup ratio 0.1
   - Evaluation/saving every 20 steps, accuracy as the best-model metric
   - `push_to_hub=True` (model artifacts uploaded to the Hub)

Run:

```bash
python train.py
```

## Inference / API

`main.py` boots a FastAPI app (imported from `app.controller.api`) with Uvicorn:

```bash
python main.py
# → http://0.0.0.0:5000
```

## Requirements

Core dependencies:

- `transformers`
- `datasets`
- `torch`, `torchvision`
- `numpy`
- `fastapi`, `uvicorn`

## Security note

Hugging Face tokens must **never** be hardcoded in source files. Authenticate via the CLI or an environment variable instead:

```bash
huggingface-cli login
# or
export HF_TOKEN=...
```
