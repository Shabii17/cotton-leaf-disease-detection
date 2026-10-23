# Cotton Leaf Disease Detection

Final Year Project, BS Information Technology, The Islamia University of Bahawalpur
**Student:** Muhammad Shoaib Sajjad (F23BINFT1M01137) | **Supervisor:** Dr Munawar

A web app where a farmer uploads or captures a photo of a cotton leaf and receives the likely disease,
a confidence score and basic treatment advice in English or Urdu. Two transfer-learning models
(MobileNetV2 and EfficientNet-B0) are trained and compared, and the better one is used by the app.

| Crop | Classes |
| --- | --- |
| Cotton | bacterial blight, leaf curl virus, healthy |

> **Scope note:** the project scope is cotton only (agreed with the supervisor). The starter code was written for
> cotton and wheat, so `src/config.py`, `app/advice.json` and `scripts/make_synthetic_dataset.py` still list the wheat
> classes until they are removed (see "Changing the classes" below). Train only on cotton images.

## Features

- Upload a JPG or PNG (max 10 MB) or take a photo with the phone camera
- Crop and disease name, confidence percentage and top three results
- Warning when the model is not sure (below 70%) or the image is not a supported leaf (below 40%)
- Symptoms, treatment and prevention advice from an editable file (`app/advice.json`)
- English and Urdu interface, works on a phone screen
- "This result is wrong" button saves the photo for review; photos are not stored otherwise

## Project structure

| Path | Purpose |
| --- | --- |
| `src/` | data_prep, train, evaluate, compare, predict, preprocess, models, feedback, config |
| `app/app.py`, `app/advice.json` | Streamlit web app and its advice text |
| `scripts/` | import_folder.py (import a dataset), make_synthetic_dataset.py (fake images to test the code) |
| `notebooks/train_on_colab.ipynb` | Training on a free Google Colab GPU |
| `tests/` | Automated tests (pytest) |
| `docs/` | Proposal, SRS, SDD and other project documents |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## How to run

1. Get data: download public cotton leaf datasets (check each licence) and import them with `scripts/import_folder.py`.
2. Collect cotton field photos into `data/field_test/<class>/` (never used for training).
3. Prepare data: `python -m src.data_prep --overwrite`
4. Train both models (use Google Colab for a GPU):
```bash
   python -m src.train --model mobilenet_v2 --epochs 20
   python -m src.train --model efficientnet_b0 --epochs 20
```
5. Evaluate and choose the best model:
```bash
   python -m src.evaluate --checkpoint models/mobilenet_v2_best.pt
   python -m src.evaluate --checkpoint models/efficientnet_b0_best.pt
   python -m src.compare
```
6. Run the app: `streamlit run app/app.py`

## Results

To be filled in after training on real data.

| Model | Public test accuracy | Field-photo accuracy |
| --- | --- | --- |
| MobileNetV2 | | |
| EfficientNet-B0 | | |

No real accuracy has been measured yet.

## Limitations

- Decision support only; it does not replace an agriculture expert.
- Treatment advice is draft text and must be checked by an agriculture expert.
- The Urdu text needs review by a native speaker.
- Accuracy on field photos is expected to be lower than on public images.

## Documents

Project documents are in the `docs/` folder: proposal, SRS and SDD.
