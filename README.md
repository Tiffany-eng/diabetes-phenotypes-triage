# Comorbidity Phenotype Discovery and Triage-Time Classification in Diabetic Inpatients

**Machine Learning module, Summative Project** | Author: Emma Tiffany Umwari (BSc. Software Engineering, African Leadership University, Kigali)

Unsupervised discovery of comorbidity phenotypes (K-Means) followed by supervised prediction of the phenotype from information available at triage, comparing **traditional ML (scikit-learn)** with **deep learning (TensorFlow/Keras, `tf.data`)** on the Diabetes 130-US Hospitals dataset.

## Project links

| Deliverable | Link |
|---|---|
| Written report (IEEE style) | [REPORT_LINK](https://docs.google.com/document/d/1iTD-sdX6thMy6rwgb-Svh3_-9ut3xqP0MEoisUI4wEE/edit?usp=sharing) |
| Google Drive folder (raw data zip, best models; anyone with the link can view) |[DRIVE_URL](https://drive.google.com/drive/folders/1bfA3Yi49PF869gY2FEPUk83fxKSgJ5GB?usp=drive_link) |
| Presentation video (5-10 min) | [VIDEO_URL](https://docs.google.com/document/d/1wKSKST0iY72LQ5t6KAo7PjP_n3S48l4eLVb-_1APWGk/edit?usp=sharing) |

## Problem and approach

Diabetic inpatients are not one homogeneous group. If a hospital could tell on arrival which kind of patient is coming, it could plan beds and staff earlier.

1. **Phase 1 (unsupervised).** Cluster 71,515 first encounters on 26 retrospective clinical-burden features (diagnosis groups, lab and procedure counts, medications, length of stay, prior use) into 4 phenotypes (K-Means after PCA; stability ARI 0.951 +/- 0.080).
2. **Phase 2 (supervised, 4-class).** Predict the phenotype from 12 triage-time features. An **information-ladder** experiment shows why the clustering inputs cannot be used (the task would be circular).

## Headline results (held-out test set, 10,728 patients)

| Model | Macro ROC-AUC (95% CI) | Balanced accuracy | Macro F1 |
|---|---|---|---|
| Tuned Hist. Gradient Boosting (best traditional) | 0.7316 (0.724-0.739) | 0.524 | 0.529 |
| Sequential MLP, 1 hidden layer (best deep learning) | 0.7322 (0.725-0.740) | 0.518 | 0.533 |

Triage-only features give a validation macro ROC-AUC of 0.672; adding the primary-diagnosis group gives 0.728; the circular oracle gives 0.9996 and is not a reportable result. The two families are statistically tied, so performance is limited by the information available at triage, not by the model. All 32 experiments (E01-E32) are in `results/experiment_log.csv`.

## Repository structure

```
.
├── ml-summative-diabetes-phenotypes.ipynb   # the full, documented pipeline (run top to bottom)
├── requirements.txt                         # dependencies
├── results/                                 # experiment_log.csv (E01-E32), validation/test metrics, bootstrap CIs, phenotype profiles
├── logs/tensorboard/                        # one folder per experiment; folder name starts with the exp_id
├── docs/figures/                            # figures used in the report
├── data/README.md                           # how to obtain the data (the data itself is NOT in this repo)
└── LICENSE
```

Trained models (`.joblib`, `.keras`, `.h5`, SavedModel) and the raw dataset are in the Google Drive folder, not in this repository.

## How to run

**Kaggle (what the results were produced on).** Create a notebook, upload `ml-summative-diabetes-phenotypes.ipynb`, turn Internet on (or attach a Kaggle dataset containing `diabetic_data.csv`) and run all cells. About 30-60 minutes on CPU.

**Local or Colab**
```bash
git clone [GITHUB_URL](https://github.com/Tiffany-eng/diabetes-phenotypes-triage)
cd diabetes-phenotypes-triage
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook ml-summative-diabetes-phenotypes.ipynb
```
Set `FAST_DEBUG = True` in the configuration cell first for a quick end-to-end check with tiny budgets.

**Data.** The notebook loads, in order: a cached local copy, an attached Kaggle input, or a download from UCI (ID 296). See `data/README.md`.

## Reproducibility

* Seed 42 for Python, NumPy, scikit-learn and TensorFlow.
* Patient-level `GroupShuffleSplit` (70/15/15) before anything is fitted; imputers, scalers, PCA, clustering and every model are fitted on the training split only; the test set is used once.
* Random search is not bit-for-bit repeatable across runs and machines (parallel jobs, library versions), so re-running may select slightly different hyperparameters. The numbers in the report come from the logged run in `results/` and `logs/tensorboard/`.

## Experiment tracking

Every run has an id (E01-E32) in `results/experiment_log.csv` and a TensorBoard folder in `logs/tensorboard/` whose name starts with the same id.

```bash
tensorboard --logdir logs/tensorboard
```

| Group | Runs |
|---|---|
| Information ladder | E01-E03 |
| Baselines | E04-E07 |
| Class weighting | E08-E09 |
| Feature engineering | E10-E13 |
| Tuning (random search, 10 candidates x 3 folds) | E14-E16 |
| Deep learning: Sequential MLP (8 trials), Functional embeddings (6 trials) | E17-E30 |
| Final test evaluation | E31-E32 |

## Dataset and licence

Strack, B. et al., "Diabetes 130-US Hospitals for Years 1999-2008", UCI Machine Learning Repository, 2014, https://doi.org/10.24432/C5230J. Check the licence on the dataset page before redistributing. Code is released under the MIT licence (see `LICENSE`).

## Limitations

US hospitals, 1999-2008; phenotypes are labels from a clustering step and depend on features, algorithm and K; the primary-diagnosis feature may not be known at the moment of triage; no external validation yet. See the report for the full discussion.
