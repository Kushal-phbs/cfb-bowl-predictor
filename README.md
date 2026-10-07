# College Football Bowl Game Predictor

UE24CS352A – Machine Learning · Mini-Project (Problem #101)
Team: `<Name 1 – SRN>`, `<Name 2 – SRN>`

Predicts the winner of college football bowl games (and a confidence value for "Bowl Pick'em") from regular-season box-score statistics.
The model predicts the **score margin**: its sign is the pick and its magnitude is the confidence.

## Project structure

```
cfb-bowl-predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── src/                                     # reusable code imported by the notebooks
│   ├── config.py                            # paths + settings (seed, primary model, ...)
│   ├── features.py                          # cleaning, team-season features, bowl matchups
│   ├── models.py                            # models, leave-one-season-out CV, metrics
│   └── demo.py                              # predict_matchup() and pickem_sheet()
├── data/
│   ├── raw/cfb_box_scores_2000_2010.csv     # original dataset (7,531 games)
│   └── processed/                           # written by notebook 01
├── notebooks/                               # run in this order
│   ├── 01_data_audit_and_features.ipynb     # audit → cleaning → features → matchups
│   ├── 02_models_and_cv.ipynb               # models → LOSO CV → results & confidence analysis
│   └── 03_final_model_and_demo.ipynb        # final model, live demo, conclusions
├── reports/
│   ├── figures/                             # plots used in the write-up / slides
│   └── results/                             # model_comparison.csv, cv_predictions.csv
└── docs/                                    # put the 2-page PDF write-up and slide deck here
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Run the three notebooks **in order** (each one reads the files written by the previous one):

```bash
jupyter notebook notebooks/
```

| # | Notebook | Reads | Writes | Time |
|---|---|---|---|---|
| 1 | `01_data_audit_and_features.ipynb` | `data/raw/` | `data/processed/*.csv`, figures 01–02 | seconds |
| 2 | `02_models_and_cv.ipynb` | `data/processed/bowl_matchups.csv` | `reports/results/*.csv`, figures 03–05 | ~1 min |
| 3 | `03_final_model_and_demo.ipynb` | outputs of 1 and 2 | figure 06 | seconds |

Use **Kernel → Restart & Run All** in each. The notebooks work whether launched from the repo root or from `notebooks/`.
The processed data and result files are also committed, so notebook 3 (the demo) can be run on its own.

### Live demo (notebook 3)
```python
from src.demo import predict_matchup, pickem_sheet
predict_matchup("Oregon", "Auburn", 2010)   # one game
pickem_sheet(2010)                          # full Pick'em sheet for a season
```
The demo excludes the chosen season from training, so it is a fair out-of-sample prediction.

## Team contributions
| Member | Owns |
|---|---|
| `<Name 1 – SRN>` | `src/features.py`, notebook 01, write-up |
| `<Name 2 – SRN>` | `src/models.py`, `src/demo.py`, notebooks 02–03, slide deck |

## Data notes
- Box-score statistics are empty for 2000–2003, so only **2004–2010** is used (223 bowl games).
- Features are built from regular-season games only (no leakage from the bowl game itself).
- `possession_*` columns contain impossible values and are excluded.

## Method summary
- 28 team-season features (scoring, yardage, efficiency, turnovers, penalties, strength of schedule) → differences `A − B` per bowl game.
- Symmetric (no-intercept) models, so swapping A and B flips the prediction.
- Evaluation: 7-fold leave-one-season-out CV; scaling, tuning and feature selection happen inside each training fold.
- Models: 3 rule baselines, Score Regression, Differential OLS / Ridge / p-value-selected OLS, Logistic Regression, SVM, Random Forest.

## Results (pooled CV accuracy, 223 games)
| Model | Accuracy |
|---|---|
| SVM (RBF) | 60.1 % |
| Score Regression | 59.2 % |
| Baseline: higher avg margin | 58.7 % |
| **Differential Ridge (primary)** | 55.6 % |
| Baseline: listed home team | 52.9 % |

Accuracy rises with predicted-margin confidence (about 45 % → 58 % → 64 % across terciles). Most model differences are within the 95 % CI (about ±6 points).
See the notebook's *Conclusions* for the full discussion and limitations.

## Reference
Cheshire, Childs, Leung. *College Football Bowl Game Predictor.* Stanford CS229 project report.

## Submission checklist
- [ ] Private GitHub repo shared with faculty (and TAs)
- [ ] 2-page PDF write-up in `docs/`
- [ ] Slide deck in `docs/`
- [ ] Replace the name/SRN placeholders in the README and the three notebooks
