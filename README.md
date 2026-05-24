# covid_acuity_score

**Public.** Training and daily-prediction code for the COVID-19 acuity /
risk-prediction score developed in the CDAC group.

## Layout

```
a6_train_risk_model.py   train the COVID risk-prediction model
a7_daily_prediction.py   daily inference: pull new patients, predict, store
models/                  trained model artifacts
```

## Required environment

Python 3 with `scikit-learn`, `numpy`, `pandas`, `scipy`, `tqdm`.

## Status

Paper-accompanying code. Public; carries the standard noncommercial LICENSE.
