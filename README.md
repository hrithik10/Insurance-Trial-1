# 🛡️ Insurance Policy Status Classifier

An end-to-end Streamlit ML pipeline that predicts **POLICY_STATUS** from insurance data using:
- Decision Tree
- Random Forest
- Gradient Boosted Tree

## Features
| Step | Description |
|------|-------------|
| 1 | Package imports |
| 2 | Data check (shape, dtypes, nulls) – drops `POLICY_NO` & `PI_NAME` |
| 3 | Missing value imputation (mean for numeric, mode for categorical) |
| 4 | Label encoding with downloadable mapping CSV |
| 5 | Feature / Label split |
| 6 | 80:20 stratified train-test split |
| 7 | Model training (DT, RF, GBT) |
| 8 | Accuracy comparison table + bar chart |
| 9 | Confusion matrices (TP/FP/TN/FN labelled) |
| 10 | Feature importance bar charts |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, file `app.py`.
4. Click **Deploy**.

## Dataset
Upload any CSV with a `POLICY_STATUS` column. The default dataset uses:
`PI_GENDER, SUM_ASSURED, ZONE, PAYMENT_MODE, EARLY_NON, PI_OCCUPATION,`
`MEDICAL_NONMED, PI_STATE, REASON_FOR_CLAIM, PI_AGE, PI_ANNUAL_INCOME`

## Screenshot
Light-themed dashboard with sidebar hyperparameter controls, tabbed data views,
confusion matrices, and feature importance charts.
