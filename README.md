# Heart Disease Risk Prediction

An educational ML project that predicts heart disease risk from patient health data.

## What was fixed before publishing

1. **Inverted label bug (critical).** The Kaggle copy of `heart.csv` used here has its
   `target` column encoded backwards versus the standard convention. This was verified
   empirically: clinically risky values (high `oldpeak`, exercise angina, more blocked
   vessels) were correlating with `target=1` in the wrong direction, and the three demo
   patients in the original GUI predicted the opposite of what they should have
   ("High Risk Demo" → predicted no disease, and vice versa). Fixed by flipping the
   label (`target = 1 - target`) in both `Program1.py` and `GUI_App.py`. Cross-validated
   accuracy is unchanged (~82%) after the fix, confirming it was purely a label-direction
   issue, not a modeling problem.
2. **Train/test leakage in the analysis script.** `Program1.py` did not remove duplicate
   rows before splitting; this dataset has ~70% exact duplicates, so any reported accuracy
   from that script was inflated. Deduplication added.
3. **Hardcoded local file path** in `Program1.py` replaced with a relative path.
4. **Stale/version-mismatched model pickle** removed from the deployed package — the app
   retrains itself at startup instead, which avoids cross-machine/scikit-learn version
   issues entirely.
5. Stale `heart_patient_prediction_records.csv` sample data was not carried into the
   public web version (the web app does not persist patient data anywhere by default).

## Real model performance (5-fold cross-validated, after fixes)

| Metric | Score |
|---|---|
| Accuracy | ~82% |
| Precision | ~82% |
| Recall | ~78–86% (varies slightly by split) |
| F1 Score | ~80–84% |

This is in line with what's normally achievable on this dataset with logistic regression.
Don't expect higher without a much larger/cleaner clinical dataset — and don't be tempted
to "fix" a lower number by re-introducing duplicates or leaking test data, since that
just produces a fake high score.

## Files

- `app.py` — the public web app (Streamlit). This is what you deploy.
- `heart.csv` — training dataset (must stay in the same folder/repo as `app.py`).
- `requirements.txt` — Python dependencies for deployment.
- `GUI_App.py` — original desktop (Tkinter) version, fixed, kept for reference/demo to your instructor.
- `Program1.py` — original EDA/analysis script, fixed, useful for your project report's charts.

## How to deploy so anyone can use it with one click (free)

**Streamlit Community Cloud** (recommended — free, no server management):

1. Create a free GitHub account if you don't have one, and create a new public repository.
2. Upload `app.py`, `heart.csv`, and `requirements.txt` to that repository (just these
   three files — `GUI_App.py` and `Program1.py` don't need to go in the deployed repo,
   keep those for your own report/submission).
3. Go to https://share.streamlit.io, sign in with GitHub, click "New app," and point it
   at your repository and `app.py`.
4. Click Deploy. After a minute you'll get a public link like
   `https://your-app-name.streamlit.app` — anyone can open it and use the predictor
   directly in their browser, no installation needed.
5. Share that link. Each time you push a change to GitHub, the live app updates automatically.

No backend, server, or "files for everyone" — just the one link.

## Limitations to be upfront about (worth mentioning in your report/disclaimer)

- Trained on 310 unique patient records — solid for a class project, small by clinical
  research standards. Don't claim clinical-grade accuracy.
- It's a screening tool, not a diagnostic device. The app already shows this disclaimer
  to users — keep it there.
