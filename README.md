# IPL Win Predictor — Streamlit

This project turns the model pipeline from `Untitled53.ipynb` into a polished Streamlit UI.

## 1. Create the model

Run the final cells in your notebook. Your notebook saves the fitted pipeline as:

```text
pipe.pkl
```

The pipeline is a `ColumnTransformer` + `OneHotEncoder` + `LogisticRegression` and uses:

- batting_team
- bowling_team
- city
- runs_left
- balls_left
- wickets
- crr
- rrr
- total_runs_x

The notebook reports an accuracy of about **80.15%** on its held-out test split.

## 2. Put files together

```text
ipl_win_predictor_streamlit/
├── app.py
├── pipe.pkl
└── requirements.txt
```

## 3. Install and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app automatically reads the categories learned by your OneHotEncoder, so the team and city dropdowns stay compatible with the fitted model.

## Important

Do not load pickle files from untrusted sources. Python pickle can execute arbitrary code when deserialized.
