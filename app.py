import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(255, 94, 0, .12), transparent 28%),
        radial-gradient(circle at 90% 5%, rgba(255, 193, 7, .10), transparent 25%),
        linear-gradient(135deg, #09090b 0%, #111318 48%, #0a0b0f 100%);
    color: #f7f7f8;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

.hero {
    padding: 2.2rem 2.4rem;
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(255,94,0,.16), rgba(255,193,7,.06) 45%, rgba(255,255,255,.025));
    box-shadow: 0 20px 70px rgba(0,0,0,.28);
    margin-bottom: 1.4rem;
}

.hero-kicker {
    color: #ff9d5c;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
}

.hero h1 {
    font-size: clamp(2.3rem, 6vw, 4.8rem);
    line-height: .98;
    margin: .55rem 0 .8rem;
    font-weight: 800;
    letter-spacing: -.055em;
}

.hero p {
    color: #b8bbc4;
    font-size: 1.02rem;
    max-width: 720px;
    line-height: 1.7;
}

.card {
    padding: 1.35rem 1.5rem;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 22px;
    background: rgba(255,255,255,.035);
    box-shadow: 0 14px 45px rgba(0,0,0,.20);
}

.metric-label {
    color: #9da1ad;
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 700;
}

.metric-value {
    font-size: 1.65rem;
    font-weight: 800;
    margin-top: .2rem;
}

.predict-box {
    padding: 1.8rem;
    border-radius: 26px;
    background: linear-gradient(145deg, rgba(255,94,0,.18), rgba(255,255,255,.035));
    border: 1px solid rgba(255,145,60,.24);
    margin-top: 1.2rem;
}

.win {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -.04em;
}

.small {
    color: #9da1ad;
    font-size: .86rem;
}

div[data-testid="stButton"] > button {
    border-radius: 14px;
    border: 0;
    font-weight: 800;
    min-height: 3rem;
    background: linear-gradient(90deg, #ff5e00, #ff9d00);
    color: white;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #cfd2da;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(path="pipe.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_categories(model):
    """Read categories learned by the notebook's OneHotEncoder."""
    try:
        ct = model.named_steps["step1"]
        encoder = ct.named_transformers_["trf"]
        return {
            "batting_team": list(encoder.categories_[0]),
            "bowling_team": list(encoder.categories_[1]),
            "city": list(encoder.categories_[2]),
        }
    except Exception:
        return None


# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">Machine Learning • IPL Analytics</div>
  <h1>IPL Win Predictor 🏏</h1>
  <p>
    A polished interface for your Logistic Regression pipeline. Set the match
    situation, run the model, and get a live probability estimate for the
    chasing team's win.
  </p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ Model")
    model_path = st.text_input("Model file", value="pipe.pkl")
    st.caption("Place the pickle exported by your notebook next to app.py.")

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "The notebook trains a Pipeline with OneHotEncoder for batting team, "
        "bowling team and city, followed by Logistic Regression."
    )

model = None
model_error = None
if Path(model_path).exists():
    try:
        model = load_model(model_path)
    except Exception as exc:
        model_error = str(exc)
else:
    model_error = f"Could not find `{model_path}`."

if model_error:
    st.warning(
        f"Model not loaded yet: {model_error}\n\n"
        "Run the final notebook cell that creates `pipe.pkl`, then put that "
        "file in the same folder as this Streamlit app."
    )
    st.stop()

categories = get_categories(model)
if not categories:
    st.error("This app could not read the expected OneHotEncoder categories from the pipeline.")
    st.stop()

# ---------- Top stats ----------
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("MODEL", "Logistic Regression"),
    ("PIPELINE", "OneHotEncoder + LR"),
    ("TARGET", "Chasing win"),
    ("NOTEBOOK ACC.", "80.15%"),
]
for col, (label, value) in zip((c1, c2, c3, c4), stats):
    with col:
        st.markdown(
            f'<div class="card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("## Match situation")
st.caption("These fields match the features used by your trained pipeline.")

left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    batting = st.selectbox("🏏 Batting team", categories["batting_team"])
    bowling_options = [x for x in categories["bowling_team"] if x != batting] or categories["bowling_team"]
    bowling = st.selectbox("🎯 Bowling team", bowling_options)
    city = st.selectbox("📍 City / venue city", categories["city"])
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    runs_left = st.number_input("Runs left", min_value=1.0, max_value=300.0, value=60.0, step=1.0)
    balls_left = st.number_input("Balls left", min_value=1.0, max_value=120.0, value=36.0, step=1.0)
    wickets = st.number_input("Wickets remaining", min_value=0.0, max_value=10.0, value=7.0, step=1.0)
    target = st.number_input("First-innings score / target basis", min_value=1.0, max_value=350.0, value=170.0, step=1.0)
    st.markdown("</div>", unsafe_allow_html=True)

# CRR and RRR are exactly the formulas used in the notebook.
current_score = max(target - runs_left, 0.0)
balls_bowled = max(120.0 - balls_left, 1.0)
crr = (current_score * 6.0) / balls_bowled
rrr = (runs_left * 6.0) / balls_left

m1, m2, m3 = st.columns(3)
for col, label, value in [
    (m1, "Current score", f"{current_score:.0f}"),
    (m2, "Current run rate", f"{crr:.2f}"),
    (m3, "Required run rate", f"{rrr:.2f}"),
]:
    with col:
        st.markdown(
            f'<div class="card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown("### Prediction")
if st.button("✨ Predict win probability", use_container_width=True):
    X = pd.DataFrame([{
        "batting_team": batting,
        "bowling_team": bowling,
        "city": city,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets": wickets,
        "crr": crr,
        "rrr": rrr,
        "total_runs_x": target,
    }])

    try:
        probability = float(model.predict_proba(X)[0][1])
        prediction = int(model.predict(X)[0])

        st.markdown('<div class="predict-box">', unsafe_allow_html=True)
        if prediction == 1:
            title = f"{batting} is favored to win"
        else:
            title = f"{bowling} is favored to defend"

        st.markdown(f'<div class="win">{title}</div>', unsafe_allow_html=True)
        st.progress(min(max(probability, 0.0), 1.0))
        st.markdown(
            f"**Chasing win probability: {probability:.1%}**  \n"
            f"<span class='small'>Model output based on the supplied match situation.</span>",
            unsafe_allow_html=True,
        )

        with st.expander("View model input"):
            st.dataframe(X, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as exc:
        st.error(
            "Prediction failed. Make sure the `pipe.pkl` was generated from the "
            "same notebook pipeline and that the selected categories are present. "
            f"Details: {exc}"
        )

st.markdown("---")
st.caption("Built around the model pipeline in Untitled53.ipynb • For analysis and demonstration.")
