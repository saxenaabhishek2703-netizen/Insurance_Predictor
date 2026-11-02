import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🏥", layout="centered")

# ---------- styling ----------
st.markdown("""
<style>
.block-container {max-width: 820px; padding-top: 2rem;}
.hero {
    background: linear-gradient(135deg, #1f4e8c 0%, #2a9d8f 100%);
    border-radius: 18px; padding: 28px 32px; color: #fff; margin-bottom: 22px;
}
.hero h1 {margin: 0 0 6px 0; font-size: 2rem; color: #fff;}
.hero p  {margin: 0; opacity: .92; font-size: 1.02rem;}
.result {
    border-radius: 18px; padding: 26px 30px; text-align: center; margin: 18px 0 8px 0;
    background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%); color: #fff;
}
.result .label {font-size: .95rem; letter-spacing: .08em; text-transform: uppercase; opacity: .85;}
.result .amount {font-size: 3rem; font-weight: 800; margin: 4px 0;}
.result .range {font-size: 1rem; opacity: .9;}
.chip {
    display: inline-block; padding: 5px 14px; border-radius: 999px;
    font-size: .85rem; font-weight: 600; margin: 3px 6px 3px 0; color: #fff;
}
.chip.red {background: #dc2626;} .chip.amber {background: #d97706;} .chip.green {background: #16a34a;}
.chip.blue {background: #2563eb;}
.small {opacity: .7; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)

# ---------- load the saved model and scaler ----------
@st.cache_resource
def load_files():
    return joblib.load("insurance_predictor_model.pkl"), joblib.load("scaler.pkl")

model, scaler = load_files()

# ---------- must match the notebook exactly ----------
COLS = ["age", "sex", "bmi", "children", "smoker", "region"]
SEX = {"male": 1, "female": 0}
SMOKER = {"yes": 1, "no": 0}
REGION = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
TYPICAL_ERROR = 2450          # model MAE on the test set (approx.)


def predict_charge(age, sex, bmi, children, smoker, region):
    row = pd.DataFrame([{
        "age": age, "sex": SEX[sex], "bmi": bmi, "children": children,
        "smoker": SMOKER[smoker], "region": REGION[region],
    }])[COLS]
    num = ["age", "bmi", "children"]
    row[num] = scaler.transform(row[num])      # transform only, never fit
    return float(model.predict(row)[0])


def bmi_category(bmi):
    if bmi < 18.5: return "Underweight", "blue"
    if bmi < 25:   return "Normal weight", "green"
    if bmi < 30:   return "Overweight", "amber"
    return "Obese", "red"


# ---------- header ----------
st.markdown("""
<div class="hero">
  <h1>🏥 Insurance Cost Predictor</h1>
  <p>Enter a few details and get an instant estimate of the yearly medical insurance charge.</p>
</div>
""", unsafe_allow_html=True)

# ---------- input form ----------
with st.form("inputs"):
    st.subheader("👤 About the customer")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 64, 35)
        sex = st.radio("Sex", list(SEX.keys()), horizontal=True, format_func=str.capitalize)
        children = st.slider("Children", 0, 5, 0)
    with c2:
        bmi = st.slider("BMI", 15.0, 55.0, 26.0, step=0.1)
        smoker = st.radio("Smoker", list(SMOKER.keys()), index=1, horizontal=True,
                          format_func=str.capitalize)
        region = st.selectbox("Region", list(REGION.keys()), format_func=str.capitalize)
    submitted = st.form_submit_button("💰 Predict my charge", use_container_width=True, type="primary")

# ---------- result ----------
if submitted:
    charge = predict_charge(age, sex, bmi, children, smoker, region)
    low, high = max(charge - TYPICAL_ERROR, 0), charge + TYPICAL_ERROR

    st.markdown(f"""
    <div class="result">
      <div class="label">Estimated yearly charge</div>
      <div class="amount">{charge:,.0f}</div>
      <div class="range">Likely range: {low:,.0f} – {high:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    # what drives the estimate
    cat, colour = bmi_category(bmi)
    chips = f'<span class="chip {colour}">BMI {bmi:.1f} · {cat}</span>'
    chips += ('<span class="chip red">Smoker</span>' if smoker == "yes"
              else '<span class="chip green">Non-smoker</span>')
    chips += f'<span class="chip blue">Age {age}</span>'
    st.markdown(chips, unsafe_allow_html=True)

    # what-if comparison
    st.markdown("#### 🔍 What-if comparison")
    other_smoker = "no" if smoker == "yes" else "yes"
    healthy_bmi = predict_charge(age, sex, min(bmi, 24.9), children, smoker, region)
    alt_smoker = predict_charge(age, sex, bmi, children, other_smoker, region)

    m1, m2 = st.columns(2)
    m1.metric("Same person, " + ("non-smoker" if smoker == "yes" else "smoker"),
              f"{alt_smoker:,.0f}", f"{alt_smoker - charge:+,.0f}", delta_color="inverse")
    m2.metric("Same person, BMI 24.9 or below", f"{healthy_bmi:,.0f}",
              f"{healthy_bmi - charge:+,.0f}", delta_color="inverse")

    st.markdown('<p class="small">This is a statistical estimate from a model trained on past data '
                f'(typical error ≈ ±{TYPICAL_ERROR:,}). It is not an official insurance quote.</p>',
                unsafe_allow_html=True)
else:
    st.info("Fill in the form above and press **Predict my charge**.")