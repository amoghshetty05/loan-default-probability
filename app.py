"""Streamlit UI: score a borrower's probability of loan charge-off.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import xgboost as xgb

from loan_features import encode_row, monthly_installment

ROOT = Path(__file__).parent
st.set_page_config(page_title="Loan Default Predictor", page_icon="💳", layout="wide")


@st.cache_resource
def load_model():
    meta = json.loads((ROOT / "model" / "meta.json").read_text())
    booster = xgb.Booster()
    booster.load_model(str(ROOT / "model" / "xgb_model.json"))
    return booster, meta


try:
    booster, meta = load_model()
except Exception:
    st.error("Model files not found. Run `python train.py` first to create model/xgb_model.json.")
    st.stop()

med, cats, rng = meta["medians"], meta["categories"], meta["ranges"]


def predict(row: dict):
    X = encode_row(row, meta)
    dm = xgb.DMatrix(X, feature_names=meta["columns"])
    p = float(booster.predict(dm)[0])
    contrib = booster.predict(dm, pred_contribs=True)[0][:-1]  # last value is the bias term
    return p, pd.Series(contrib, index=meta["columns"])


def risk_band(p: float):
    if p < 0.10:
        return "Low", "🟢"
    if p < 0.25:
        return "Medium", "🟡"
    if p < 0.40:
        return "High", "🟠"
    return "Very high", "🔴"


def _render_result(p, band, icon, contrib, threshold):
    r1, r2, r3 = st.columns([1, 1, 2])
    r1.metric("Probability of charge-off", f"{p:.1%}",
              delta=f"{(p - meta['default_rate']) * 100:+.1f} pts vs portfolio avg ({meta['default_rate']:.1%})",
              delta_color="inverse")
    r1.progress(min(p, 1.0))
    r2.metric("Risk band", f"{icon} {band}")
    if p > threshold:
        r2.error(f"Recommend REJECT (PD {p:.1%} > {threshold:.1%})")
    else:
        r2.success(f"Recommend APPROVE (PD {p:.1%} ≤ {threshold:.1%})")

    with r3:
        st.markdown("**Why this score?** Top drivers (log-odds contribution; red raises risk, green lowers it)")
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:8][::-1]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.barh([n.replace("=", " = ") for n in top.index], top.values,
                color=["#C44E52" if v > 0 else "#55A868" for v in top.values])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Effect on default risk (log-odds)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.divider()


st.title("💳 Loan Default Probability")
st.caption("Lending Club loans 2007–2017Q3 · XGBoost model trained on application-time data only "
           "(no leakage). Educational demo, not credit advice.")

tab_pred, tab_perf, tab_about = st.tabs(["Predict", "Model performance", "About"])

# ----------------------------------------------------------------------------- Predict
with tab_pred:
    with st.sidebar:
        st.header("Decision policy")
        ratio = st.slider("A missed default costs X times a rejected good loan", 1, 20, 5,
                          help="Cost-sensitive threshold: reject if PD > cost_FP / (cost_FP + cost_FN).")
        threshold = 1 / (1 + ratio)
        st.metric("Reject loans with PD above", f"{threshold:.1%}")
        st.caption("With calibrated probabilities, this cut-off minimises expected cost. "
                   f"A plain 50% cut-off catches only {meta['metrics']['Recall@0.5']:.0%} of defaulters in testing.")

    result_box = st.container()
    st.subheader("Borrower and loan details")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Loan**")
        amount = st.number_input("Loan amount ($)", 1000, 40000, int(med.get("loan_amnt", 12000)), 500)
        term = st.selectbox("Term (months)", [36, 60], index=0)
        rate = st.slider("Interest rate (%)", 5.0, 31.0, round(med.get("int_rate", 13.0), 1), 0.1)
        grade = st.selectbox("Grade", cats["grade"], index=min(1, len(cats["grade"]) - 1))
        sub_n = st.selectbox("Sub-grade level", [1, 2, 3, 4, 5], index=2)
        purpose = st.selectbox("Purpose", cats["purpose"],
                               index=cats["purpose"].index("debt_consolidation")
                               if "debt_consolidation" in cats["purpose"] else 0)
        installment = monthly_installment(amount, rate, term)
        st.caption(f"Monthly installment: **${installment:,.0f}**")

    with c2:
        st.markdown("**Borrower**")
        income = st.number_input("Annual income ($)", 4000, 500000, int(med.get("annual_inc", 65000)), 1000)
        dti = st.slider("Debt-to-income (%)", 0.0, 50.0, round(med.get("dti", 17.0), 1), 0.1)
        fico = st.slider("FICO score", 660, 850, int(med.get("fico_avg", 690)), 1)
        emp = st.slider("Employment length (years, 10 = 10+)", 0, 10, int(med.get("emp_length", 6)))
        home = st.selectbox("Home ownership", cats["home_ownership"],
                            index=cats["home_ownership"].index("MORTGAGE")
                            if "MORTGAGE" in cats["home_ownership"] else 0)
        verif = st.selectbox("Income verification", cats["verification_status"])

    with c3:
        st.markdown("**Credit history**")
        hist = st.slider("Credit history (years)", 0, 60, int(med.get("credit_hist_years", 15)))
        open_acc = st.slider("Open accounts", 0, 40, int(med.get("open_acc", 11)))
        total_acc = st.slider("Total accounts ever", 1, 80, int(med.get("total_acc", 24)))
        revol_bal = st.number_input("Revolving balance ($)", 0, 200000, int(med.get("revol_bal", 11000)), 500)
        revol_util = st.slider("Revolving utilisation (%)", 0.0, 120.0, round(med.get("revol_util", 50.0), 1), 0.5)
        inq = st.slider("Credit inquiries, last 6 months", 0, 8, int(med.get("inq_last_6mths", 0)))
        delinq = st.slider("Delinquencies, last 2 years", 0, 10, int(med.get("delinq_2yrs", 0)))
        pub_rec = st.slider("Public records", 0, 5, int(med.get("pub_rec", 0)))
        bankr = st.slider("Public-record bankruptcies", 0, 3, int(med.get("pub_rec_bankruptcies", 0)))
        mort = st.slider("Mortgage accounts", 0, 10, int(med.get("mort_acc", 1)))

    row = {
        "loan_amnt": amount, "term": term, "int_rate": rate, "installment": installment,
        "grade": grade, "sub_grade": f"{grade}{sub_n}", "emp_length": emp, "home_ownership": home,
        "annual_inc": income, "verification_status": verif, "purpose": purpose, "dti": dti,
        "delinq_2yrs": delinq, "inq_last_6mths": inq, "open_acc": open_acc, "pub_rec": pub_rec,
        "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc, "mort_acc": mort,
        "pub_rec_bankruptcies": bankr, "credit_hist_years": hist, "fico_avg": fico,
        "loan_to_income": amount / income, "installment_to_income": 12 * installment / income,
        "application_type": "Individual", "initial_list_status": "w",
    }
    p, contrib = predict(row)
    band, icon = risk_band(p)

    with result_box:
        _render_result(p, band, icon, contrib, threshold)

# ----------------------------------------------------------------------------- Performance
with tab_perf:
    m = meta["metrics"]
    st.subheader("XGBoost on the held-out test set")
    st.caption(f"{meta['n_train']:,} train / {meta['n_val']:,} validation / {meta['n_test']:,} test loans "
               f"(stratified). Best boosting round: {meta['best_iteration']}.")
    cols = st.columns(6)
    for col, k in zip(cols, ["ROC-AUC", "PR-AUC", "RMSE", "MAE", "R2", "LogLoss"]):
        col.metric(k, f"{m[k]:.3f}")

    a, b, c = st.columns(3)
    with a:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(meta["roc"]["fpr"], meta["roc"]["tpr"], label=f"XGBoost (AUC {m['ROC-AUC']:.3f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
        ax.legend(loc="lower right")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with b:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(meta["calibration"]["predicted"], meta["calibration"]["observed"], marker="o", label="XGBoost")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.set(xlabel="Predicted PD", ylabel="Observed default rate", title="Calibration")
        ax.legend()
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c:
        gr = meta["grade_default_rate"]
        if gr:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.bar(list(gr.keys()), list(gr.values()), color="#DD8452")
            ax.set(ylabel="Charge-off rate (%)", title="Default rate by grade")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    comp = ROOT / "model_comparison.csv"
    st.subheader("Five-framework comparison")
    if comp.exists():
        st.dataframe(pd.read_csv(comp, index_col=0).round(4))
    else:
        st.info("Run the Colab notebook to the end and add the exported `model_comparison.csv` "
                "to the repo root; the table of XGBoost, Keras, fastai, TabNet and Lightning appears here.")

# ----------------------------------------------------------------------------- About
with tab_about:
    st.markdown(f"""
### What this is
A web front-end for the *Machine Learning Model for Predicting Loan Default Probability* case study.
The target is **Charge-Off** (1 = the lender gave up collecting, 0 = fully paid). It is a binary
classification problem, and the model outputs a **probability of default (PD)**.

### How it works
1. Dataset: {meta['n_loans']:,} closed Lending Club loans, charge-off rate {meta['default_rate']:.1%}.
2. Only variables known at application time are used; post-origination fields are excluded to avoid leakage.
3. Engineered features: FICO average, credit-history length, loan-to-income, installment-to-income.
4. XGBoost with early stopping. No class re-weighting, so probabilities stay calibrated.
5. **Explanations** use XGBoost's built-in SHAP contributions for each individual score.
6. **Decision policy** uses a cost ratio instead of a fixed 50% cut-off.

### Limitations
Random (not time-based) split, application-time data only, and the cost ratio is an assumption.
Not a real underwriting tool: it is a coursework demo.
""")
