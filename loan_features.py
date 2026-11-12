"""Shared feature engineering for training (train.py) and the Streamlit app (app.py).

Only variables known at loan application time are used, so there is no data leakage
from post-origination fields such as total_pymnt, recoveries or last_pymnt_d.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

CANDIDATES = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "purpose", "dti", "delinq_2yrs",
    "earliest_cr_line", "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    "application_type", "initial_list_status",
]

REFERENCE_YEAR = 2017  # dataset runs to 2017Q3; used for credit history length


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Lending Club columns and add engineered features. Returns features only."""
    feat = [c for c in CANDIDATES if c in df.columns]
    d = df[feat].copy()

    for c in ["term", "int_rate", "revol_util"]:
        if c in d.columns and not is_numeric_dtype(d[c]):
            d[c] = pd.to_numeric(d[c].astype(str).str.extract(r"([\d\.]+)")[0], errors="coerce")

    if "emp_length" in d.columns and not is_numeric_dtype(d["emp_length"]):
        e = d["emp_length"].astype(str)
        years = pd.to_numeric(e.str.extract(r"(\d+)")[0], errors="coerce")
        d["emp_length"] = np.where(e.str.contains("<"), 0, years)

    if "earliest_cr_line" in d.columns:
        yr = pd.to_datetime(d["earliest_cr_line"], errors="coerce", format="mixed").dt.year
        d["credit_hist_years"] = REFERENCE_YEAR - yr
        d = d.drop(columns="earliest_cr_line")

    if {"fico_range_low", "fico_range_high"} <= set(d.columns):
        d["fico_avg"] = (d["fico_range_low"] + d["fico_range_high"]) / 2
        d = d.drop(columns=["fico_range_low", "fico_range_high"])

    if {"loan_amnt", "annual_inc"} <= set(d.columns):
        d["loan_to_income"] = d["loan_amnt"] / d["annual_inc"].replace(0, np.nan)
    if {"installment", "annual_inc"} <= set(d.columns):
        d["installment_to_income"] = 12 * d["installment"] / d["annual_inc"].replace(0, np.nan)

    return d


def monthly_installment(amount: float, annual_rate_pct: float, months: int) -> float:
    """Standard amortisation formula; matches how Lending Club computes `installment`."""
    r = annual_rate_pct / 1200.0
    if r == 0:
        return amount / months
    return amount * r * (1 + r) ** months / ((1 + r) ** months - 1)


def encode_row(row: dict, meta: dict) -> pd.DataFrame:
    """Turn one borrower (dict of clean feature values) into the model's encoded columns."""
    out = {}
    for col in meta["columns"]:
        if "=" in col:
            cat, val = col.split("=", 1)
            out[col] = 1.0 if str(row.get(cat)) == val else 0.0
        else:
            v = row.get(col)
            out[col] = meta["medians"][col] if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    return pd.DataFrame([out], columns=meta["columns"]).astype("float32")
