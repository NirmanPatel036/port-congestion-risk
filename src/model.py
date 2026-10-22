import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import DMatrix, XGBClassifier

from .config import TEST_START, VAL_START
from .features import FEATURES, RISING

DEVICE = "cuda" if shutil.which("nvidia-smi") else "cpu"
PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5, subsample=0.8,
              colsample_bytree=0.8, tree_method="hist", eval_metric="logloss")


def design(df, columns=None, with_port=True):
    feats = FEATURES if with_port else [f for f in FEATURES if f != "port"]
    X = df[feats]
    if with_port:
        X = pd.get_dummies(X, columns=["port"], dtype=float)
    X = X.astype(float)
    return X if columns is None else X.reindex(columns=columns, fill_value=0.0)


def time_split(df, train_from=None):
    """Fixed-date train / validation / test masks. A row is dropped from a split if its outcome only became
    known after the next split began: in production that label would not exist yet (embargo)."""
    val_start, test_start = pd.Timestamp(VAL_START), pd.Timestamp(TEST_START)
    train = (df.arrival < val_start) & (df.label_time < val_start)
    if train_from is not None:
        train &= df.arrival >= pd.Timestamp(train_from)
    val = (df.arrival >= val_start) & (df.arrival < test_start) & (df.label_time < test_start)
    test = df.arrival >= test_start
    return train.to_numpy(), val.to_numpy(), test.to_numpy()


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_xgb(X, y, seed=0):
    mono = "(" + ",".join("1" if c in RISING else "0" for c in X.columns) + ")"
    m = XGBClassifier(**PARAMS, monotone_constraints=mono, random_state=seed, device=DEVICE).fit(X, y)
    return m.set_params(device="cpu")  # so the saved model also runs on a laptop without a GPU


def fit_ensemble(X_fit, y_fit, X_cal, y_cal, n=5):
    # bootstrap members: their disagreement is the confidence score
    rng = np.random.default_rng(0)
    members = [fit_xgb(X_fit.iloc[idx], y_fit.iloc[idx], seed)
               for seed, idx in enumerate(rng.integers(0, len(X_fit), (n, len(X_fit))))]
    raw = np.mean([m.predict_proba(X_cal)[:, 1] for m in members], axis=0)
    calibrator = LogisticRegression().fit(logit(raw)[:, None], y_cal)
    return {"members": members, "calibrator": calibrator, "columns": list(X_fit.columns)}


def predict(bundle, df):
    X = design(df, bundle["columns"])
    P = np.column_stack([m.predict_proba(X)[:, 1] for m in bundle["members"]])
    prob = bundle["calibrator"].predict_proba(logit(P.mean(1))[:, None])[:, 1]
    spread = P.std(1)
    confidence = np.select([spread < 0.05, spread < 0.12], ["High", "Medium"], "Low")
    return prob, spread, confidence


def contributions(bundle, df):
    """Per-feature push on the log-odds (TreeSHAP), averaged over members; port dummies folded into one."""
    X = design(df, bundle["columns"])
    dm = DMatrix(X)
    c = np.mean([m.get_booster().predict(dm, pred_contribs=True) for m in bundle["members"]], axis=0)[:, :-1]
    c = pd.DataFrame(c, columns=X.columns, index=df.index)
    ports = [col for col in c if col.startswith("port_")]
    return c.drop(columns=ports).assign(port=c[ports].sum(axis=1))
