"""Student programming-skills prediction: data preparation, models, evaluation.

Everything the notebook and ``main.py`` need lives here so it can be tested.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

TARGET = "n_languages"
RANDOM_STATE = 42

# Raw survey column (after stripping) -> short, stable name.
# Matching on the text after the leading "N." numbering avoids the original
# bug where "2.Gender" (no space after the dot) was never cleaned.
COLUMN_RENAMES = {
    "Gender": "gender",
    "Year of Study": "year",
    "Branch": "branch",
    "Type of Schooling till 10th Grade:": "school_type_10",
    "Percentage in Class 10th?": "pct_10",
    "Location of Class 10th School:": "location_10",
    "Type of Schooling for class 12th:": "school_type_12",
    "Percentage in Class 12th?": "pct_12",
    "Location of Class 12th School:": "location_12",
    "Dropped an year after School?": "gap_year",
    "Are you about to become first generation graduate ?": "first_gen",
    "Involved in additional volunteer groups?": "volunteer",
    "How many programming languages do you know?": TARGET,
    "Actively involved in Specific technology?": "technologies",
    "Exposure to Data structure and algorithms?": "dsa",
    "Exposure to GitHub ?": "github",
    "Are you associated with any developer Community?": "in_community",
    "Active in developer Communities": "community_activity",
    "Participating in Hackathons and other competitions?": "hackathons",
    "Have you made any Software based projects?": "software_project",
    "Have you made any Hardware based project": "hardware_project",
    "Have you ever pitched any idea?": "pitched_idea",
    "Do you have any additional skills?": "extra_skills",
    "Achievements:": "achievements",
}

# The CSV was exported through Excel, which turned the answer ranges "2-3" and
# "4-5" into the dates "3-Feb" and "5-Apr". Accept both spellings.
TARGET_MAP = {
    "none": 0.0,
    "1": 1.0,
    "2-3": 2.5, "3-feb": 2.5,
    "4-5": 4.5, "5-apr": 4.5,
    "more than 5": 6.0,
}

BINARY_COLS = ["gender", "gap_year", "first_gen", "volunteer", "in_community",
               "software_project", "hardware_project", "pitched_idea"]
BINARY_POSITIVE = {"gender": "Female"}  # every other binary column: "Yes" = 1

# Ordered categories get integer codes that respect the order.
ORDINAL_MAPS = {
    "year": {"First": 1, "Second": 2, "Third": 3},
    "pct_10": {"<60%": 0, ">=60% and <70%": 1, ">=70% and <80%": 2,
               ">=80% and <90%": 3, ">=90%": 4},
    "pct_12": {"<60%": 0, ">=60% and <70%": 1, ">=70% and <80%": 2,
               ">=80% and <90%": 3, ">=90%": 4},
    "dsa": {"Easy": 0, "Medium": 1, "Hard": 2},
    "github": {"Easy": 0, "Medium": 1, "Hard": 2},
    # NaN here means "not in a community", which is structural, not random.
    "community_activity": {"Not in a community": 0, "Inactive Member": 1,
                           "Moderately active": 2, "Active Member": 3},
    "hackathons": {"Not yet participated": 0, "Participated": 1,
                   "Participated and Won": 2},
}

NOMINAL_COLS = ["branch", "school_type_10", "school_type_12", "location_10",
                "location_12"]
MULTI_COLS = ["technologies", "extra_skills", "achievements"]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the stray index column and rename survey questions to short names."""
    df = df.copy()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    new = []
    for col in df.columns:
        key = col.strip()
        if "." in key[:4]:  # strip a leading "12." / "2." question number
            key = key.split(".", 1)[1].strip()
        # Long question with a parenthetical list of examples
        for raw in COLUMN_RENAMES:
            if key.startswith(raw):
                key = raw
                break
        if key not in COLUMN_RENAMES:
            raise KeyError(f"Unexpected column in CSV: {col!r}")
        new.append(COLUMN_RENAMES[key])
    df.columns = new
    return df


def load_data(path: str | Path = "Studentdata.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    df = clean_columns(pd.read_csv(path, dtype=str, keep_default_na=False,
                                   na_values=[""]))
    # Normalise whitespace so ">= 90%" and ">=90%" become one category.
    for col in df.columns:
        df[col] = df[col].str.strip().str.replace(r"\s+", " ", regex=True)
        df[col] = df[col].str.replace(">= ", ">=", regex=False)
    return df


def encode_target(series: pd.Series) -> pd.Series:
    mapped = series.str.lower().map(TARGET_MAP)
    unknown = series[mapped.isna() & series.notna()].unique()
    if len(unknown):
        raise ValueError(f"Unrecognised target values: {list(unknown)}")
    return mapped


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Turn the cleaned survey into a numeric design matrix and target."""
    df = df.copy()
    y = encode_target(df[TARGET])
    keep = y.notna()
    df, y = df[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    y.name = TARGET

    parts = []
    for col in BINARY_COLS:
        pos = BINARY_POSITIVE.get(col, "Yes")
        parts.append((df[col] == pos).astype(int).rename(col))

    df["community_activity"] = df["community_activity"].fillna("Not in a community")
    for col, mapping in ORDINAL_MAPS.items():
        coded = df[col].map(mapping)
        if coded.isna().any():
            raise ValueError(f"Unmapped values in {col}: "
                             f"{df.loc[coded.isna(), col].unique()}")
        parts.append(coded.astype(int).rename(col))

    # One-hot for unordered categories. Rare levels (<5 rows) are pooled so the
    # model is not fitted on a column with one or two ones in it.
    for col in NOMINAL_COLS:
        s = df[col].copy()
        counts = s.value_counts()
        s[s.isin(counts[counts < 5].index)] = "Other"
        parts.append(pd.get_dummies(s, prefix=col, drop_first=True, dtype=int))

    # Multi-select answers ("A;B;C") -> one indicator column per choice.
    # A missing answer means no choice was ticked, so it is all zeros.
    for col in MULTI_COLS:
        multi = df[col].fillna("").str.get_dummies(sep=";")
        multi = multi.drop(columns=["None"], errors="ignore")
        multi.columns = [f"{col}_{c.strip().replace(' ', '_')}" for c in multi.columns]
        parts.append(multi.astype(int))

    X = pd.concat(parts, axis=1)
    return X, y


class ManualLinearRegression:
    """Least squares via the normal equation, with optional ridge penalty.

    Uses a linear solve (pseudo-inverse fallback) rather than ``np.linalg.inv``,
    so it still works when features are collinear. The intercept is never
    penalised.
    """

    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha
        self.weights: np.ndarray | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_b = np.c_[np.ones(X.shape[0]), X]
        gram = X_b.T @ X_b
        if self.alpha:
            penalty = self.alpha * np.eye(gram.shape[0])
            penalty[0, 0] = 0.0
            gram = gram + penalty
        try:
            self.weights = np.linalg.solve(gram, X_b.T @ y)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.pinv(gram) @ (X_b.T @ y)
        return self

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        return np.c_[np.ones(X.shape[0]), X] @ self.weights

    @property
    def intercept_(self):
        return self.weights[0]

    @property
    def coef_(self):
        return self.weights[1:]


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


def time_fit(model, X, y, repeats: int = 200) -> float:
    """Median fit time in seconds (single timings are dominated by noise)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.fit(X, y)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def cross_validate(make_model, X, y, n_splits: int = 5) -> dict:
    """Mean/std of each metric over shuffled k-fold CV."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    rows = []
    for tr, te in KFold(n_splits, shuffle=True, random_state=RANDOM_STATE).split(X):
        m = make_model().fit(X[tr], y[tr])
        rows.append(regression_metrics(y[te], m.predict(X[te])))
    out = pd.DataFrame(rows)
    return {k: (out[k].mean(), out[k].std()) for k in out}


def baseline_metrics(y_train, y_test) -> dict:
    """Always predict the training mean: the bar any model has to clear."""
    return regression_metrics(y_test, np.full(len(y_test), np.mean(y_train)))


def make_sklearn():
    return LinearRegression()
