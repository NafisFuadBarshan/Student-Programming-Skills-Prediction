import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import student_skills as ss

DATA = Path(__file__).resolve().parents[1] / "Studentdata.csv"


@pytest.fixture(scope="module")
def data():
    return ss.build_features(ss.load_data(DATA))


def test_no_index_column_leak(data):
    X, _ = data
    assert not any(c.lower().startswith("unnamed") for c in X.columns)


def test_all_numeric_no_nan(data):
    X, y = data
    assert not X.isna().any().any() and not y.isna().any()
    assert all(np.issubdtype(t, np.number) for t in X.dtypes)
    assert len(X) == len(y) == 256


def test_percentage_categories_merged():
    assert ss.load_data(DATA)["pct_10"].nunique() == 5


def test_target_values(data):
    assert set(data[1].unique()) == {1.0, 2.5, 4.5, 6.0}


def test_target_accepts_both_spellings():
    out = ss.encode_target(pd.Series(["2-3", "3-Feb", "4-5", "5-Apr", "None"]))
    assert out.tolist() == [2.5, 2.5, 4.5, 4.5, 0.0]


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        ss.encode_target(pd.Series(["lots"]))


def test_manual_matches_sklearn(data):
    X, y = data
    a, b = ss.ManualLinearRegression().fit(X, y), LinearRegression().fit(X, y)
    assert np.allclose(a.coef_, b.coef_, atol=1e-6)
    assert np.allclose(a.predict(X), b.predict(X), atol=1e-6)


def test_manual_ridge_matches_sklearn(data):
    X, y = data
    a = ss.ManualLinearRegression(alpha=5.0).fit(X, y)
    b = Ridge(alpha=5.0).fit(X, y)
    assert np.allclose(a.coef_, b.coef_, atol=1e-6)
    assert a.intercept_ == pytest.approx(b.intercept_)


def test_collinear_features_do_not_crash():
    x = np.arange(10.0)
    m = ss.ManualLinearRegression().fit(np.c_[x, 2 * x], 3 * x + 1)
    assert np.allclose(m.predict(np.c_[x, 2 * x]), 3 * x + 1, atol=1e-6)


def test_predict_before_fit():
    with pytest.raises(RuntimeError):
        ss.ManualLinearRegression().predict(np.zeros((2, 2)))


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        ss.load_data("nope.csv")
