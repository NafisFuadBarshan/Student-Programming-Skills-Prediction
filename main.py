"""Run the full pipeline: python main.py [--data Studentdata.csv]"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import student_skills as ss


def main(data_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    df = ss.load_data(data_path)
    X, y = ss.build_features(df)
    print(f"Rows: {len(X)}  Features: {X.shape[1]}")
    print("Target distribution:\n", y.value_counts().sort_index().to_string(), "\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=ss.RANDOM_STATE)

    # Pick the ridge penalty by CV on the training split only (no test leakage).
    grid = [0.1, 1, 3, 10, 30, 100, 300]
    best_alpha = max(grid, key=lambda a: ss.cross_validate(
        lambda: ss.ManualLinearRegression(alpha=a), X_tr, y_tr)["R2"][0])
    print(f"Ridge alpha chosen by training-set CV: {best_alpha}\n")

    models = {
        "Mean baseline": None,
        "Scikit-learn OLS": ss.make_sklearn,
        "Manual OLS (NumPy)": lambda: ss.ManualLinearRegression(),
        f"Manual Ridge (a={best_alpha})": lambda: ss.ManualLinearRegression(alpha=best_alpha),
    }
    rows, preds = [], {}
    for name, factory in models.items():
        if factory is None:
            m = ss.baseline_metrics(y_tr, y_te)
            t = np.nan
        else:
            model = factory().fit(X_tr, y_tr)
            preds[name] = model.predict(X_te)
            m = ss.regression_metrics(y_te, preds[name])
            t = ss.time_fit(factory(), X_tr, y_tr)
        cv = None if factory is None else ss.cross_validate(factory, X, y)
        rows.append({"Model": name, **m, "Fit time (s)": t,
                     "CV R2 (mean)": cv["R2"][0] if cv else np.nan,
                     "CV R2 (std)": cv["R2"][1] if cv else np.nan})
    table = pd.DataFrame(rows).set_index("Model")
    pd.set_option("display.width", 160)
    print(table.round(4).to_string())
    table.round(6).to_csv(out / "metrics.csv")

    skl, man = ss.make_sklearn().fit(X_tr, y_tr), ss.ManualLinearRegression().fit(X_tr, y_tr)
    assert np.allclose(skl.coef_, man.coef_, atol=1e-6), "manual != sklearn"
    print("\nManual and scikit-learn coefficients match.")

    coefs = pd.DataFrame({"feature": X.columns, "coef": man.coef_})
    coefs = coefs.reindex(coefs.coef.abs().sort_values(ascending=False).index)
    print("\nTop 10 coefficients (by magnitude):\n", coefs.head(10).to_string(index=False))
    coefs.to_csv(out / "coefficients.csv", index=False)

    # Figure 1: metrics bars (errors and R2 on separate axes: different scales)
    plot = table.drop(index="Mean baseline")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    plot[["MSE", "RMSE", "MAE"]].T.plot.bar(ax=a1, rot=0)
    a1.set_title("Test-set error (lower is better)")
    a1.set_ylim(0, 1.3)
    pd.concat([plot["R2"], plot["CV R2 (mean)"]], axis=1, keys=["Test R2", "5-fold CV R2"]
              ).plot.bar(ax=a2, rot=15)
    a2.axhline(0, color="k", lw=0.8)
    a2.set_title("R2 (higher is better)")
    fig.tight_layout()
    fig.savefig(out / "1.png", dpi=130)

    # Figure 2: actual vs predicted (jittered: the target only has 4 levels)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, name in zip(axes, ["Scikit-learn OLS", "Manual OLS (NumPy)"]):
        ax.scatter(y_te + rng.normal(0, 0.08, len(y_te)), preds[name], alpha=0.7)
        lim = [y.min() - .5, y.max() + .5]
        ax.plot(lim, lim, "k--")
        ax.set(title=name, xlabel="Actual (jittered)", ylabel="Predicted")
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "2.png", dpi=130)
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Studentdata.csv")
    p.add_argument("--out", default="Output Visuals")
    a = p.parse_args()
    main(a.data, a.out)
