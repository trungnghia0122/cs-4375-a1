# part2.py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


PLOTS_DIR = Path("plots")
REPORTS_DIR = Path("reports")
LOGS_DIR = Path("logs")
for d in [PLOTS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

def plot_learning_curve_sgd(loss_curve, out_path):
    if not loss_curve:
        return
    plt.figure()
    plt.plot(np.arange(1, len(loss_curve) + 1), loss_curve)
    plt.xlabel("Iteration")
    plt.ylabel("Average loss")
    plt.title("SGDRegressor learning curve (loss vs iterations)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_parity(y_true, y_pred, out_path):
    plt.figure()
    plt.scatter(y_true, y_pred, s=12, alpha=0.6)
    m = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(m, m)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Parity plot")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_residuals(y_true, y_pred, out_path):
    res = y_pred - y_true
    plt.figure()
    plt.hist(res, bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_coefficients(feature_names, coefs, out_path):
    order = np.argsort(np.abs(coefs))[::-1]
    plt.figure()
    plt.bar(range(len(coefs)), coefs[order])
    plt.xticks(range(len(coefs)), np.array(feature_names)[order], rotation=45, ha="right")
    plt.ylabel("Weight")
    plt.title("Model coefficients (sorted by |weight|)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ds = fetch_ucirepo(id=186)
    X = ds.data.features.select_dtypes(include=[np.number]).copy()
    y = ds.data.targets.squeeze()

    # Split 80/20, then carve out 10 percent of train for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=123, shuffle=True)

    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SGDRegressor(
            loss="squared_error",
            penalty=None,           # no regularization to mirror part 1
            fit_intercept=True,
            tol=None,
            random_state=0,
            average=False           # keep simple vanilla SGD
        ))
    ])

    # Hyperparameter grid
    grid_eta0 = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    grid_iters = [2000, 5000, 10000]

    log_path = LOGS_DIR / "part2_tuning_log.csv"
    log_path.write_text("trial,eta0,max_iter,train_mse,val_mse\n", encoding="utf-8")

    best = {"val_mse": np.inf, "eta0": None, "max_iter": None, "pipe": None}
    trial = 0

    for eta0 in grid_eta0:
        for max_iter in grid_iters:
            trial += 1
            pipe = base_pipe.set_params(
                model__learning_rate="constant",
                model__eta0=eta0,
                model__max_iter=max_iter
            )
            pipe.fit(X_tr, y_tr)

            y_tr_pred = pipe.predict(X_tr)
            y_val_pred = pipe.predict(X_val)
            tr_mse = mean_squared_error(y_tr, y_tr_pred)
            v_mse = mean_squared_error(y_val, y_val_pred)

            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{trial},{eta0},{max_iter},{tr_mse:.6f},{v_mse:.6f}\n")

            if v_mse < best["val_mse"]:
                best.update({"val_mse": v_mse, "eta0": eta0, "max_iter": max_iter, "pipe": pipe})

    print(f"Best params -> eta0={best['eta0']}, max_iter={best['max_iter']}, val MSE={best['val_mse']:.6f}")
    print(f"Log saved to: {log_path.resolve()}")

    # Retrain on train+val
    X_full = pd.concat([X_tr, X_val], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_tr, y_val], axis=0).reset_index(drop=True)

    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SGDRegressor(
            loss="squared_error",
            penalty=None,
            fit_intercept=True,
            tol=None,
            random_state=0,
            learning_rate="constant",
            eta0=best["eta0"],
            max_iter=best["max_iter"]
        ))
    ])
    final_pipe.fit(X_full, y_full)

    # Metrics on test
    y_test_pred = final_pipe.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_ev = explained_variance_score(y_test, y_test_pred)

    # Coefficients after training
    model = final_pipe.named_steps["model"]
    coefs = model.coef_
    intercept = model.intercept_[0]

    # Save metrics and coefficients
    metrics_path = REPORTS_DIR / "part2_final_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"Best params: eta0={best['eta0']}, max_iter={best['max_iter']}\n")
        f.write(f"Test MSE: {test_mse:.6f}\n")
        f.write(f"Test R2: {test_r2:.6f}\n")
        f.write(f"Explained Variance: {test_ev:.6f}\n")
        f.write(f"Intercept: {intercept:.6f}\n")
        f.write("Coefficients:\n")
        for name, w in zip(X.columns, coefs):
            f.write(f"  {name}: {w:.6f}\n")

    # Plots
    # 1) Learning curve if available (sklearn exposes average_loss_ only when early stopping used,
    # so we will not rely on it. Instead, plot parity and residuals and coefficients.)
    plot_parity(y_test.to_numpy(), y_test_pred, PLOTS_DIR / "part2_parity.png")
    plot_residuals(y_test.to_numpy(), y_test_pred, PLOTS_DIR / "part2_residuals.png")
    plot_coefficients(X.columns.tolist(), coefs, PLOTS_DIR / "part2_coefficients.png")

    print("Part 2 complete.")
    print(f"Logs: {log_path.resolve()}")
    print(f"Metrics: {metrics_path.resolve()}")
    print(f"Plots saved to: {PLOTS_DIR.resolve()}")

if __name__ == "__main__":
    main()
