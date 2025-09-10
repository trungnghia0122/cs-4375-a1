# part1.py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # for headless runs
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo


# ---------------------------
# Utilities
# ---------------------------
PLOTS_DIR = Path("plots")
REPORTS_DIR = Path("reports")
LOGS_DIR = Path("logs")
for d in [PLOTS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

def mse(y_true, y_pred):
    e = y_pred - y_true
    return float(np.mean(e * e))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot

def explained_variance(y_true, y_pred):
    var_res = np.var(y_true - y_pred)
    var_y = np.var(y_true)
    return 1.0 - var_res / var_y


# ---------------------------
# Data split and scaling
# ---------------------------
def train_test_split_df(X_df, y_ser, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X_df)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return (
        X_df.iloc[train_idx].reset_index(drop=True),
        X_df.iloc[test_idx].reset_index(drop=True),
        y_ser.iloc[train_idx].reset_index(drop=True),
        y_ser.iloc[test_idx].reset_index(drop=True),
    )

class StandardScalerFromScratch:
    def __init__(self):
        self.mu_ = None
        self.sigma_ = None
    def fit(self, X_df):
        self.mu_ = X_df.mean(axis=0)
        self.sigma_ = X_df.std(axis=0).replace(0, 1.0)
        return self
    def transform(self, X_df):
        return (X_df - self.mu_) / self.sigma_
    def fit_transform(self, X_df):
        return self.fit(X_df).transform(X_df)


# ---------------------------
# Linear Regression (GD)
# ---------------------------
class LinearRegressionGD:
    def __init__(self, lr=1e-3, max_iters=5000, tol=1e-8, patience=100, seed=0):
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol
        self.patience = patience
        self.seed = seed
        self.w_ = None
        self.history_ = []

    @staticmethod
    def _add_bias(X):
        return np.c_[np.ones((len(X), 1)), X]

    def fit(self, X_df, y_ser, verbose=False):
        X = X_df.to_numpy(dtype=float)
        y = y_ser.to_numpy(dtype=float).reshape(-1)
        Xb = self._add_bias(X)

        rng = np.random.default_rng(self.seed)
        self.w_ = rng.normal(0.0, 0.01, size=Xb.shape[1])
        self.history_.clear()

        best_w = self.w_.copy()
        best_mse = np.inf
        wait = 0

        for t in range(self.max_iters):
            y_pred = Xb @ self.w_
            err = y_pred - y
            grad = (2.0 / len(Xb)) * (Xb.T @ err)
            self.w_ = self.w_ - self.lr * grad
            cur_mse = mse(y, y_pred)
            self.history_.append(cur_mse)

            if cur_mse + self.tol < best_mse:
                best_mse = cur_mse
                best_w = self.w_.copy()
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    if verbose:
                        print(f"Early stop at iter {t+1} with best train MSE {best_mse:.6f}")
                    break

        self.w_ = best_w
        return self

    def predict(self, X_df):
        X = X_df.to_numpy(dtype=float)
        Xb = self._add_bias(X)
        return Xb @ self.w_

    @property
    def coef_(self):
        if self.w_ is None:
            return None
        return self.w_[1:]

    @property
    def intercept_(self):
        if self.w_ is None:
            return None
        return float(self.w_[0])


# ---------------------------
# Plots
# ---------------------------
def plot_learning_curve(history, out_path):
    plt.figure()
    plt.plot(np.arange(1, len(history) + 1), history)
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("Learning curve (MSE vs iterations)")
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

def plot_top_correlations(X_df, y_ser, k, out_dir):
    corrs = X_df.corrwith(y_ser).abs().sort_values(ascending=False)
    top = corrs.head(k).index.tolist()
    for col in top:
        plt.figure()
        plt.scatter(X_df[col], y_ser, s=10, alpha=0.5)
        plt.xlabel(col)
        plt.ylabel("quality")
        plt.title(f"quality vs {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_quality_vs_{col}.png", dpi=150)
        plt.close()


# ---------------------------
# Training orchestration
# ---------------------------
def main():
    # Load data
    ds = fetch_ucirepo(id=186)
    X = ds.data.features.select_dtypes(include=[np.number]).copy()
    y = ds.data.targets.squeeze()

    # Splits: train, test then a validation slice from train
    X_train, X_test, y_train, y_test = train_test_split_df(X, y, test_size=0.2, seed=42)
    X_tr, X_val, y_tr, y_val = train_test_split_df(X_train, y_train, test_size=0.1, seed=123)

    # Scale
    scaler = StandardScalerFromScratch()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Hyperparameters to try
    grid_lr = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    grid_iters = [2000, 5000, 10000]
    tol = 1e-8
    patience = 150

    # Log file
    log_path = LOGS_DIR / "part1_tuning_log.csv"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("trial,lr,max_iters,tol,patience,seed,train_mse,val_mse\n")

    best = {"val_mse": np.inf, "lr": None, "max_iters": None, "w": None, "history": None}
    trial = 0

    for lr in grid_lr:
        for iters in grid_iters:
            trial += 1
            model = LinearRegressionGD(lr=lr, max_iters=iters, tol=tol, patience=patience, seed=0)
            model.fit(X_tr_s, y_tr)
            y_tr_pred = model.predict(X_tr_s)
            y_val_pred = model.predict(X_val_s)
            tr_mse = mse(y_tr.to_numpy(), y_tr_pred)
            v_mse = mse(y_val.to_numpy(), y_val_pred)

            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"{trial},{lr},{iters},{tol},{patience},0,{tr_mse:.6f},{v_mse:.6f}\n")

            if v_mse < best["val_mse"]:
                best.update({"val_mse": v_mse, "lr": lr, "max_iters": iters, "w": model.w_.copy(), "history": model.history_.copy()})

    # Retrain on train+val, then evaluate on test
    X_full = pd.concat([X_tr, X_val], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_tr, y_val], axis=0).reset_index(drop=True)
    scaler_full = StandardScalerFromScratch()
    X_full_s = scaler_full.fit_transform(X_full)
    X_test_s_final = scaler_full.transform(X_test)

    final = LinearRegressionGD(lr=best["lr"], max_iters=best["max_iters"], tol=tol, patience=patience, seed=0)
    final.fit(X_full_s, y_full)

    y_test_pred = final.predict(X_test_s_final)
    test_mse = mse(y_test.to_numpy(), y_test_pred)
    test_r2 = r2_score(y_test.to_numpy(), y_test_pred)
    test_ev = explained_variance(y_test.to_numpy(), y_test_pred)

    # Save metrics and coefficients
    metrics_path = REPORTS_DIR / "part1_final_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"Best params: lr={best['lr']}, max_iters={best['max_iters']}\n")
        f.write(f"Test MSE: {test_mse:.6f}\n")
        f.write(f"Test R2: {test_r2:.6f}\n")
        f.write(f"Explained Variance: {test_ev:.6f}\n")
        f.write("Intercept: " + str(final.intercept_) + "\n")
        f.write("Coefficients:\n")
        for name, w in zip(X.columns, final.coef_):
            f.write(f"  {name}: {w:.6f}\n")

    # Plots
    plot_learning_curve(best["history"], PLOTS_DIR / "part1_learning_curve.png")
    plot_parity(y_test.to_numpy(), y_test_pred, PLOTS_DIR / "part1_parity.png")
    plot_residuals(y_test.to_numpy(), y_test_pred, PLOTS_DIR / "part1_residuals.png")
    plot_coefficients(X.columns.tolist(), final.coef_, PLOTS_DIR / "part1_coefficients.png")
    plot_top_correlations(X, y, k=4, out_dir=PLOTS_DIR)

    print("Part 1 complete.")
    print(f"Logs: {log_path.resolve()}")
    print(f"Metrics: {metrics_path.resolve()}")
    print(f"Plots saved to: {PLOTS_DIR.resolve()}")

if __name__ == "__main__":
    main()
