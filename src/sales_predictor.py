from pathlib import Path

import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna
import joblib

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "McDonalds_Canada_Sales_Data_with_Temperature.csv"
OUTPUT_DIR = ROOT / "output"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def canadian_holidays(years):
    """Ontario public + optional holidays for the given years."""
    cal = holidays.Canada(subdiv="ON", years=years, categories=("public", "optional"))
    return pd.to_datetime(list(cal.keys()))



def load_and_engineer(file_path):
    """Load the CSV, engineer features, and return the DataFrame + feature list."""
    df = pd.read_csv(file_path)

    #-- Clean and parse the date column ---------------------------------------
    df["Date"] = df["Date"].replace(["N/A", "", "unknown", "null", "nan"], np.nan)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", dayfirst=True, errors="coerce")
    df["Date"] = df["Date"].ffill()
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    years = range(df["Date"].dt.year.min() - 1, df["Date"].dt.year.max() + 2)
    holiday_dates = canadian_holidays(years)

    #-- Detect same-day component columns (lagged, never used raw) -------------
    component_cols = [c for c in df.columns
                      if c.endswith("_Sales") and c != "Total_Sales"]
    print(f"Detected {len(component_cols)} same-day component column(s) "
          f"(lagged, never used raw): {component_cols}")

    #--- Calendar features (known in advance -> no leakage) --------------------
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Is_Holiday"] = df["Date"].isin(holiday_dates).astype(int)

    # --- Lag features (all strictly backward-looking) ----------------------
    lag_cols = []
    df["Prev_Day_Sales"] = df["Total_Sales"].shift(1)
    df["Prev_Week_Sales"] = df["Total_Sales"].shift(7)
    df["Prev_Three_Week_Sales"] = df["Total_Sales"].shift(21)
    lag_cols += ["Prev_Day_Sales", "Prev_Week_Sales", "Prev_Three_Week_Sales"]

    # --- Lag features for component columns --------------------------------
    for col in component_cols:
        lag_name = f"Prev_Day_{col}"
        df[lag_name] = df[col].shift(1)
        lag_cols.append(lag_name)

    # --- Rolling averages (strictly backward-looking) ----------------------
    df["Prev_Roll7_Sales"] = df["Total_Sales"].shift(1).rolling(7).mean()
    df["Prev_Roll14_Sales"] = df["Total_Sales"].shift(1).rolling(14).mean()
    df["Prev_2Week_Sales"] = df["Total_Sales"].shift(14)
    lag_cols += ["Prev_Roll7_Sales", "Prev_Roll14_Sales", "Prev_2Week_Sales"]

    # Calendar-derived flags (known in advance -> no leakage).
    df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)
    df["Day_Before_Holiday"] = (df["Date"] + pd.Timedelta(days=1)).isin(holiday_dates).astype(int)
    df["Day_After_Holiday"] = (df["Date"] - pd.Timedelta(days=1)).isin(holiday_dates).astype(int)

    #--- Interaction features (known in advance -> no leakage) ----------------
    if "Local_Event" in df.columns and pd.api.types.is_numeric_dtype(df["Local_Event"]):
        df["Local_Event_x_Weekend"] = df["Local_Event"] * df["Is_Weekend"]

    #--- Weather features (known in advance -> no leakage) --------------------
    weather_dummies = pd.get_dummies(df["Weather"], prefix="Weather").astype(int)
    df = pd.concat([df, weather_dummies], axis=1)
    for col in weather_dummies.columns:
        df[f"Temp_{col}"] = df["Temperature_Celsius"] * df[col]

    #-- Drop rows with NaN in any lagged feature (first few rows) ---------------
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    #-- Final feature list: exclude target, date, weather, temperature, and raw component columns
    exclude = {"Total_Sales", "Date", "Weather", "Temperature_Celsius"}
    exclude.update(component_cols)
    feature_cols = [c for c in df.columns if c not in exclude]

    #-- Sanity check: no same-day sales columns should remain in features
    leaked = [c for c in feature_cols
              if c.endswith("_Sales") and not c.startswith("Prev_")]
    if leaked:
        raise ValueError(f"Same-day sales columns still in features: {leaked}")

    print(f"Final feature count: {len(feature_cols)}")
    return df, feature_cols


def chronological_split(df, feature_cols, target="Total_Sales", test_size=0.2):
    # Split the DataFrame into chronological training and test sets, returning X/y for each.
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    X_train, y_train = train_df[feature_cols], train_df[target]
    X_test, y_test = test_df[feature_cols], test_df[target]
    dates_test = test_df["Date"]
    return X_train, y_train, X_test, y_test, dates_test


def _objective(trial, X_train, y_train, val_fraction=0.2):
    # Split the training data into an internal training and validation set for Optuna hyperparameter tuning.
    inner_split = int(len(X_train) * (1 - val_fraction))
    X_tr, X_val = X_train.iloc[:inner_split], X_train.iloc[inner_split:]
    y_tr, y_val = y_train.iloc[:inner_split], y_train.iloc[inner_split:]

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 2.0),
    }

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", xgb.XGBRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_val)
    return mean_absolute_error(y_val, preds)


def tune_hyperparameters(X_train, y_train, n_trials=30):
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda t: _objective(t, X_train, y_train), n_trials=n_trials)
    return study.best_params


def train_final_model(X_train, y_train, best_params):
    """Fit the stacked model inside a scaling pipeline on the training data."""
    estimators = [
        ("xgb", xgb.XGBRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)),
        ("lr", LinearRegression()),
    ]
    
    #--- Stacking Regressor with XGBoost and Linear Regression as final estimator ---
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=5,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("model", stack)])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate(pipe, X_test, y_test):
    preds = pipe.predict(X_test)
    return preds, {
        "mae": mean_absolute_error(y_test, preds),
        "mse": mean_squared_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }


def get_feature_importance(pipe, feature_cols):
    stack = pipe.named_steps["model"]
    xgb_model = stack.named_estimators_["xgb"]
    return (pd.Series(xgb_model.feature_importances_, index=feature_cols)
            .sort_values(ascending=False))


def export_predictions(dates_test, y_test, preds, path=OUTPUT_DIR / "sales_predictions_results.csv"):
    out = pd.DataFrame({
        "Date": dates_test.values,
        "Actual_Sales": y_test.values,
        "Predicted_Sales": preds,
    })
    out.to_csv(path, index=False)
    print(f"Predictions exported to '{path.relative_to(ROOT)}'")


def plot_predictions(dates_test, y_test, preds, baseline, path=OUTPUT_DIR / "actual_vs_predicted.png"):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates_test, y_test, label="Actual", color="#222222", linewidth=1.4)
    ax.plot(dates_test, preds, label="Stacked model", color="#d62728", linewidth=1.4)
    ax.plot(dates_test, baseline, label="Baseline: mean by Local_Event",
            color="#1f77b4", linewidth=1.2, linestyle="--")
    ax.set_title("Daily sales on the hold-out period: actual vs predicted")
    ax.set_ylabel("Total sales")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Plot saved to '{path.relative_to(ROOT)}'")


def baseline_predictions(X_train, y_train, X_test):
    """Simple rules the model has to beat, keyed by name."""
    preds = {"train_mean": pd.Series(y_train.mean(), index=X_test.index)}
    if "Local_Event" in X_train.columns:
        event_means = y_train.groupby(X_train["Local_Event"]).mean()
        preds["mean_by_local_event"] = (X_test["Local_Event"].map(event_means)
                                        .fillna(y_train.mean()))
    if "Prev_Day_Sales" in X_test.columns:
        preds["yesterday"] = X_test["Prev_Day_Sales"]
    if "Prev_Week_Sales" in X_test.columns:
        preds["same_weekday_last_week"] = X_test["Prev_Week_Sales"]
    return preds


def score_baselines(X_train, y_train, X_test, y_test):
    return {name: {"mae": mean_absolute_error(y_test, p), "r2": r2_score(y_test, p)}
            for name, p in baseline_predictions(X_train, y_train, X_test).items()}


def walk_forward_cv(df, feature_cols, target="Total_Sales", n_splits=5, n_trials=15):
    """Expanding-window CV: tune + fit on the past, score on the next block."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X, y = df[feature_cols], df[target]
    rows = []
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        X_tr, y_tr, X_te, y_te = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
        params = tune_hyperparameters(X_tr, y_tr, n_trials=n_trials)
        preds = train_final_model(X_tr, y_tr, params).predict(X_te)
        row = {"fold": fold, "model": mean_absolute_error(y_te, preds)}
        row.update({name: m["mae"]
                    for name, m in score_baselines(X_tr, y_tr, X_te, y_te).items()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("fold")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df, feature_cols = load_and_engineer(DATA_PATH)
    X_train, y_train, X_test, y_test, dates_test = chronological_split(df, feature_cols)
    print(f"Train: {len(X_train)} days, test: {len(X_test)} days")

    print("Tuning hyperparameters (Optuna, internal validation split)...")
    best_params = tune_hyperparameters(X_train, y_train, n_trials=30)

    pipe = train_final_model(X_train, y_train, best_params)
    preds, metrics = evaluate(pipe, X_test, y_test)

    print("\nTest Set Metrics (honest, leakage-free):")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"R^2: {metrics['r2']:.3f}")

    #-- Compare against baselines: the model has to beat the best of them --
    baselines = score_baselines(X_train, y_train, X_test, y_test)
    print("\nBaselines (for comparison):")
    for name, m in baselines.items():
        print(f"  {name:<24} MAE: {m['mae']:.2f}  R^2: {m['r2']:.3f}")
    best_name = min(baselines, key=lambda n: baselines[n]["mae"])
    best_mae = baselines[best_name]["mae"]
    diff = 100 * (best_mae - metrics["mae"]) / best_mae
    if diff > 0:
        print(f"\nVerdict: model beats the best baseline ({best_name}) by {diff:.1f}% on MAE.")
    else:
        print(f"\nVerdict: model does NOT beat the best baseline ({best_name}); "
              f"it is {-diff:.1f}% worse on MAE.")

    print("\nWalk-forward CV (5 expanding folds, MAE per fold):")
    cv = walk_forward_cv(df, feature_cols)
    print(cv.round(1).to_string())
    print("\nMean MAE across folds (std):")
    for col in cv.columns:
        print(f"  {col:<24} {cv[col].mean():8.1f}  ({cv[col].std():.1f})")

    print("\nTop feature importances:")
    print(get_feature_importance(pipe, feature_cols).head(10).to_string())

    export_predictions(dates_test, y_test, preds)
    event_baseline = baseline_predictions(X_train, y_train, X_test)["mean_by_local_event"]
    plot_predictions(dates_test, y_test, preds, event_baseline)
    model_path = OUTPUT_DIR / "stacked_sales_model.pkl"
    joblib.dump(pipe, model_path)
    print(f"Model saved to '{model_path.relative_to(ROOT)}'")


if __name__ == "__main__":
    main()
