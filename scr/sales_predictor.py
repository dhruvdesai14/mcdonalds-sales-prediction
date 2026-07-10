import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna
import joblib

RANDOM_STATE = 42

HOLIDAYS = pd.to_datetime([
    "2023-01-01", "2023-02-20", "2023-04-07", "2023-05-22", "2023-07-01",
    "2023-08-07", "2023-09-04", "2023-10-09", "2023-11-11", "2023-12-25",
    "2024-01-01", "2024-02-19", "2024-03-29", "2024-05-20", "2024-07-01",
    "2024-08-05", "2024-09-02", "2024-10-14", "2024-11-11", "2024-12-25",
    "2025-01-01", "2025-02-17", "2025-04-18", "2025-05-19", "2025-07-01",
    "2025-08-04", "2025-09-01", "2025-10-13", "2025-11-11", "2025-12-25",
])


def load_and_engineer(file_path):
    """Load the CSV, engineer features, and return the DataFrame + feature list."""
    df = pd.read_csv(file_path)

    #-- Clean and parse the date column ---------------------------------------
    df["Date"] = df["Date"].replace(["N/A", "", "unknown", "null", "nan"], np.nan)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", dayfirst=True, errors="coerce")
    df["Date"] = df["Date"].ffill()
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    #-- Detect same-day component columns (lagged, never used raw) -------------
    component_cols = [c for c in df.columns
                      if c.endswith("_Sales") and c != "Total_Sales"]
    print(f"Detected {len(component_cols)} same-day component column(s) "
          f"(lagged, never used raw): {component_cols}")

    #--- Calendar features (known in advance -> no leakage) --------------------
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Is_Holiday"] = df["Date"].isin(HOLIDAYS).astype(int)

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
    df["Day_Before_Holiday"] = (df["Date"] + pd.Timedelta(days=1)).isin(HOLIDAYS).astype(int)
    df["Day_After_Holiday"] = (df["Date"] - pd.Timedelta(days=1)).isin(HOLIDAYS).astype(int)

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


def remove_training_outliers(X_train, y_train, target_iqr_multiplier=1.5):
    # Remove outliers from the training target using the IQR method.
    q1, q3 = y_train.quantile(0.25), y_train.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - target_iqr_multiplier * iqr, q3 + target_iqr_multiplier * iqr
    keep = (y_train >= lower) & (y_train <= upper)
    return X_train[keep], y_train[keep]


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
    study = optuna.create_study(direction="minimize")
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


def export_predictions(dates_test, y_test, preds, path="sales_predictions_results.csv"):
    out = pd.DataFrame({
        "Date": dates_test.values,
        "Actual_Sales": y_test.values,
        "Predicted_Sales": preds,
    })
    out.to_csv(path, index=False)
    print(f"Predictions exported to '{path}'")


def naive_baselines(X_test, y_test):
    ## Compute naive baseline predictions using lagged features and return their metrics.
    results = {}
    if "Prev_Day_Sales" in X_test.columns:
        p = X_test["Prev_Day_Sales"]
        results["yesterday"] = {"mae": mean_absolute_error(y_test, p),
                                "r2": r2_score(y_test, p)}
    if "Prev_Week_Sales" in X_test.columns:
        p = X_test["Prev_Week_Sales"]
        results["same_weekday_last_week"] = {"mae": mean_absolute_error(y_test, p),
                                             "r2": r2_score(y_test, p)}
    return results


def main():
    file_path = "McDonalds_Canada_Sales_Data_with_Temperature.csv"

    df, feature_cols = load_and_engineer(file_path)
    X_train, y_train, X_test, y_test, dates_test = chronological_split(df, feature_cols)
    X_train, y_train = remove_training_outliers(X_train, y_train)

    print("Tuning hyperparameters (Optuna, internal validation split)...")
    best_params = tune_hyperparameters(X_train, y_train, n_trials=30)

    pipe = train_final_model(X_train, y_train, best_params)
    preds, metrics = evaluate(pipe, X_test, y_test)

    print("\nTest Set Metrics (honest, leakage-free):")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"R^2: {metrics['r2']:.3f}")

    #-- Compare against naive baselines (yesterday's sales, same weekday last week) --
    baselines = naive_baselines(X_test, y_test)
    print("\nNaive baselines (for comparison):")
    best_baseline_mae = float("inf")
    for name, m in baselines.items():
        print(f"  {name:<24} MAE: {m['mae']:.2f}  R^2: {m['r2']:.3f}")
        best_baseline_mae = min(best_baseline_mae, m["mae"])
    if baselines:
        if metrics["mae"] < best_baseline_mae:
            gain = 100 * (best_baseline_mae - metrics["mae"]) / best_baseline_mae
            print(f"\nVerdict: model beats the best naive baseline by {gain:.1f}% on MAE.")
        else:
            print("\nVerdict: model does NOT beat the best naive baseline on MAE. "
                  "The trivial rule is as good or better here.")

    print("\nTop feature importances:")
    print(get_feature_importance(pipe, feature_cols).head(10).to_string())

    export_predictions(dates_test, y_test, preds)
    joblib.dump(pipe, "stacked_sales_model.pkl")
    print("Model saved to 'stacked_sales_model.pkl'")


if __name__ == "__main__":
    main()
