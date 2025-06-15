import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import optuna
import joblib


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = df['Date'].replace(['N/A', '', 'unknown', 'null', 'nan'], np.nan)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', dayfirst=True, errors='coerce')
    df['Date'] = df['Date'].fillna(method='ffill')
    df['Date'] = df['Date'].fillna(pd.to_datetime('2023-01-01'))
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    holidays = [
        '2023-01-01', '2023-02-20', '2023-04-07', '2023-05-22', '2023-07-01',
        '2023-08-07', '2023-09-04', '2023-10-09', '2023-11-11', '2023-12-25',
        '2024-01-01', '2024-02-19', '2024-03-29', '2024-05-20', '2024-07-01',
        '2024-08-05', '2024-09-02', '2024-10-14', '2024-11-11', '2024-12-25',
        '2025-01-01', '2025-02-17', '2025-04-18', '2025-05-19', '2025-07-01',
        '2025-08-04', '2025-09-01', '2025-10-13', '2025-11-11', '2025-12-25'
    ]
    df['Is_Holiday'] = df['Date'].isin(pd.to_datetime(holidays)).astype(int)
    df = df.sort_values('Date')
    df['Prev_Day_Sales'] = df['Total_Sales'].shift(1).fillna(df['Total_Sales'].mean())
    df['Prev_Week_Sales'] = df['Total_Sales'].shift(7).fillna(df['Total_Sales'].mean())
    df['Prev_Three_Week_Sales'] = df['Total_Sales'].shift(21).fillna(df['Total_Sales'].mean())
    df['Big_Mac_Ratio'] = df['Big_Mac_Sales'] / (df['Total_Sales'] + 1e-6)
    df['Weather'] = df['Weather'].astype('category')
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    weather_encoded = encoder.fit_transform(df[['Weather']])
    weather_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(['Weather']), index=df.index)
    for col in ['Weather_Sunny', 'Weather_Rainy', 'Weather_Snowy']:
        df[f'Temp_{col}'] = df['Temperature_Celsius'] * weather_df.get(col, 0)
    columns_to_drop = ['Weather', 'Date', 'Temperature_Celsius']
    if 'Weather_Cloudy' in df.columns:
        columns_to_drop.append('Weather_Cloudy')
    df = pd.concat([df.drop(columns_to_drop, axis=1), weather_df], axis=1)
    Q1 = df['Total_Sales'].quantile(0.25)
    Q3 = df['Total_Sales'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['Total_Sales'] < (Q1 - 1.5 * IQR)) | (df['Total_Sales'] > (Q3 + 1.5 * IQR)))]
    X = df.drop('Total_Sales', axis=1)
    y = df['Total_Sales']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, df.index.to_series(), X.columns


def objective(trial, X_train, y_train, X_test, y_test):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2.0)
    }
    model = xgb.XGBRegressor(**param, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)


def train_and_evaluate_model(X, y, feature_names, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    train_idx = y.index[:split_index]
    test_idx = y.index[split_index:]
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=30)
    estimators = [
        ('xgb', xgb.XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)),
        ('lr', LinearRegression())
    ]
    model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    if hasattr(model.estimators_[0], 'feature_importances_'):
        feature_importance = pd.Series(model.estimators_[0].feature_importances_, index=feature_names).sort_values(ascending=False)
    else:
        feature_importance = pd.Series(index=feature_names).fillna(0)
    return model, y_pred, y_test, test_idx, metrics, feature_importance


def export_predictions(dates, y_test, y_pred, test_idx):
    results_df = pd.DataFrame({
        "Date": dates.loc[test_idx].values,
        "Actual_Sales": y_test.values,
        "Predicted_Sales": y_pred
    })
    results_df.to_csv("sales_predictions_results.csv", index=False)
    print("Predictions exported to 'sales_predictions_results.csv'")


def main():
    file_path = "McDonalds_Canada_Sales_Data_with_Temperature.csv"
    X_scaled, y, dates, feature_names = load_and_preprocess_data(file_path)
    model, y_pred, y_test, test_idx, metrics, feature_importance = train_and_evaluate_model(X_scaled, y, feature_names)
    print("\nTest Set Metrics:")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"R²: {metrics['r2']:.2f}")
    export_predictions(dates, y_test, y_pred, test_idx)
    joblib.dump(model, "stacked_sales_model.pkl")
    print("Model saved to 'stacked_sales_model.pkl'")


if __name__ == "__main__":
    main()
