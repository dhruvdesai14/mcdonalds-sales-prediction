# 🍟 McDonald's Sales Prediction Canada

This project predicts daily sales for McDonald's Canada using historical data, including temperature, weather, holidays, and sales trends. It reflects my journey from working as a Crew Member at McDonald's to becoming a data analyst by applying machine learning and data science skills.

## 🧠 Project Highlights

- Cleans and preprocesses messy Date and weather data
- Feature engineering: lag features, holiday flagging, weather influence
- One-hot encoding and scaling
- XGBoost + Linear Regression in a Stacking Regressor
- Hyperparameter tuning using Optuna
- Exporting predictions and the trained model

## 📁 Folder Structure

```
mcdonalds-sales-prediction/
├── src/
│   └── sales_predictor.py        # Main Python script
├── data/
│   └── McDonalds_Canada_Sales_Data_with_Temperature.csv
├── output/
│   ├── sales_predictions_results.csv
│   └── stacked_sales_model.pkl
├── README.md
└── requirements.txt
```

> Note: The dataset file included in the repository tries to mimic real-world sales data for privacy.

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/dhruvdesai14/mcdonalds-sales-prediction.git
   cd mcdonalds-sales-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure the file is in the `data/` folder.

4. Run the script:
   ```bash
   python src/sales_predictor.py
   ```
## Model Performance & Data Integrity

> **Headline:** On a highly volatile daily-sales series where trivial forecasts fail,
> the model reduces prediction error by **~41%** versus naive baselines. Getting to
> an *honest* result meant finding and removing data leakage across three iterations
> which is the part of this project I'm most proud of.

### Results (chronological hold-out, no leakage)

| Model / Baseline                    | MAE     | R²      |
|-------------------------------------|---------|---------|
| **Stacked model (XGBoost + Linear)**| **1236**| **0.087** |
| Naive: same weekday last week       | 2140    | −1.30   |
| Naive: yesterday                    | 2084    | −1.47   |

The model beats the best naive baseline by **40.7% on MAE**. The two carry-forward
Baseline score *negative* R² worse than predicting the daily average, which shows
how little day-to-day persistence this series has. Predicting it at all is genuinely hard.

**Top signal drivers:** local events, weekend effect, weather/temperature interactions,
and 7- and 14-day rolling sales averages.

### Why R² is low and why that's the honest answer

An R² of ~0.09 means the available features explain only a small share of daily
sales variance. That is a real property of the data, not a modelling failure:
Single-location daily sales are dominated by factors that this dataset doesn't contain.
The right way to judge the model here is **against a baseline**, and by that measure,
it clearly adds value; it turns a negative-R² problem positive and cuts error by ~41%.

A high R² on this data would have been a red flag, which is exactly what the earlier
Versions produced.

### The leakage story (three iterations)

| Version | R²    | Cause                                                                 |
|---------|-------|-----------------------------------------------------------------------|
| v1      | 0.99  | A `Big_Mac_Ratio` feature divided by the target, and same-day Big Mac sales were used directly. |
| v2      | 0.865 | Other same-day menu categories (fries, nuggets, desserts, drinks) still leaked — they sum to the target. |
| v3      | 0.087 | All same-day components lagged to previous-day values; honest result. |

Each drop came from removing a feature that secretly contained the answer. `Total_Sales`
is essentially the sum of its menu categories, so **any same-day category value lets the
model reconstruct the total instead of forecasting it.**

### Safeguards in the current pipeline

- **No same-day target components.** Every `*_Sales` column is auto-detected and used
  only as a *previous-day* lag; a guard raises an error if any same-day sales column
  reaches the model.
- **Scaling inside the pipeline.** `StandardScaler` is fit on training folds only, so
  test statistics never leak into the transform.
- **Honest tuning.** Optuna optimises against an internal chronological validation slice
  carved from the training data; the test set is untouched until final evaluation.
- **Time-ordered split.** Train on the past, test on the most recent period.
- **Baseline comparison.** Every run reports naive baselines, so the model has to prove
  it beats a trivial rule.

### Reproduce

```bash
pip install -r requirements.txt
python src/sales_predictor.py
```

## 📊 Output

- `sales_predictions_results.csv`: Contains actual vs predicted sales for test data
- `stacked_sales_model.pkl`: Saved trained model

## 📈 Model Metrics

After training:
- **MAE**, **MSE**, and **R²** metrics are printed for evaluation.
- Feature importance is computed using the XGBoost base estimator.

## 🧑‍💼 About Me

Hi, I’m Dhruv Desai, currently working at McDonald’s and transitioning into a Data Analyst role. This project represents my passion for learning and applying machine learning to real-world problems.

Let's connect on [LinkedIn](https://www.linkedin.com/in/dhruvdesai14)!

---

### 📌 Tech Stack

- Python
- Pandas, NumPy
- Scikit-Learn
- XGBoost
- Optuna
- Matplotlib
- Joblib

---

### 📬 Contact

Feel free to reach out if you're hiring for data analyst roles or have feedback on the project!
