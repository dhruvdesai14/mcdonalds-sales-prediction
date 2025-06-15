# 🍟 McDonald's Sales Prediction – Canada

This project predicts daily sales for McDonald's Canada using historical data, including temperature, weather, holidays, and sales trends. It reflects my journey from working as a Crew Member at McDonald's to becoming a data analyst by applying machine learning and data science skills.

## 🧠 Project Highlights

- Cleans and preprocesses messy date and weather data
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

## 📊 Output

- `sales_predictions_results.csv`: Contains actual vs predicted sales for test data
- `stacked_sales_model.pkl`: Saved trained model

## 📈 Model Metrics

After training:
- **MAE**, **MSE**, and **R²** metrics are printed for evaluation.
- Feature importance is computed using the XGBoost base estimator.

## 🧑‍💼 About Me

Hi, I’m Dhruv Desai — currently working at McDonald’s and transitioning into a Data Analyst role. This project represents my passion for learning and applying machine learning to real-world problems.

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

## 🧪 Sample Output (Console)

```text
[I 2025-06-15 16:50:42,463] A new study created in memory...
...
[I 2025-06-15 16:51:07,575] Trial 29 finished with value: 262.771...

Test Set Metrics:
MAE: 142.83
MSE: 35292.58
R²: 0.99
Predictions exported to 'sales_predictions_results.csv'
Model saved to 'stacked_sales_model.pkl'
```

> Optuna performed 30 trials to optimize XGBoost parameters. The best trial achieved an MAE of 200.21 during tuning. The final model yielded an MAE of 142.83 and an R² of 0.99 on the test set.
