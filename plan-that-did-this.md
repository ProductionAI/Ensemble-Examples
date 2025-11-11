# THE PROMPT THAT MADE THIS PLAN

I want a new folder called Module-12 and I want a new jupyter notebook in the new folder called assignment.ipynb. I want this notebook to pull or use mock financial data and then the notebook will combine complementary model families (linear, tree-based, and boosting) to predict short-horizon returns or the probability of an up-move. This can be a basic example. For this to happen want a .venv virtual environment created and activated to install all the required dependencies.

# Module-12: Financial ML Notebook with Ensemble Models

## What we'll add

- `Module-12/` directory
- Python `.venv` virtual environment scoped to `Module-12`
- `requirements.txt` with all deps and kernel setup
- `assignment.ipynb` Jupyter notebook demonstrating:
  - Mock OHLCV data generation (no external calls)
  - Feature engineering (returns, rolling stats, simple technicals)
  - Time-based split (train/valid/test)
  - Three model families:
    - Linear: LogisticRegression (probability of up-move)
    - Tree-based: RandomForestClassifier
    - Boosting: XGBoost (XGBClassifier)
  - Simple ensemble: average of model probabilities; compare ROC-AUC/accuracy
  - Optional: regression target (next return) with LinearRegression/XGBRegressor

## Environment setup

From project root on Windows PowerShell:

```bash
# Create folder
mkdir Module-12
cd Module-12

# Create venv
python -m venv .venv

# Activate
. .venv/Scripts/Activate.ps1

# Requirements file
@"
jupyter
ipykernel
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
"@ | Out-File -Encoding utf8 requirements.txt

# Install
pip install -r requirements.txt

# Register kernel
python -m ipykernel install --user --name module12-venv --display-name "Python (Module-12)"
```

## Notebook outline (`Module-12/assignment.ipynb`)

1. Imports and configuration
2. Generate mock OHLCV time series (random walk + volatility clusters)
3. Feature engineering:

   - Log returns, next-period direction label (up/down)
   - Rolling mean/std, momentum (rolling returns), RSI-like proxy

4. Time-aware split (no leakage)
5. Train models:

   - LogisticRegression (with class_weight='balanced')
   - RandomForestClassifier
   - XGBClassifier (eval metric=auc)

6. Evaluate each model (AUC, accuracy, confusion matrix) on validation/test
7. Ensemble probabilities (mean) → evaluate
8. (Optional) Regression: predict next return; compare RMSE
9. Plots: ROC curves and feature importances (RF/XGB)

## Key code snippets

- Label creation:
```python
returns = np.log(close).diff()
y = (returns.shift(-1) > 0).astype(int)
```

- Time split:
```python
train, valid, test = df.iloc[:n1], df.iloc[n1:n2], df.iloc[n2:]
```

- Ensemble:
```python
p_ens = (p_lr + p_rf + p_xgb) / 3
```


## Run notebook

```bash
cd Module-12
. .venv/Scripts/Activate.ps1
jupyter notebook assignment.ipynb
# In UI, select kernel: Python (Module-12)
```