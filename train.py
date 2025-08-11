# train.py - Buying Power Prediction +
# multi-step future forecast
# Uses: scikit-learn LinearRegression on
# percent change targets
# Author: Daniel Aklilu


#Imports
from pathlib import Path
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
# ML imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# CSV file location
DATA_PATH = r"data/cleaned_buying_power.csv"
# Name of date column
DATE_COL = "DATE"
# Name of numeric value column
VALUE_COL = "VALUE"
# Number of past months to use as features
N_LAGS = 12
# Number of past months to use as features
FUTURE_MONTHS = 12


#IS_CLASSIFICATION = False
#TARGET_MODE = "direction"
#TARGET_COL = "LABEL"
#RANDOM_SEED = 42

def load_series(path: str, date_col: str, value_col: str) -> pd.DataFrame:
    p = Path(path)
    print("Loading:", p.resolve())
    if not p.exists():
        raise FileNotFoundError(p.resolve())

    df = pd.read_csv(p)

    # Normalize headers: strip spaces and uppercase
    orig_cols = df.columns.tolist()
    df.columns = df.columns.str.strip().str.upper()

    # Map requested names (DATE, VALUE) to normalized
    date_col_n = date_col.strip().upper()
    value_col_n = value_col.strip().upper()

    # Debug: show what we actually have
    print("Columns found:", df.columns.tolist())

    if date_col_n not in df.columns or value_col_n not in df.columns:
        raise KeyError(
            f"Expected columns '{date_col}' and '{value_col}' not found. "
            f"Available: {orig_cols}")

    # keep only needed cols and tidy
    df = df[[date_col_n, value_col_n]].dropna()
    df = df.sort_values(date_col_n)
    df[date_col_n] = pd.to_datetime(df[date_col_n], errors="coerce")

    # enforce monthly freq & fill
    df = df.set_index(date_col_n).asfreq("MS")
    df[value_col_n] = df[value_col_n].interpolate(limit_direction="both")
    df = df.reset_index()

    # rename back to canonical names used downstream
    df = df.rename(columns={date_col_n: DATE_COL, value_col_n: VALUE_COL})
    return df

def make_lags(df: pd.DataFrame, target_col: str, n_lags: int) -> pd.DataFrame:
    out = df.copy()
    for k in range(1, n_lags + 1):
        out[f"lag_{k}"] = out[target_col].shift(k)
    return out.dropna().reset_index(drop=True)

def split_time(X, y, dates, train=0.70, val=0.15):
    X = np.asarray(X);
    y = np.asarray(y);
    dates = np.asarray(dates)

    #Compute sizes
    n = len(X)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    n_test = n - n_train - n_val
    #Ensure at least 1 test sample
    if n_test < 1:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1
    # Slice datasets
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_va, y_va = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_te, y_te = X[n_train + n_val:], y[n_train + n_val:]
    dates_te = dates[n_train + n_val:]
    print("sizes -> train:", len(X_tr), "val:", len(X_va), "test:", len(X_te))
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), dates_te
def forecast_future_levels(df, model, scaler, n_lags, months, date_col: str,value_col:str):

    assert isinstance(date_col, str) and isinstance(value_col, str),\
        f"date_col/value_col must be strings, got {type(date_col)}{type(value_col)}"

    #Recursive multi-step forecast using percent-change model
    s = df.set_index(date_col)[value_col]
    s = s.asfreq("MS").interpolate(limit_direction="both")
    history = s.iloc[-n_lags:].astype(float).copy()
    last_date = s.index[-1]
    current = history.iloc[-1]

    future_dates, future_levels = [], []
    #Loop forward month-by-month
    for step in range(months):
        lags = np.array(history.iloc[-n_lags:][::-1],
        dtype= np.float32).reshape(1,-1)
        # Scale and predict percent change
        X_one = scaler.transform(lags)
        pct_pred = float(model.predict(X_one)[0])
        # Apply percent change to get next month's value
        next_level = float(current * (1.0 + pct_pred))
        next_date = (last_date + relativedelta(months=step+1)).to_pydatetime()
        # Store forecasted date + value
        future_dates.append(next_date)
        future_levels.append(next_level)
        #Append to history for next prediction step
        history = pd.concat([history,pd.Series([next_level])])
        current = next_level
        # Return DataFrame with forecast results
    return pd.DataFrame({date_col: future_dates, "PREDICTED_VALUE": future_levels})

#Regression pipeline
def run_regression(df: pd.DataFrame, date_col:str, value_col:str):
    # predict next month given past N_LAGS of VALUE
    df = df.copy()
    # Shift value column by -1 month to get "next month" value
    df["y_next"] = df[value_col].shift(-1)
    df = df.dropna()

    # Create lag features for current values
    df_lag = make_lags(df[[date_col, value_col, "y_next"]],value_col,N_LAGS)
    FEATURES = [f"lag_{k}" for k in range(1, N_LAGS + 1)]

    # ---- Target = percent change ----
    df_lag["pct_change"] = (df_lag["y_next"] - df_lag[value_col]) / df_lag[value_col]
    # Convert to numpy arrays
    X = df_lag[FEATURES].values.astype("float32")
    y = df_lag["pct_change"].values.astype("float32")
    dates = df_lag[date_col].values
    last_current = df_lag[value_col].values  # to reconstruct actual values

    # Time-based split
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), dates_te = split_time(X, y, dates)
    curr_te = last_current[-len(y_te):]  # align current values for reconstruction

    # Scale features
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    # Train model (can swap for MLPRegressor)
    model = LinearRegression()
    X_train_full = np.vstack([X_tr, X_va])
    y_train_full = np.concatenate([y_tr, y_va])
    print("Training LinearRegression on percent change…")
    model.fit(X_train_full, y_train_full)

    # Predict percent change
    pct_pred = model.predict(X_te).astype(float).reshape(-1)

    # Reconstruct predicted actual values
    y_level_pred = curr_te * (1.0 + pct_pred)
    true_next_level = curr_te * (1.0 + y_te)

    # MAE on level predictions
    # the lower the number the less it makes mistakes
    mae = np.mean(np.abs(true_next_level - y_level_pred))
    print(f"Test MAE (level): {mae:.4f}")

    # Plot avtual vs predicted for test set
    plt.figure(figsize=(10, 4))
    plt.plot(dates_te, true_next_level, label="Actual")
    plt.plot(dates_te, y_level_pred, label="Predicted")
    plt.title("Buying Power — Actual vs Predicted (Test)")
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pred_vs_actual.png")
    print("Saved plot -> pred_vs_actual.png")

    #---- Multi-step forcast ---
    future = forecast_future_levels(df,model, scaler,N_LAGS,
    FUTURE_MONTHS, date_col ,value_col)
    future.to_csv("future_forecast.csv", index= False)
    print("Saved future forecast -> future_forecast.csv")

    # Plot last 24 months + forecast together
    hist = df[[date_col, value_col]].copy()
    hist[date_col] = pd.to_datetime(hist[date_col])
    hist = hist.sort_values(date_col).tail(24) #Last 24 months

    plt.figure(figsize=(10,4))
    plt.plot(hist[date_col],hist[value_col], label="History (last 24m)")
    plt.plot(future[date_col], future["PREDICTED_VALUE"],
    label=f"Forecast (+{FUTURE_MONTHS}M")
    plt.title(f"Buying Power - {FUTURE_MONTHS}-Month Forecast")
    plt.tight_layout()
    plt.savefig("future_forecast.png")
    print("Saved plot -> future_forecast.png")

def main():
    # 1. Load the cleaned buying power data.
    # 2. Run the regression and forecasting pipeline.
     df = load_series(DATA_PATH, DATE_COL, VALUE_COL)
     run_regression(df, DATE_COL,VALUE_COL)
     print("DEBUG names -> ", DATE_COL,VALUE_COL, type(DATE_COL),
     type(VALUE_COL))





if __name__ == "__main__":
    print(">> starting main()")
    main()

