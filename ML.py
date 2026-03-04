import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# new imports for ML/backtest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_ml_backtest(df,
                    test_ratio=0.2,
                    start_date=None,
                    end_date=None):

    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]

    all_preds = []
    metrics = []

    for ticker, grp in df.groupby('Ticker'):

        grp = grp.sort_values('Date').reset_index(drop=True)

        # === TECHNICAL INDICATORS ===
        grp['EMA_20'] = grp['Close'].ewm(span=20).mean()
        grp['EMA_50'] = grp['Close'].ewm(span=50).mean()

        delta = grp['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        grp['RSI_14'] = 100 - (100 / (1 + rs))

        ema12 = grp['Close'].ewm(span=12).mean()
        ema26 = grp['Close'].ewm(span=26).mean()
        grp['MACD'] = ema12 - ema26

        grp['Return'] = grp['Close'].pct_change()
        grp = grp.dropna().reset_index(drop=True)

        if len(grp) < 100:
            continue

        feature_cols = [
            'Open','High','Low','Close','Volume',
            'EMA_20','EMA_50','RSI_14','MACD'
        ]

        X = grp[feature_cols].values
        y = grp['Return'].values

        split = int(len(X)*(1-test_ratio))

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # === SCALE TRAIN ONLY ===
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # === RANDOM FOREST MODEL ===
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        actual = y_test

        # === SIGNAL GENERATION ===
        signal = np.where(preds > 0, 1, -1)
        strategy_returns = signal * actual

        # === METRICS ===
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)

        cumulative = (1 + strategy_returns).cumprod()
        max_dd = (cumulative / cumulative.cummax() - 1).min()
        cagr = cumulative[-1] ** (252/len(strategy_returns)) - 1

        mae = np.mean(np.abs(actual - preds))
        rmse = np.sqrt(np.mean((actual - preds)**2))

        dates = grp['Date'].iloc[split:].reset_index(drop=True)

        out_df = pd.DataFrame({
            'Date': dates,
            'Ticker': ticker,
            'Actual_Return': actual,
            'Pred_Return': preds,
            'Signal': signal,
            'Strategy_Return': strategy_returns
        })

        metrics.append({
            'Ticker': ticker,
            'MAE': mae,
            'RMSE': rmse,
            'Sharpe': sharpe,
            'Max_Drawdown': max_dd,
            'CAGR': cagr
        })

        all_preds.append(out_df)

    # === SAVE RESULTS ===
    if len(all_preds) > 0:
        combined = pd.concat(all_preds, ignore_index=True)
        combined.to_csv('ml_predictions.csv', index=False)

    if len(metrics) > 0:
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv('ml_metrics.csv', index=False)
    else:
        metrics_df = pd.DataFrame()

    return metrics_df
