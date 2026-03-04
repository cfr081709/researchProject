import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# === Constants & utilities ===
scaler = MinMaxScaler(feature_range=(0, 1))

# load the data file once so helper classes can use it
_data_path = r'C:/Users/Owner/researchProject/Backtests/backtestingData.csv'
try:
    dataFile = pd.read_csv(_data_path, parse_dates=['Date'])
except Exception:
    dataFile = pd.DataFrame()

class dataOrganization:
    """Helpers that return scaled price arrays from `dataFile`."""
    def _scale_column(self, column_name):
        arr = dataFile[column_name].values.reshape(-1, 1)
        return scaler.fit_transform(arr)

    def organizeClosePrices(self):
        return self._scale_column('Close')

    def organizeOpenPrices(self):
        return self._scale_column('Open')

    def organizeHighPrices(self):
        return self._scale_column('High')

    def organizeLowPrices(self):
        return self._scale_column('Low')

    def organizeData(self):
        return (
            self.organizeLowPrices(),
            self.organizeHighPrices(),
            self.organizeOpenPrices(),
            self.organizeClosePrices(),
        )

class LSTMModel:
    """Lightweight wrapper for an LSTM-based price predictor.

    Add date filtering to support running a backtest over a specific
    window (e.g. 2000‑01‑01 through 2026‑01‑01).
    """

    def __init__(self, df: pd.DataFrame):
        # require Date, Ticker, Close columns at minimum
        self.raw = df.copy()
        self.data = df.copy()

    def filter_date_range(self, start_date=None, end_date=None):
        """Restrict `self.data` to the specified inclusive window."""
        if start_date is not None:
            self.data = self.data[self.data['Date'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            self.data = self.data[self.data['Date'] <= pd.to_datetime(end_date)]
        return self.data

    @staticmethod
    def createSequences(series: np.ndarray, time_step: int):
        """Convert a (n,1) array into LSTM training sequences."""
        x, y = [], []
        for i in range(len(series) - time_step):
            x.append(series[i : i + time_step, 0])
            y.append(series[i + time_step, 0])
        x = np.array(x).reshape(-1, time_step, 1)
        y = np.array(y)
        return x, y

    def prepare_dataset(self, ticker=None, time_step=60):
        """Return sequences for a ticker after applying any date filter."""
        df = self.data
        if ticker is not None:
            df = df[df['Ticker'] == ticker]
        df = df.sort_values('Date').reset_index(drop=True)
        close = df['Close'].values.reshape(-1, 1)
        scaled = scaler.fit_transform(close)
        return self.createSequences(scaled, time_step)

    def train_and_backtest(self, test_ratio=0.2, time_step=60,
                           epochs=10, batch_size=32,
                           start_date=None, end_date=None):
        """Train separate LSTM for each ticker and backtest.

        Filters data by date range, then for each ticker:
          * scales Close prices
          * creates time_step sequences
          * splits into train/test using test_ratio
          * fits a simple 1-layer LSTM
          * predicts on test set
          * computes MAE/RMSE + Sharpe/drawdown on actual/predicted
        Saves combined predictions and metrics to CSV/XLSX.
        Returns (preds_path, metrics_path, metrics_df)
        """
        # apply date filtering on the dataset copy
        if start_date is not None or end_date is not None:
            self.filter_date_range(start_date, end_date)
        all_preds = []
        metrics = []

        def price_metrics(prices):
            returns = pd.Series(prices).pct_change().dropna()
            if returns.empty or returns.std() == 0:
                return np.nan, np.nan
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
            cum = (1 + returns).cumprod()
            dd = cum / cum.cummax() - 1
            max_dd = dd.min()
            return sharpe, max_dd

        for ticker, grp in self.data.groupby('Ticker'):
            grp = grp.sort_values('Date').reset_index(drop=True)
            if len(grp) < time_step + 5:
                continue
            close = grp['Close'].values.reshape(-1, 1)
            scaled = scaler.fit_transform(close)
            X, y = self.createSequences(scaled, time_step)
            n = len(X)
            split = int(np.ceil(n * (1 - test_ratio)))
            if split >= n:
                split = n - 1
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            # build LSTM model
            tf.keras.backend.clear_session()
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(50, input_shape=(time_step, 1)),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            preds = model.predict(X_test).flatten()
            # unscale predictions and actuals
            preds_un = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            actual_un = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            dates = grp['Date'].iloc[time_step + split:].reset_index(drop=True)

            out_df = pd.DataFrame({'Date': dates, 'Actual': actual_un})
            out_df['Ticker'] = ticker
            out_df['Pred_LSTM'] = preds_un

            mae = np.mean(np.abs(actual_un - preds_un))
            rmse = np.sqrt(np.mean((actual_un - preds_un) ** 2))
            sharpe_act, dd_act = price_metrics(out_df['Actual'])
            sharpe_pred, dd_pred = price_metrics(out_df['Pred_LSTM'])

            metrics.append({
                'Ticker': ticker,
                'N_train': len(X_train), 'N_test': len(X_test),
                'MAE_LSTM': mae, 'RMSE_LSTM': rmse,
                'Sharpe_Actual': sharpe_act, 'MaxDD_Actual': dd_act,
                'Sharpe_LSTM': sharpe_pred, 'MaxDD_LSTM': dd_pred,
            })

            all_preds.append(out_df)

        if len(all_preds) > 0:
            combined = pd.concat(all_preds, ignore_index=True)
            preds_out = Path('lstm_predictions.csv')
            combined.to_csv(preds_out, index=False)
            combined.to_excel('lstm_predictions.xlsx', index=False)
        else:
            preds_out = None

        if len(metrics) > 0:
            metrics_df = pd.DataFrame(metrics).sort_values('Ticker').reset_index(drop=True)
            metrics_out = Path('lstm_metrics.csv')
            metrics_df.to_csv(metrics_out, index=False)
            metrics_df.to_excel('lstm_metrics.xlsx', index=False)
        else:
            metrics_df = pd.DataFrame()
            metrics_out = None

        return preds_out, metrics_out, metrics_df

# quick example when run as a script
if __name__ == '__main__':
    model = LSTMModel(dataFile)
    model.filter_date_range('2000-01-01', '2026-01-01')
    X, y = model.prepare_dataset(ticker='AAPL', time_step=60)
    print('rows after filter', len(model.data), 'sequence count', len(X))
    # run a quick backtest/train for everyone
    preds_path, metrics_path, mdf = model.train_and_backtest(test_ratio=0.2,
                                                             time_step=60,
                                                             epochs=5,
                                                             batch_size=32,
                                                             start_date='2000-01-01',
                                                             end_date='2026-01-01')
    print('LSTM backtest outputs', preds_path, metrics_path)
