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

class MachineLearningModel:
    def __init__(self, csv_path, start_date=None, end_date=None, out_dir=None):
        self.csv_path = Path(csv_path)
        self.start_date = pd.to_datetime(start_date) if start_date is not None else None
        self.end_date = pd.to_datetime(end_date) if end_date is not None else None
        self.out_dir = Path(out_dir) if out_dir else self.csv_path.with_suffix('').parent / "output_by_ticker"
        self.plots_dir = self.out_dir / "plots_by_ticker"
        self.per_ticker_dir = self.out_dir / "per_ticker_csv"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.per_ticker_dir.mkdir(parents=True, exist_ok=True)
        self.data = self._load_data()

    def _load_data(self):
        # try parsing Date if present, otherwise load without date parsing
        try:
            df = pd.read_csv(self.csv_path, parse_dates=['Date'], dayfirst=False)
        except Exception:
            df = pd.read_csv(self.csv_path)

        # Ensure Ticker present
        if 'Ticker' not in df.columns:
            raise ValueError("CSV must contain a 'Ticker' column. Found columns: " + ", ".join(df.columns))

        # If Date missing, reconstruct per-ticker
        if 'Date' not in df.columns:
            reconstructed = []
            grouped = df.groupby('Ticker', sort=False)
            for ticker, grp in grouped:
                n = len(grp)
                # try business day range if start/end provided
                if self.start_date is not None and self.end_date is not None:
                    rng = pd.bdate_range(self.start_date, self.end_date)
                    if len(rng) >= n:
                        dates = rng[-n:]  # assume CSV is chronological; take last n trading days
                    else:
                        dates = pd.date_range(self.start_date, periods=n, freq='D')
                else:
                    dates = pd.date_range(pd.Timestamp("1970-01-01"), periods=n, freq='D')
                grp = grp.reset_index(drop=True)
                grp['Date'] = dates
                reconstructed.append(grp)
            df = pd.concat(reconstructed, ignore_index=True)
        else:
            # ensure Date is datetime
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # normalize numeric columns
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Drop rows without Ticker or Date
        df = df.dropna(subset=['Ticker', 'Date']).reset_index(drop=True)

        # Add DaysSince (numeric time feature) relative to each ticker's first date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        df['DaysSince'] = df.groupby('Ticker')['Date'].transform(lambda x: (x - x.min()).dt.days.astype(int))

        return df

    def split_save_by_ticker(self, folder_name="per_ticker_csv"):
        dest = self.out_dir / folder_name
        dest.mkdir(parents=True, exist_ok=True)
        saved_files = []
        for ticker, grp in self.data.groupby('Ticker'):
            out_path = dest / f"{ticker}.csv"
            grp.to_csv(out_path, index=False)
            saved_files.append(str(out_path))
        return saved_files

    def get_statistics_by_ticker(self):
        stats = []
        grouped = self.data.groupby('Ticker')
        for ticker, grp in grouped:
            d = {'Ticker': ticker, 'N': len(grp)}
            for col in ['Close', 'Open', 'High', 'Low']:
                if col in grp.columns:
                    s = grp[col].dropna()
                    d[f'{col}_mean'] = s.mean()
                    d[f'{col}_median'] = s.median()
                    modes = s.mode()
                    d[f'{col}_mode'] = modes.iloc[0] if not modes.empty else np.nan
                    d[f'{col}_std'] = s.std()
                    d[f'{col}_var'] = s.var()
                else:
                    d.update({f'{col}_mean': np.nan, f'{col}_median': np.nan,
                              f'{col}_mode': np.nan, f'{col}_std': np.nan, f'{col}_var': np.nan})
            stats.append(d)
        return pd.DataFrame(stats).sort_values('Ticker').reset_index(drop=True)

    def polynomial_regression_and_plot(self, degree=3, save_plots=True):
        results = []
        grouped = self.data.groupby('Ticker')
        for ticker, grp in grouped:
            grp = grp.sort_values('Date').reset_index(drop=True)
            if 'Close' not in grp.columns or grp['Close'].dropna().size < degree + 1:
                continue
            # x: DaysSince, y: Close
            x_num = grp['DaysSince'].values.astype(float)
            y_num = grp['Close'].values.astype(float)
            mask = ~np.isnan(x_num) & ~np.isnan(y_num)
            x_num = x_num[mask]
            y_num = y_num[mask]
            if x_num.size < degree + 1:
                continue
            coeffs = np.polyfit(x_num, y_num, deg=degree)
            model = np.poly1d(coeffs)
            x_line = np.linspace(x_num.min(), x_num.max(), 200)
            y_line = model(x_line)
            date_line = grp['Date'].min() + pd.to_timedelta(x_line, unit='D')

            if save_plots:
                plt.figure(figsize=(8, 4))
                plt.scatter(grp['Date'], grp['Close'], s=8, label='Close', color='tab:blue')
                plt.plot(date_line, y_line, color='tab:red', label=f'Poly deg={degree}')
                plt.title(f'{ticker} Close Price - Polynomial fit (deg={degree})')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.tight_layout()
                out_path = self.plots_dir / f'{ticker}_poly_deg{degree}.png'
                plt.savefig(out_path)
                plt.close()

            results.append({'Ticker': ticker, 'coefficients': coeffs})
        return pd.DataFrame(results)

    # --- New helpers for backtesting --- #
    def _make_features(self, grp):
        # require Date sorted
        grp = grp.sort_values('Date').reset_index(drop=True)
        grp['Lag1_Close'] = grp['Close'].shift(1)
        grp['Lag2_Close'] = grp['Close'].shift(2)
        # use DaysSince, Lag1, Lag2 as features
        features = ['DaysSince', 'Lag1_Close', 'Lag2_Close']
        return grp.dropna(subset=['Close']).reset_index(drop=True), features

    def _time_train_test_split(self, grp, test_ratio=0.2):
        # grp must be sorted by Date
        n = len(grp)
        if n < 3:
            return None, None  # insufficient
        split = int(np.ceil(n * (1 - test_ratio)))
        if split == n:
            split = n - 1
        train = grp.iloc[:split].reset_index(drop=True)
        test = grp.iloc[split:].reset_index(drop=True)
        return train, test

    def backtest_models(self, test_ratio=0.2, degree=3, save_plots=False,
                        start_date=None, end_date=None):
        all_preds = []
        metrics = []
        grouped = self.data.groupby('Ticker')
        for ticker, grp in grouped:
            grp = grp.sort_values('Date').reset_index(drop=True)
            # apply date range filter if requested
            if start_date is not None:
                grp = grp[grp['Date'] >= pd.to_datetime(start_date)]
            if end_date is not None:
                grp = grp[grp['Date'] <= pd.to_datetime(end_date)]
            if grp.empty:
                continue
            grp_feat, feat_cols = self._make_features(grp)
            if grp_feat.shape[0] < 5:
                # skip tickers with too few points
                continue
            train, test = self._time_train_test_split(grp_feat, test_ratio=test_ratio)
            if train is None or test is None or len(test) == 0:
                continue

            # prepare arrays
            X_train_days = train['DaysSince'].values.reshape(-1, 1)
            X_test_days = test['DaysSince'].values.reshape(-1, 1)
            y_train = train['Close'].values
            y_test = test['Close'].values

            # Polynomial (np.polyfit) on DaysSince
            pred_poly = np.full(len(test), np.nan)
            if len(train) >= degree + 1:
                coeffs = np.polyfit(train['DaysSince'].values, y_train, deg=degree)
                poly = np.poly1d(coeffs)
                pred_poly = poly(test['DaysSince'].values)

            # LinearRegression on DaysSince
            lr = LinearRegression()
            lr.fit(X_train_days, y_train)
            pred_lr = lr.predict(X_test_days)

            # Prepare multi-feature arrays for tree models (drop rows with NaNs)
            X_train_multi = train[feat_cols].dropna()
            y_train_multi = train.loc[X_train_multi.index, 'Close']
            X_test_multi = test[feat_cols].reindex(columns=feat_cols)
            # If test has NaNs in features align by dropping those rows (we'll keep index mapping)
            valid_test_mask = ~X_test_multi.isna().any(axis=1)
            X_test_multi_valid = X_test_multi[valid_test_mask]

            pred_rf = np.full(len(test), np.nan)
            pred_gb = np.full(len(test), np.nan)
            if len(X_train_multi) >= 5 and X_test_multi_valid.shape[0] > 0:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train_multi.values, y_train_multi.values)
                preds_rf_valid = rf.predict(X_test_multi_valid.values)
                pred_rf[valid_test_mask.values] = preds_rf_valid

                gb = GradientBoostingRegressor(random_state=42)
                gb.fit(X_train_multi.values, y_train_multi.values)
                preds_gb_valid = gb.predict(X_test_multi_valid.values)
                pred_gb[valid_test_mask.values] = preds_gb_valid

            # Build per-ticker prediction DataFrame
            out_df = test[['Date', 'DaysSince', 'Close']].copy().reset_index(drop=True)
            out_df.rename(columns={'Close': 'Actual'}, inplace=True)
            out_df['Ticker'] = ticker
            out_df['Pred_Poly'] = pred_poly
            out_df['Pred_Linear'] = pred_lr
            out_df['Pred_RF'] = pred_rf
            out_df['Pred_GB'] = pred_gb

            # compute metrics for this ticker for each model (only where actual & pred present)
            def compute_metrics(y_true, y_pred):
                mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
                if mask.sum() == 0:
                    return np.nan, np.nan
                mae = mean_absolute_error(y_true[mask], y_pred[mask])
                rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                return mae, rmse

            mae_poly, rmse_poly = compute_metrics(out_df['Actual'].values, out_df['Pred_Poly'].values)
            mae_lr, rmse_lr = compute_metrics(out_df['Actual'].values, out_df['Pred_Linear'].values)
            mae_rf, rmse_rf = compute_metrics(out_df['Actual'].values, out_df['Pred_RF'].values)
            mae_gb, rmse_gb = compute_metrics(out_df['Actual'].values, out_df['Pred_GB'].values)

            # calculate Sharpe and drawdown for actual and each prediction
            def price_metrics(prices):
                returns = pd.Series(prices).pct_change().dropna()
                if returns.empty or returns.std() == 0:
                    return np.nan, np.nan
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
                cum = (1 + returns).cumprod()
                dd = cum / cum.cummax() - 1
                max_dd = dd.min()
                return sharpe, max_dd

            sharpe_act, dd_act = price_metrics(out_df['Actual'])
            sharpe_poly, dd_poly = price_metrics(out_df['Pred_Poly'])
            sharpe_lr, dd_lr = price_metrics(out_df['Pred_Linear'])
            sharpe_rf, dd_rf = price_metrics(out_df['Pred_RF'])
            sharpe_gb, dd_gb = price_metrics(out_df['Pred_GB'])

            metrics.append({
                'Ticker': ticker,
                'N_train': len(train), 'N_test': len(test),
                'MAE_Poly': mae_poly, 'RMSE_Poly': rmse_poly,
                'MAE_Linear': mae_lr, 'RMSE_Linear': rmse_lr,
                'MAE_RF': mae_rf, 'RMSE_RF': rmse_rf,
                'MAE_GB': mae_gb, 'RMSE_GB': rmse_gb,
                'Sharpe_Actual': sharpe_act, 'MaxDD_Actual': dd_act,
                'Sharpe_Poly': sharpe_poly, 'MaxDD_Poly': dd_poly,
                'Sharpe_Linear': sharpe_lr, 'MaxDD_Linear': dd_lr,
                'Sharpe_RF': sharpe_rf, 'MaxDD_RF': dd_rf,
                'Sharpe_GB': sharpe_gb, 'MaxDD_GB': dd_gb
            })

            # save per-ticker backtest CSV
            per_ticker_out = self.per_ticker_dir / f"backtest_{ticker}.csv"
            out_df.to_csv(per_ticker_out, index=False)

            all_preds.append(out_df)

            # optional: save plots of predictions vs actual for test segment
            if save_plots:
                plt.figure(figsize=(8, 4))
                plt.plot(out_df['Date'], out_df['Actual'], marker='o', label='Actual', color='black')
                plt.plot(out_df['Date'], out_df['Pred_Poly'], marker='.', label='Poly', alpha=0.8)
                plt.plot(out_df['Date'], out_df['Pred_Linear'], marker='.', label='Linear', alpha=0.8)
                plt.plot(out_df['Date'], out_df['Pred_RF'], marker='.', label='RF', alpha=0.8)
                plt.plot(out_df['Date'], out_df['Pred_GB'], marker='.', label='GB', alpha=0.8)
                plt.title(f'Backtest predictions - {ticker}')
                plt.xlabel('Date'); plt.ylabel('Price'); plt.legend(); plt.tight_layout()
                plt_path = self.plots_dir / f'backtest_{ticker}.png'
                plt.savefig(plt_path); plt.close()

        # combine and save all predictions & metrics
        if len(all_preds) > 0:
            combined = pd.concat(all_preds, ignore_index=True)
            preds_out = self.out_dir / f'predictions_backtest.csv'
            combined.to_csv(preds_out, index=False)
            # save to Excel as well
            combined.to_excel(str(self.out_dir / 'predictions_backtest.xlsx'), index=False)
        else:
            preds_out = None

        if len(metrics) > 0:
            metrics_df = pd.DataFrame(metrics).sort_values('Ticker').reset_index(drop=True)
            metrics_out = self.out_dir / 'metrics_summary.csv'
            metrics_df.to_csv(metrics_out, index=False)
            metrics_df.to_excel(str(self.out_dir / 'metrics_summary.xlsx'), index=False)
        else:
            metrics_df = pd.DataFrame()
            metrics_out = None

        return preds_out, metrics_out, metrics_df

if __name__ == '__main__':
    CSV = r'd:\Code\Python\New folder (2)\researchProject\Data\backtestingData.csv'  # adjust path if needed
    # Provide the date range used when originally collecting data so missing Date can be reconstructed correctly:
    startDate = "2025-08-01"
    endDate = "2025-08-30"

    model = MachineLearningModel(CSV, start_date=startDate, end_date=endDate)

    # split and save per-ticker CSVs
    saved = model.split_save_by_ticker()
    print("Saved per-ticker CSVs:", saved[:5], "..." if len(saved) > 5 else "")

    stats_df = model.get_statistics_by_ticker()
    stats_out = model.out_dir / 'stats_by_ticker.csv'
    stats_df.to_csv(stats_out, index=False)
    print('Saved statistics:', stats_out)

    # existing polynomial fits and plots
    fits_df = model.polynomial_regression_and_plot(degree=3, save_plots=True)
    fits_out = model.out_dir / 'poly_fits_by_ticker.csv'
    fits_df.to_csv(fits_out, index=False)
    print('Saved polynomial fit info and plots to:', model.plots_dir)

    # Run backtest (80/20) and save predictions + metrics
    # restrict the backtest to 2000-01-01 through 2026-01-01 per user request
    preds_path, metrics_path, metrics_df = model.backtest_models(
        test_ratio=0.2,
        degree=3,
        save_plots=True,
        start_date='2000-01-01',
        end_date='2026-01-01'
    )
    print('Saved backtest predictions to:', preds_path)
    print('Saved metrics summary to:', metrics_path)

