class LSTMModel:
    def __init__(self, df: pd.DataFrame):
        self.raw = df.copy()
        self.data = df.copy()

    def filter_date_range(self, start_date=None, end_date=None):
        if start_date:
            self.data = self.data[self.data['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            self.data = self.data[self.data['Date'] <= pd.to_datetime(end_date)]
        return self.data

    # ==============================
    # Technical Indicators
    # ==============================
    def add_indicators(self, df):
        df = df.copy()

        # EMA
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26

        # Target
        df['Return'] = df['Close'].pct_change()

        df = df.dropna().reset_index(drop=True)
        return df

    # ==============================
    # Sequence Builder
    # ==============================
    @staticmethod
    def createSequences(features, target, time_step):
        X, y = [], []
        for i in range(len(features) - time_step):
            X.append(features[i:i+time_step])
            y.append(target[i+time_step])
        return np.array(X), np.array(y)

    # ==============================
    # Training + Backtest
    # ==============================
    def train_and_backtest(self,
                           test_ratio=0.2,
                           time_step=60,
                           epochs=10,
                           batch_size=32,
                           start_date=None,
                           end_date=None):

        if start_date or end_date:
            self.filter_date_range(start_date, end_date)

        all_preds = []
        metrics = []

        for ticker, grp in self.data.groupby('Ticker'):

            grp = grp.sort_values('Date').reset_index(drop=True)
            grp = self.add_indicators(grp)

            if len(grp) < time_step + 20:
                continue

            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'EMA_20', 'EMA_50', 'RSI_14', 'MACD'
            ]

            features = grp[feature_cols].values
            target = grp['Return'].values

            split_index = int(len(features) * (1 - test_ratio))

            X_train_raw = features[:split_index]
            X_test_raw = features[split_index:]

            y_train_raw = target[:split_index]
            y_test_raw = target[split_index:]

            # === Scale TRAIN only ===
            scaler_X = MinMaxScaler()
            X_train_scaled = scaler_X.fit_transform(X_train_raw)
            X_test_scaled = scaler_X.transform(X_test_raw)

            scaler_y = MinMaxScaler()
            y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))
            y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1))

            # === Create sequences
            X_train, y_train = self.createSequences(X_train_scaled, y_train_scaled, time_step)
            X_test, y_test = self.createSequences(X_test_scaled, y_test_scaled, time_step)

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            # === Model
            tf.keras.backend.clear_session()

            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, return_sequences=True,
                                     input_shape=(time_step, len(feature_cols))),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')

            model.fit(X_train, y_train,
                      epochs=epochs,
                      batch_size=batch_size,
                      verbose=0)

            preds_scaled = model.predict(X_test).flatten()
            preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

            # === Trading Strategy
            signal = np.where(preds > 0, 1, -1)
            strategy_returns = signal * actual

            # === Performance Metrics
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)

            cumulative = (1 + strategy_returns).cumprod()
            max_dd = (cumulative / cumulative.cummax() - 1).min()

            cagr = cumulative.iloc[-1] ** (252/len(strategy_returns)) - 1

            mae = np.mean(np.abs(actual - preds))
            rmse = np.sqrt(np.mean((actual - preds) ** 2))

            dates = grp['Date'].iloc[split_index + time_step:].reset_index(drop=True)

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

        # ==============================
        # Save Outputs
        # ==============================
        if len(all_preds) > 0:
            combined = pd.concat(all_preds, ignore_index=True)
            combined.to_csv('lstm_predictions.csv', index=False)

        if len(metrics) > 0:
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv('lstm_metrics.csv', index=False)
        else:
            metrics_df = pd.DataFrame()

        return metrics_df
