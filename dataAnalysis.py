from dataRetrievalV1 import dataRetrieval, stockList
import pandas as pd
import numpy as np
import csv

# === Const. Variables === #
csvDataFilePath = 'D:/Code/Python/New folder (2)/researchProject/Data/backtestingData.csv'
csvAnalysisFilePath = 'D:/Code/Python/New folder (2)/researchProject/Data/dataAnalysis.csv'
xlsxAnalysisFilePath = 'D:/Code/Python/New folder (2)/researchProject/Data/dataAnalysis.xlsx'
numRowsToAnalyze = 5  # Last N rows per ticker

dataFile = pd.read_csv(csvDataFilePath)

# === Read Data === #
class DataAnalysis:

    @staticmethod
    def getTickerRows(dataFile, ticker, columnName='Ticker'):
        return dataFile[dataFile[columnName] == ticker]

    @staticmethod
    def movingAverageAnalysis(row):
        SMA_20, SMA_50, SMA_100, SMA_200 = row['SMA_20'], row['SMA_50'], row['SMA_100'], row['SMA_200']
        EMA_12, EMA_26, EMA_50, EMA_200 = row['EMA_12'], row['EMA_26'], row['EMA_50'], row['EMA_200']

        # SMA Trend
        if SMA_20 > SMA_50 > SMA_100 > SMA_200:
            smaTrend = "Strong Uptrend"
        elif SMA_20 < SMA_50 < SMA_100 < SMA_200:
            smaTrend = "Strong Downtrend"
        elif SMA_20 > SMA_50 and SMA_50 < SMA_100:
            smaTrend = "Potential Uptrend"
        elif SMA_20 < SMA_50 and SMA_50 > SMA_100:
            smaTrend = "Potential Downtrend"
        else:
            smaTrend = "Sideways/Unclear Trend"

        # EMA Trend
        if EMA_12 > EMA_26 and EMA_50  > EMA_200:
            emaTrend = "Strong Uptrend"
        elif EMA_12 < EMA_26 and EMA_50 < EMA_200:
            emaTrend = "Strong Downtrend"
        elif EMA_12 > EMA_26 and EMA_50 < EMA_200:
            emaTrend = "Potential Uptrend"
        elif EMA_12 < EMA_26 and EMA_50 > EMA_200:
            emaTrend = "Potential Downtrend"
        else:
            emaTrend = "Sideways/Unclear Trend"

        return smaTrend, emaTrend

    @staticmethod
    def macdAnalysis(row):
        macd, signalLine = row['MACD'], row['Signal_Line']
        if macd > signalLine:
            return "Bullish Momentum"
        elif macd < signalLine:
            return "Bearish Momentum"
        else:
            return "Neutral Momentum"

    @staticmethod
    def adxAnalysis(row):
        adx = row['ADX']
        if adx > 75:
            return "Extremely Strong Trend"
        elif adx > 50:
            return "Very Strong Trend"
        elif adx > 25:
            return "Strong Trend"
        elif 20 < adx <= 25:
            return "Weak Trend"
        else:
            return "Weak/No Trend"

    @staticmethod
    def rsiAnalysis(row):
        rsi = row['RSI']
        if rsi > 70:
            return "Overbought"
        elif rsi < 30:
            return "Oversold"
        else:
            return "Neutral"

    @staticmethod
    def obvAnalysis(row):
        obv = row['OBV']
        if obv > 0:
            return "Buying Pressure"
        elif obv < 0:
            return "Selling Pressure"
        else:
            return "Neutral"

# === Analyze last N rows per ticker and save === #
allResults = []

for ticker in stockList["nasdaqStocks"] + stockList["nyseStockExchange"]:
    tickerData = DataAnalysis.getTickerRows(dataFile, ticker)
    if not tickerData.empty:
        lastRows = tickerData.tail(numRowsToAnalyze)  # Last N rows
        for _, row in lastRows.iterrows():
            smaTrend, emaTrend = DataAnalysis.movingAverageAnalysis(row)
            macdTrend = DataAnalysis.macdAnalysis(row)
            adxTrend = DataAnalysis.adxAnalysis(row)
            rsiTrend = DataAnalysis.rsiAnalysis(row)
            obvTrend = DataAnalysis.obvAnalysis(row)

            analysisResult = {
                'Ticker': ticker,
                'Date': row['Date'],
                'SMA_Trend': smaTrend,
                'EMA_Trend': emaTrend,
                'MACD_Trend': macdTrend,
                'ADX_Trend': adxTrend,
                'RSI_Trend': rsiTrend,
                'OBV_Trend': obvTrend
            }
            allResults.append(analysisResult)

# Export all results to CSV
if allResults:
    analysisDF = pd.DataFrame(allResults)
    analysisDF.to_csv(csvAnalysisFilePath, index=False)

    # Export Excel with one sheet per ticker
    with pd.ExcelWriter(xlsxAnalysisFilePath) as writer:
        for ticker in stockList["nasdaqStocks"] + stockList["nyseStockExchange"]:
            tickerDF = analysisDF[analysisDF['Ticker'] == ticker]
            if not tickerDF.empty:
                tickerDF.to_excel(writer, sheet_name=ticker[:31], index=False)

print(f"Analysis complete → CSV & Excel saved. Last {numRowsToAnalyze} rows per ticker analyzed.")
