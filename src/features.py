import numpy as np
import pandas as pd
import pandas_ta as ta
import warnings


def add_features(df):
    warnings.filterwarnings("ignore")

    df = df.copy()
    df.sort_values(["Symbol", "Date"], inplace=True)

    if "Adj Close" in df.columns:
        df.drop(columns=["Adj Close"], inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])

    def compute_indicators(group):
        group.ta.ema(length=9, append=True)
        group.ta.ema(length=21, append=True)
        group.ta.ema(length=50, append=True)
        group.ta.ema(length=100, append=True)
        group.ta.ema(length=200, append=True)
        group.ta.rsi(length=14, append=True)
        group.ta.stochrsi(length=14, append=True)
        group.ta.cci(length=20, append=True)
        group.ta.adx(length=14, append=True)
        group.ta.bbands(length=20, std=2, append=True)
        group.ta.atr(length=14, append=True)
        group.ta.obv(append=True)
        group.ta.mfi(length=14, append=True)
        group.ta.cmf(length=20, append=True)
        group.ta.log_return(append=True)
        group.ta.macd(fast=12, slow=26, signal=9, append=True)
        return group

    def rename_columns(group):
        rename_map = {
            "EMA_9": "EMA_SHORT",
            "EMA_21": "EMA_MID",
            "EMA_50": "EMA_LONG",
            "EMA_200": "EMA_TREND",
            "RSI_14": "RSI",
            "STOCHRSIk_14_14_3_3": "STOCH_K",
            "STOCHRSId_14_14_3_3": "STOCH_D",
            "ADX_14": "ADX",
            "DMP_14": "ADX_POS",
            "DMN_14": "ADX_NEG",
            "CCI_20_0.015": "CCI",
            "ATRr_14": "ATR",
            "MFI_14": "MFI",
            "CMF_20": "CMF",
            "OBV": "OBV",
            "LOGRET_1": "LOG_RET",
            "MACD_12_26_9": "MACD_LINE",
            "MACDh_12_26_9": "MACD_HIST",
            "MACDs_12_26_9": "MACD_SIGNAL",
            "BBU_20_2.0": "BB_UPPER",
            "BBL_20_2.0": "BB_LOWER",
            "BBM_20_2.0": "BB_MID",
        }
        cols_found = [c for c in rename_map.keys() if c in group.columns]
        actual_map = {k: rename_map[k] for k in cols_found}
        group.rename(columns=actual_map, inplace=True)
        return group

    def custom_features(group):
        close = group["Close"]
        open_ = group["Open"]
        high = group["High"]
        low = group["Low"]

        group["candle_body"] = abs(close - open_)
        group["candle_range"] = high - low
        group["upper_wick"] = high - group[["Open", "Close"]].max(axis=1)
        group["lower_wick"] = group[["Open", "Close"]].min(axis=1) - low
        group["upper_wick_ratio"] = group["upper_wick"] / (group["candle_body"] + 1e-9)
        group["lower_wick_ratio"] = group["lower_wick"] / (group["candle_body"] + 1e-9)

        vol_ma = group["Volume"].rolling(window=24).mean()
        group["RVOL"] = group["Volume"] / (vol_ma + 1e-9)

        group["dist_ema_short"] = (close - group["EMA_SHORT"]) / group["EMA_SHORT"]
        group["dist_ema_long"] = (close - group["EMA_LONG"]) / group["EMA_LONG"]

        if "BB_UPPER" in group.columns and "BB_LOWER" in group.columns:
            group["bb_position"] = (close - group["BB_LOWER"]) / (
                group["BB_UPPER"] - group["BB_LOWER"]
            )
            group["bb_width"] = (group["BB_UPPER"] - group["BB_LOWER"]) / group[
                "BB_MID"
            ]

        if "MFI" in group.columns:
            group["rsi_mfi_diff"] = group["RSI"] - group["MFI"]

        if "ADX" in group.columns:
            group["is_trending"] = (group["ADX"] > 25).astype(int)

        return group

    def add_time_features(group):

        date_col = pd.to_datetime(group["Date"])

        group["hour"] = date_col.dt.hour
        group["day_of_week"] = date_col.dt.dayofweek
        group["day_of_month"] = date_col.dt.day
        group["month"] = date_col.dt.month
        group["quarter"] = date_col.dt.quarter
        group["week_of_year"] = date_col.dt.isocalendar().week

        group["hour_sin"] = np.sin(2 * np.pi * group["hour"] / 24)
        group["hour_cos"] = np.cos(2 * np.pi * group["hour"] / 24)

        group["day_sin"] = np.sin(2 * np.pi * group["day_of_week"] / 7)
        group["day_cos"] = np.cos(2 * np.pi * group["day_of_week"] / 7)

        group["month_sin"] = np.sin(2 * np.pi * group["month"] / 12)
        group["month_cos"] = np.cos(2 * np.pi * group["month"] / 12)

        group["is_weekend"] = (group["day_of_week"] >= 5).astype(int)

        group["is_asian_hours"] = ((group["hour"] >= 0) & (group["hour"] < 8)).astype(
            int
        )
        group["is_european_hours"] = (
            (group["hour"] >= 8) & (group["hour"] < 16)
        ).astype(int)
        group["is_us_hours"] = ((group["hour"] >= 16) & (group["hour"] < 24)).astype(
            int
        )

        group["is_market_open"] = ((group["hour"] >= 9) & (group["hour"] <= 16)).astype(
            int
        )
        group["is_morning"] = ((group["hour"] >= 6) & (group["hour"] < 12)).astype(int)
        group["is_afternoon"] = ((group["hour"] >= 12) & (group["hour"] < 18)).astype(
            int
        )
        group["is_evening"] = ((group["hour"] >= 18) & (group["hour"] < 24)).astype(int)
        group["is_night"] = ((group["hour"] >= 0) & (group["hour"] < 6)).astype(int)

        group["is_month_start"] = (group["day_of_month"] <= 5).astype(int)
        group["is_month_end"] = (group["day_of_month"] >= 25).astype(int)

        group["is_quarter_start"] = (
            (group["month"] % 3 == 1) & (group["day_of_month"] <= 5)
        ).astype(int)
        group["is_quarter_end"] = (
            (group["month"] % 3 == 0) & (group["day_of_month"] >= 25)
        ).astype(int)

        return group

    def add_lags(group):
        cols_to_lag = ["LOG_RET", "RSI", "MACD_HIST", "RVOL", "MFI"]
        for col in cols_to_lag:
            if col in group.columns:
                group[f"{col}_lag1"] = group[col].shift(1)
                group[f"{col}_lag2"] = group[col].shift(2)
        return group

    df = df.groupby("Symbol", group_keys=False).apply(compute_indicators)
    df = df.groupby("Symbol", group_keys=False).apply(rename_columns)
    df = df.groupby("Symbol", group_keys=False).apply(custom_features)
    df = df.groupby("Symbol", group_keys=False).apply(add_time_features)
    df = df.groupby("Symbol", group_keys=False).apply(add_lags)

    df.dropna(inplace=True)

    return df
