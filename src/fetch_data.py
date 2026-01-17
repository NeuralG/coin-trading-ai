import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import config


def fetch_new_data():

    # === setup the paths ===

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, config.DATA_FILE_PATH)
    tickers = config.TICKERS

    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # === determine the data range ===

    start_date = None
    prev_df = pd.DataFrame()

    max_history_start = datetime.now() - timedelta(days=729)

    if os.path.exists(data_path):

        try:
            prev_df = pd.read_parquet(data_path)
            prev_df["Date"] = pd.to_datetime(prev_df["Date"]).dt.tz_localize(None)

            last_data_date = prev_df["Date"].max()

            # start from one day after last data
            start_date = last_data_date + timedelta(hours=1)

        except Exception as e:
            print(f"Error while reading file: {e}")

    if start_date is None:
        print("No files were found.")
        start_date = max_history_start

    start_date = max(start_date, max_history_start)

    end_date = datetime.now()

    if start_date >= end_date - timedelta(minutes=59):
        print("Data is already updated (Less than 1 hour passed)")
        return

    print(f"Downloading from {start_date} to {end_date}...")

    # === download the data ===

    try:
        new_data = yf.download(
            tickers,
            interval="1h",
            start=start_date,
            end=end_date,
            progress=True,
            group_by="ticker",
            auto_adjust=True,
        )

    except Exception as e:
        print(f"download failed: {e}")
        return

    # === reshaping ===

    stacked = new_data.stack(level=0)
    stacked.reset_index(inplace=True)

    cols = list(stacked.columns)

    cols[0] = "Date"

    if "Symbol" not in stacked.columns:
        cols[1] = "Symbol"

    stacked.columns = cols

    stacked["Date"] = pd.to_datetime(stacked["Date"]).dt.tz_localize(None)

    # clean metadata
    stacked.columns.name = None

    stacked.dropna(subset=["Close"], inplace=True)

    # === merge and save ===

    if not prev_df.empty:
        stacked = stacked[prev_df.columns]
        final_df = pd.concat([prev_df, stacked])
    else:
        final_df = stacked

    final_df.sort_values(["Symbol", "Date"], inplace=True)

    final_df.to_parquet(data_path, index=False)

    print("Data is successfully saved")


if __name__ == "__main__":
    fetch_new_data()
