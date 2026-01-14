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

    if os.path.exists(data_path):

        try:
            prev_df = pd.read_parquet(data_path)
            last_data_date = prev_df["Date"].max()

            # start from one day after last data
            start_date = last_data_date + timedelta(days=1)

        except Exception as e:
            print(f"Error while reading file: {e}")

    if start_date is None:
        print("No files were found.")
        start_date = datetime.now() - timedelta(days=365 * config.HISTORY_YEARS)

    end_date = datetime.now()

    if start_date.date() >= end_date.date():
        print("Data is already updated")
        return

    # === download the data ===

    try:
        new_data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by="ticker",
            auto_adjust=True,
        )

    except Exception as e:
        print(f"download failed: {e}")
        return

    # === reshaping ===

    stacked = new_data.stack(level=0, future_stack=True)
    stacked.reset_index(inplace=True)

    rename_dict = {"level_1": "Symbol", "Ticker": "Symbol"}
    stacked = stacked.rename(columns=rename_dict)

    stacked["Date"] = pd.to_datetime(stacked["Date"])

    # clean metadata
    stacked.columns.name = None

    # === merge and save ===

    if not prev_df.empty:
        stacked = stacked[prev_df.columns]
        final_df = pd.concat([prev_df, stacked])
    else:
        final_df = stacked

    final_df.to_parquet(data_path, index=False)

    print("Data is successfully saved")


if __name__ == "__main__":
    fetch_new_data()
