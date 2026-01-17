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
            start_date = last_data_date + timedelta(hours=1)
        except Exception as e:
            print(f"Error while reading file: {e}")

    if start_date is None:
        start_date = max_history_start

    start_date = max(start_date, max_history_start)
    end_date = datetime.now()

    if start_date >= end_date - timedelta(minutes=59):
        print("Data is already updated (Less than 1 hour passed)")
        return

    # === download the data ===
    try:
        new_data = yf.download(
            config.TICKER,
            interval="1h",
            start=start_date,
            end=end_date,
            progress=True,
            auto_adjust=True,
        )
    except Exception as e:
        print(f"download failed: {e}")
        return

    new_data.reset_index(inplace=True)

    new_data.columns = [
        col[0] if isinstance(col, tuple) else col for col in new_data.columns
    ]

    if "Datetime" in new_data.columns:
        new_data.rename(columns={"Datetime": "Date"}, inplace=True)

    new_data["Symbol"] = config.TICKER
    new_data["Date"] = pd.to_datetime(new_data["Date"]).dt.tz_localize(None)
    new_data.columns.name = None
    new_data.dropna(subset=["Close"], inplace=True)

    # === merge and save ===
    if not prev_df.empty:
        final_df = pd.concat([prev_df, new_data], ignore_index=True)
    else:
        final_df = new_data

    final_df.sort_values(["Symbol", "Date"], inplace=True)
    final_df.drop_duplicates(subset=["Symbol", "Date"], keep="last", inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    final_df.to_parquet(data_path, index=False)

    print(f"Data successfully updated. Added {len(new_data)} new rows.")
    print(f"Total rows in dataset: {len(final_df)}")


if __name__ == "__main__":
    fetch_new_data()
