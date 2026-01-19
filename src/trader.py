import joblib
import pandas as pd
from fastapi import FastAPI
from fetch_data import fetch_new_data
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager
from config import DATA_FILE_PATH
from features import add_features

ml_package = {}
df = pd.DataFrame()


def fetch_data_hourly():
    global df
    print("Fetching data...")
    fetch_new_data()

    try:
        df = pd.read_parquet(f"../{DATA_FILE_PATH}")
        df = add_features(df)
    except Exception as e:
        print("Error while reading file")
        raise e


def get_ml_model(path_name: str):

    print("Getting ML model from local")

    model = None

    try:
        print("File found.")
        model = joblib.load(path_name)
    except FileNotFoundError:
        print("File not found.")
        raise RuntimeError()
    except Exception as e:
        raise e

    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_package
    # initial loading
    fetch_data_hourly()
    # fetch data hourly
    scheduler = AsyncIOScheduler()
    scheduler.add_job(fetch_data_hourly, "interval", hours=1)
    scheduler.start()
    # get the model
    ml_package = get_ml_model("../models/main-model.pkl")
    yield
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/predict")
async def get_prediction():
    global df, ml_package

    row = df.iloc[[-1]].copy()
    features = ml_package["feature_names"]

    data = row[features]

    model = ml_package["model"]
    threshold = ml_package["threshold"]

    probs = model.predict_proba(data)
    short_prob = float(probs[0][0])
    long_prob = float(probs[0][2])

    status = "Neutral"
    if short_prob > threshold:
        status = "Short"
    if long_prob > threshold:
        status = "Long"

    return {
        "short_prob": short_prob,
        "long_prob": long_prob,
        "threshold": threshold,
        "action": status,
    }
