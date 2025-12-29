from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("data/raw")
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True)

FEATURES = ["onpromotion", "is_holiday", "lag_1", "lag_7", "rolling_7_mean", "rolling_14_mean"]
TARGET = "sales"


def build_features(train: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    train = train.sort_values(["store_nbr", "family", "date"]).copy()

    # Holiday flag
    holidays = holidays[(holidays["type"] == "Holiday") & (holidays["transferred"] == False)]
    holidays = holidays[["date"]].drop_duplicates()
    train = train.merge(holidays.assign(is_holiday=1), on="date", how="left")
    train["is_holiday"] = train["is_holiday"].fillna(0)

    # Lag features (store + family bazında)
    g = train.groupby(["store_nbr", "family"])["sales"]
    train["lag_1"] = g.shift(1)
    train["lag_7"] = g.shift(7)

    # Rolling mean (leakage yok: shift(1) üzerinden)
    train["rolling_7_mean"] = g.shift(1).rolling(7).mean()
    train["rolling_14_mean"] = g.shift(1).rolling(14).mean()

    train["year"] = train["date"].dt.year

    return train.dropna().copy()


def main():
    print("✅ train_model.py başladı")

    train_path = DATA_DIR / "train.csv"
    hol_path = DATA_DIR / "holidays_events.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"train.csv bulunamadı: {train_path.resolve()}")
    if not hol_path.exists():
        raise FileNotFoundError(f"holidays_events.csv bulunamadı: {hol_path.resolve()}")

    train = pd.read_csv(train_path, parse_dates=["date"])
    holidays = pd.read_csv(hol_path, parse_dates=["date"])

    fe = build_features(train, holidays)

    train_data = fe[fe["year"] < 2017]
    val_data = fe[fe["year"] == 2017]

    X_train, y_train = train_data[FEATURES], train_data[TARGET]
    X_val, y_val = val_data[FEATURES], val_data[TARGET]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")

    joblib.dump({"model": model, "features": FEATURES}, ART_DIR / "model_bundle.joblib")
    print("✅ Saved -> artifacts/model_bundle.joblib")


if __name__ == "__main__":
    main()
