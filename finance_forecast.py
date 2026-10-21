import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

ARTIFACTS_DIR = Path("outputs") / "artifacts"


def load_prices(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Close' columns.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna(subset=["Close"])
    series = df.set_index("Date")["Close"].astype(float)
    return series


def split_series(series: pd.Series, test_size: int, lookback: int) -> tuple[pd.Series, pd.Series]:
    if len(series) <= test_size + lookback + 5:
        raise ValueError("Not enough rows. Add more historical data.")
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test


def make_sequences(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    x_data, y_data = [], []
    for i in range(lookback, len(values)):
        x_data.append(values[i - lookback : i, 0])
        y_data.append(values[i, 0])
    x_array = np.array(x_data)
    y_array = np.array(y_data)
    return x_array.reshape((x_array.shape[0], x_array.shape[1], 1)), y_array


def build_model(lookback: int, units: int) -> Sequential:
    model = Sequential(
        [
            LSTM(units, input_shape=(lookback, 1)),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate_model(
    train: pd.Series, test: pd.Series, lookback: int, units: int, epochs: int, batch_size: int
) -> dict:
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))

    # Build test input with trailing training window so each test point has full context.
    full_for_test = np.concatenate([train.values[-lookback:], test.values]).reshape(-1, 1)
    test_scaled = scaler.transform(full_for_test)

    x_train, y_train = make_sequences(train_scaled, lookback)
    x_test, y_test = make_sequences(test_scaled, lookback)

    model = build_model(lookback, units)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    pred_scaled = model.predict(x_test, verbose=0)
    pred = scaler.inverse_transform(pred_scaled).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5

    forecast = pd.Series(pred, index=test.index)
    return {"model": model, "scaler": scaler, "forecast": forecast, "mae": mae, "rmse": rmse}


def forecast_future(
    full_series: pd.Series, scaler: MinMaxScaler, model: Sequential, lookback: int, days: int
) -> pd.Series:
    scaled_series = scaler.transform(full_series.values.reshape(-1, 1))
    window = scaled_series[-lookback:].reshape(1, lookback, 1)
    preds = []

    for _ in range(days):
        next_scaled = model.predict(window, verbose=0)[0, 0]
        preds.append(next_scaled)
        next_step = np.array([[[next_scaled]]])
        window = np.concatenate([window[:, 1:, :], next_step], axis=1)

    pred_values = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    start_date = full_series.index[-1] + pd.offsets.BDay(1)
    future_index = pd.bdate_range(start=start_date, periods=days)
    return pd.Series(pred_values, index=future_index)


def persist_artifacts(
    result: dict,
    params: dict[str, int | float | str],
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> Path:
    """Save trained model, fitted scaler and run metadata for reuse.

    Without this, the only complete artifacts died with the training process.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    result["model"].save(artifacts_dir / "model.keras")
    joblib.dump(result["scaler"], artifacts_dir / "scaler.pkl")
    meta = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "mae": result["mae"],
        "rmse": result["rmse"],
        **params,
    }
    (artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return artifacts_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Finance Forecast (LSTM)")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with Date and Close")
    parser.add_argument("--test-size", type=int, default=30, help="Number of days for holdout test")
    parser.add_argument("--lookback", type=int, default=20, help="Past days per sequence")
    parser.add_argument("--units", type=int, default=50, help="LSTM units")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--future-days", type=int, default=7, help="Days to forecast ahead")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    prices = load_prices(csv_path)
    train, test = split_series(prices, args.test_size, args.lookback)

    result = evaluate_model(
        train, test, lookback=args.lookback, units=args.units, epochs=args.epochs, batch_size=args.batch_size
    )
    print("\nEvaluation (holdout):")
    print(f"MAE : {result['mae']:.4f}")
    print(f"RMSE: {result['rmse']:.4f}")

    print("\nLast 5 actual vs forecast:")
    compare = pd.DataFrame({"actual": test, "forecast": result["forecast"]})
    print(compare.tail(5))
    compare.to_csv(output_dir / "holdout_actual_vs_forecast.csv", index=True)

    future_forecast = forecast_future(
        prices, scaler=result["scaler"], model=result["model"], lookback=args.lookback, days=args.future_days
    )
    print(f"\nNext {args.future_days} day forecast:")
    print(future_forecast)
    future_forecast.to_frame(name="forecast").to_csv(output_dir / "future_forecast.csv", index=True)

    artifacts_dir = persist_artifacts(
        result,
        params={
            "lookback": args.lookback,
            "units": args.units,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "test_size": args.test_size,
            "future_days": args.future_days,
        },
    )

    metrics_df = pd.DataFrame(
        [
            {
                "mae": result["mae"],
                "rmse": result["rmse"],
                "model": "lstm",
                "lookback": args.lookback,
                "units": args.units,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "test_size": args.test_size,
                "future_days": args.future_days,
            }
        ]
    )
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print("\nSaved outputs to ./outputs:")
    print("- metrics.csv")
    print("- holdout_actual_vs_forecast.csv")
    print("- future_forecast.csv")
    print(f"- artifacts (model/scaler/meta) -> {artifacts_dir}")


if __name__ == "__main__":
    main()
