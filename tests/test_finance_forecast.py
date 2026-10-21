"""Tests for the finance forecast script (pure functions + artifact persistence).

Tensorflow is heavy; artifact tests import it lazily and are skipped when the
package is unavailable (Windows local venvs without TF still run the rest).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

finance = importlib.import_module("finance_forecast")


def _series(n: int = 60, start: float = 100.0) -> pd.Series:
    index = pd.bdate_range("2025-01-01", periods=n)
    values = start + np.arange(n) * 0.5  # simple monotonic series
    return pd.Series(values, index=index, name="Close")


def test_load_prices_requires_columns(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("Wrong,Yes\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Date"):
        finance.load_prices(csv_path)


def test_load_prices_sorts_and_commons_index(tmp_path: Path):
    csv_path = tmp_path / "ok.csv"
    csv_path.write_text(
        "Date,Close\n2025-01-03,3\n2025-01-01,1\n2025-01-02,2\n",
        encoding="utf-8",
    )
    series = finance.load_prices(csv_path)
    assert list(series.index[-3:]) == list(pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]))


def test_split_series_not_enough_rows():
    tiny = _series(10)
    with pytest.raises(ValueError, match="Not enough rows"):
        finance.split_series(tiny, test_size=30, lookback=20)


def test_split_series_dedicated_holdout():
    series = _series(100)
    train, test = finance.split_series(series, test_size=30, lookback=20)
    assert len(test) == 30
    assert len(train) == 70
    assert series.index[70] == test.index[0]  # contiguous holdout


def test_make_sequences_shapes():
    values = np.arange(10).reshape(-1, 1)
    x_data, y_data = finance.make_sequences(values, lookback=3)
    assert x_data.shape == (7, 3, 1)
    assert y_data.shape == (7,)
    assert y_data[0] == values[3, 0]


def test_persist_and_reload_artifacts(tmp_path: Path):
    """CRITICAL regression: model/scaler must survive the process (audit MED)."""
    pytest.importorskip("tensorflow")
    from tensorflow.keras.models import load_model

    series = _series(120)
    train, test = finance.split_series(series, test_size=20, lookback=10)
    result = finance.evaluate_model(train, test, lookback=10, units=4, epochs=1, batch_size=8)

    artifacts_dir = finance.persist_artifacts(
        result,
        params={"lookback": 10, "units": 4, "epochs": 1, "batch_size": 8, "test_size": 20},
        artifacts_dir=tmp_path / "artifacts",
    )

    assert (artifacts_dir / "model.keras").is_file()
    assert (artifacts_dir / "scaler.pkl").is_file()
    meta_text = (artifacts_dir / "meta.json").read_text(encoding="utf-8")
    assert "trained_at_utc" in meta_text and "rmse" in meta_text

    reloaded = load_model(artifacts_dir / "model.keras")
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    assert reloaded.input_shape == (None, 10, 1)
    sample = np.array([[0.5]])
    assert np.isfinite(scaler.inverse_transform(sample)[0, 0])


def test_future_forecast_shapes():
    pytest.importorskip("tensorflow")
    series = _series(120)
    train, test = finance.split_series(series, test_size=20, lookback=10)
    result = finance.evaluate_model(train, test, lookback=10, units=4, epochs=1, batch_size=8)
    future = finance.forecast_future(
        series, scaler=result["scaler"], model=result["model"], lookback=10, days=5
    )
    assert len(future) == 5
    assert np.all(np.isfinite(future.values))
