# Finance Forecast (LSTM)

A simple stock/finance time-series forecasting project using an LSTM model in Python.

## What this project does
- Loads historical price data from CSV (`Date`, `Close` columns)
- Trains an LSTM model on the close-price series
- Evaluates performance on a holdout test set (MAE, RMSE)
- Forecasts the next N business days
- Saves results to CSV files in `outputs/`

## Project files
- `finance_forecast.py` - main training + forecasting script
- `requirements.txt` - Python dependencies
- `finance_prices_sample.csv` - sample input data
- `outputs/` - generated metrics and predictions

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run
```bash
python finance_forecast.py --csv finance_prices_sample.csv
```

## Useful options
```bash
python finance_forecast.py --csv finance_prices_sample.csv --lookback 30 --units 64 --epochs 50 --batch-size 16 --future-days 7
```

## Output files
After each run, these files are created in `outputs/`:
- `metrics.csv`
- `holdout_actual_vs_forecast.csv`
- `future_forecast.csv`

## Input format
Your CSV must include:
- `Date` (date column)
- `Close` (numeric price column)

## Notes
- Forecast dates are business days.
- Results depend on data quality and parameter choices.
