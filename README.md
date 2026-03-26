# MLOps Signal Pipeline — Task 0

A minimal MLOps-style batch job that loads OHLCV data, computes a rolling mean on the `close` price, generates a binary signal, and writes structured metrics.

---

## Repository structure

```
.
├── run.py           # Main pipeline script
├── config.yaml      # Pipeline configuration
├── data.csv         # Input OHLCV data (10 000 rows)
├── requirements.txt # Python dependencies
├── Dockerfile       # Container definition
├── metrics.json     # Sample output from a successful run
├── run.log          # Sample log from a successful run
└── README.md        # This file
```

---

## Local run

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

The final metrics JSON is printed to stdout and written to `metrics.json`.  
Detailed logs are written to `run.log`.

---

## Docker

### Build

```bash
docker build -t mlops-task .
```

### Run

```bash
docker run --rm mlops-task
```

The container prints the metrics JSON to stdout and exits 0 on success, non-zero on failure.

---

## Config reference (`config.yaml`)

| Key       | Type   | Description                              |
|-----------|--------|------------------------------------------|
| `seed`    | int    | NumPy random seed for reproducibility   |
| `window`  | int    | Rolling mean window size (rows)         |
| `version` | string | Pipeline version tag written to metrics |

---

## Processing logic

1. **Load config** — parse YAML, validate required keys (`seed`, `window`, `version`), set `numpy.random.seed(seed)`.
2. **Load dataset** — read CSV, validate non-empty and presence of `close` column.
3. **Rolling mean** — compute `close.rolling(window=window, min_periods=window).mean()`.  
   The first `window-1` rows produce `NaN` and are **excluded** from signal computation.
4. **Signal** — `signal = 1` if `close > rolling_mean`, else `0`.
5. **Metrics** — compute `rows_processed`, `signal_rate = mean(signal)`, `latency_ms`.

---

## Example `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 57,
  "seed": 42,
  "status": "success"
}
```

> **Note on `rows_processed`:** With `window=5`, the first 4 rows have no rolling mean and are excluded, giving 9 996 signal rows from 10 000 total.

---

## Error output

If the job fails for any reason (missing file, bad config, missing column, …) it exits non-zero and writes:

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong"
}
```

---

## Contact / Submission

Send your GitHub repo link + log files to:
- joydip@anything.ai
- chetan@anything.ai
- hello@anything.ai
- CC: sonika@anything.ai
