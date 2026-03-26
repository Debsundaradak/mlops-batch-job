"""
MLOps Batch Job — Rolling Mean Signal Pipeline
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Argument parsing

def parse_args():
    parser = argparse.ArgumentParser(description="MLOps signal pipeline")
    parser.add_argument("--input",    required=True, help="Path to input CSV file")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path for log file")
    return parser.parse_args()


# Logging setup

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # File handler — detailed
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — info+
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Config loading + validation

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping at the top level")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")

    if not isinstance(cfg["seed"], int):
        raise ValueError(f"Config 'seed' must be an integer, got: {type(cfg['seed'])}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"Config 'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError(f"Config 'version' must be a non-empty string")

    logger.info(
        "Config loaded and validated — seed=%d, window=%d, version=%s",
        cfg["seed"], cfg["window"], cfg["version"]
    )
    return cfg


# Dataset loading + validation

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        # The provided CSV wraps every entire row in double-quotes
        # (i.e. each line looks like "col1,col2,...").
        # We strip the outer quotes before handing the content to pandas.
        import io as _io
        with open(path, "r", encoding="utf-8") as _f:
            raw_lines = [line.rstrip("\r\n").strip('"') for line in _f]
        content = "\n".join(raw_lines)
        df = pd.read_csv(_io.StringIO(content), skipinitialspace=True)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Input CSV is empty (no data rows)")

    # Normalise column names — strip whitespace, lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Available columns: {list(df.columns)}"
        )

    # Coerce close to numeric; non-parseable values become NaN
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    n_bad = df["close"].isna().sum()
    if n_bad > 0:
        logger.warning("Found %d non-numeric value(s) in 'close'; they will be excluded", n_bad)

    logger.info("Dataset loaded — %d rows, columns: %s", len(df), list(df.columns))
    return df


# Processing

def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute rolling mean on 'close'.
    The first (window-1) rows will be NaN (min_periods=window).
    These rows are excluded from signal computation.
    """
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()
    n_nan = df["rolling_mean"].isna().sum()
    logger.info(
        "Rolling mean computed (window=%d) — %d warm-up rows excluded from signal",
        window, n_nan
    )
    return df


def compute_signal(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    signal = 1 if close > rolling_mean, else 0.
    Rows where rolling_mean is NaN are excluded (signal = NaN).
    """
    df = df.copy()
    valid = df["rolling_mean"].notna()
    df.loc[valid, "signal"] = (df.loc[valid, "close"] > df.loc[valid, "rolling_mean"]).astype(int)
    n_valid = valid.sum()
    logger.info(
        "Signal generated — %d valid rows (signal=1: %d, signal=0: %d)",
        n_valid,
        int(df["signal"].sum()),
        int(n_valid - df["signal"].sum()),
    )
    return df


# Metrics

def compute_metrics(df: pd.DataFrame, version: str, seed: int, latency_ms: float) -> dict:
    valid = df["signal"].notna()
    rows_processed = int(valid.sum())
    signal_rate = float(df.loc[valid, "signal"].mean())

    return {
        "version": version,
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": int(latency_ms),
        "seed": seed,
        "status": "success",
    }


def write_metrics(metrics: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


# Main

def main():
    args = parse_args()

    logger = setup_logging(args.log_file)
    logger.info("=== Job started ===")

    version = "unknown"  # fallback before config is loaded
    start_time = time.time()

    try:
        # 1) Load + validate config
        cfg = load_config(args.config, logger)
        version = cfg["version"]
        seed    = cfg["seed"]
        window  = cfg["window"]

        # 2) Set seed for reproducibility
        np.random.seed(seed)
        logger.debug("NumPy random seed set to %d", seed)

        # 3) Load + validate dataset
        df = load_dataset(args.input, logger)

        # 4) Rolling mean
        df = compute_rolling_mean(df, window, logger)

        # 5) Signal
        df = compute_signal(df, logger)

        # 6) Metrics
        latency_ms = (time.time() - start_time) * 1000
        metrics = compute_metrics(df, version, seed, latency_ms)

        logger.info(
            "Metrics — rows_processed=%d, signal_rate=%.4f, latency_ms=%d",
            metrics["rows_processed"], metrics["value"], metrics["latency_ms"]
        )

        write_metrics(metrics, args.output)
        logger.info("Metrics written to %s", args.output)
        logger.info("=== Job completed successfully ===")

        # Print final metrics to stdout (required by Docker spec)
        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as exc:
        latency_ms = (time.time() - start_time) * 1000
        logger.exception("Job failed: %s", exc)

        error_metrics = {
            "version": version,
            "status": "error",
            "error_message": str(exc),
        }
        try:
            write_metrics(error_metrics, args.output)
            logger.info("Error metrics written to %s", args.output)
        except Exception as write_exc:
            logger.error("Could not write error metrics: %s", write_exc)

        logger.info("=== Job ended with error ===")
        print(json.dumps(error_metrics, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
