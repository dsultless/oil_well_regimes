"""Загрузка данных: скачивание с HuggingFace, чтение CSV по частям,
оптимизация типов, сохранение в Parquet и сравнение CSV и Parquet."""

import json
import time
import tracemalloc
import urllib.request

import pandas as pd

from src import config


def download_data(force=False):
    """Скачивает CSV-файлы датасета, если их еще нет в data/raw."""
    config.ensure_dirs()
    downloaded = []
    for name in config.RAW_FILES:
        target = config.DATA_RAW / name
        if target.exists() and not force:
            continue
        url = f"{config.HF_BASE}/{name}"
        print(f"Скачиваю {url}")
        try:
            urllib.request.urlretrieve(url, target)
        except Exception:
            # На некоторых Mac у Python нет сертификатов, тогда берем их из certifi
            import ssl
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(url, context=context) as response:
                target.write_bytes(response.read())
        downloaded.append(name)
    return downloaded


def optimize_dtypes(df):
    """Уменьшает память: текстовые колонки в category, числа в меньшие типы."""
    df = df.copy()
    for col in ["well_id", "well_type", "parameter", "unit", "flag"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    df["value"] = pd.to_numeric(df["value"], downcast="float")
    df["n_obs"] = pd.to_numeric(df["n_obs"], downcast="integer")
    return df


def memory_mb(df):
    """Сколько мегабайт занимает таблица в памяти (с учетом текста)."""
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def read_full(path):
    """Обычное чтение всего CSV за один раз (вариант для сравнения)."""
    return pd.read_csv(path, parse_dates=["timestamp"])


def read_in_chunks(path, chunksize=50_000):
    """Читает CSV частями по chunksize строк и сразу сжимает каждую часть."""
    parts = []
    for chunk in pd.read_csv(path, parse_dates=["timestamp"], chunksize=chunksize):
        parts.append(optimize_dtypes(chunk))
    df = pd.concat(parts, ignore_index=True)
    # После concat категории разных частей надо снова привести к category
    return optimize_dtypes(df)


def measure(func, *args):
    """Запускает функцию и возвращает результат, время (с) и пик памяти (МБ)."""
    tracemalloc.start()
    start = time.perf_counter()
    result = func(*args)
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, seconds, peak / 1024 ** 2


def prepare_data():
    """Полная подготовка: скачать, прочитать двумя способами, сравнить,
    сохранить Parquet. Возвращает словарь с замерами."""
    download_data()
    csv_path = config.DATA_RAW / "wells_dataset.csv"

    full_df, full_time, full_peak = measure(read_full, csv_path)
    chunk_df, chunk_time, chunk_peak = measure(read_in_chunks, csv_path)

    stats = {
        "rows": int(len(chunk_df)),
        "columns": list(chunk_df.columns),
        "memory_before_mb": round(memory_mb(full_df), 2),
        "memory_after_mb": round(memory_mb(chunk_df), 2),
        "read_full_seconds": round(full_time, 3),
        "read_full_peak_mb": round(full_peak, 2),
        "read_chunks_seconds": round(chunk_time, 3),
        "read_chunks_peak_mb": round(chunk_peak, 2),
        "dtypes_before": {c: str(t) for c, t in full_df.dtypes.items()},
        "dtypes_after": {c: str(t) for c, t in chunk_df.dtypes.items()},
    }
    stats["memory_saving_percent"] = round(
        100 * (1 - stats["memory_after_mb"] / stats["memory_before_mb"]), 1)

    # Сохраняем полный датасет и разбиение train/test в Parquet
    config.ensure_dirs()
    chunk_df.to_parquet(config.DATA_PROCESSED / "wells.parquet", index=False)
    for split in ["train", "test"]:
        part = read_in_chunks(config.DATA_RAW / f"{split}.csv")
        part.to_parquet(config.DATA_PROCESSED / f"{split}.parquet", index=False)

    stats.update(compare_csv_parquet())
    with open(config.TABLES / "data_preparation.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def compare_csv_parquet(repeats=3):
    """Сравнивает CSV и Parquet по размеру файла и времени чтения."""
    csv_path = config.DATA_RAW / "wells_dataset.csv"
    parquet_path = config.DATA_PROCESSED / "wells.parquet"

    def best_time(reader):
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            reader()
            times.append(time.perf_counter() - start)
        return min(times)

    return {
        "csv_size_mb": round(csv_path.stat().st_size / 1024 ** 2, 2),
        "parquet_size_mb": round(parquet_path.stat().st_size / 1024 ** 2, 2),
        "csv_read_seconds": round(best_time(lambda: pd.read_csv(csv_path, parse_dates=["timestamp"])), 3),
        "parquet_read_seconds": round(best_time(lambda: pd.read_parquet(parquet_path)), 3),
    }


def load_dataset(split="all"):
    """Быстро читает подготовленный Parquet. split: all, train или test."""
    name = "wells.parquet" if split == "all" else f"{split}.parquet"
    path = config.DATA_PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла {path}. Сначала выполните пункт меню 1 (подготовка данных).")
    return pd.read_parquet(path)
