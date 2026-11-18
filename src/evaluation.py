"""Сравнение методов поиска смены режима.

Проблема: в датасете нет готовой разметки «здесь сменился режим».
Поэтому оцениваем двумя способами:

1. Полусинтетическая проверка. Берем шум настоящих рядов (остатки после
   скользящей медианы) и вставляем в него смены уровня в известных местах,
   плюс короткие выбросы на 1-2 точки (сбои датчика, которые НЕ являются
   сменой режима). Остановки скважины из рядов заранее убраны (transform.mark_stopped). Места вставки известны, поэтому можно
   посчитать precision, recall и F1. Параметры методов подбираем на рядах
   из train, проверяем на рядах из test (хронологическое разбиение датасета).

2. Реальные события из описания датасета (сдвиг масштаба dynamometer_area
   в мае 2026, смена режима NK-7 в июне 2026) и одна явная смена режима NK-75,
   найденная при осмотре графиков. Проверяем, находит ли метод событие в окне.
"""

import json
import time
import tracemalloc

import numpy as np
import pandas as pd

from src import config, detectors, transform

TOLERANCE = 8        # допуск: 8 точек по 6 часов = 2 суток
MIN_LENGTH = 60      # ряды короче 60 точек (15 суток) не используем
HOURS_PER_POINT = 6

# Реальные события: скважина, параметр, окно поиска, источник
REAL_EVENTS = [
    {"well_id": w, "parameter": "dynamometer_area", "start": "2026-04-28", "end": "2026-05-20",
     "source": "Описание датасета: сдвиг масштаба dynamometer_area в мае 2026"}
    for w in ["NK-7", "NK-68", "NK-77", "NK-82", "NK-86", "NK-91", "NK-93"]
] + [
    {"well_id": "NK-7", "parameter": "SPM", "start": "2026-05-25", "end": "2026-06-20",
     "source": "Описание датасета: NK-7 с июня 2026 работает в другом режиме (SPM 4-5 -> 1)"},
    {"well_id": "NK-75", "parameter": "tubing_pressure", "start": "2026-05-28", "end": "2026-06-10",
     "source": "Осмотр графиков: давление в НКТ NK-75 выросло примерно с 7 до 22 атм"},
]


def collect_series(df, split=None):
    """Собирает все ряды (скважина, параметр) для поиска смены режима.
    split: None = весь период, train или test = только эта часть."""
    if split is not None:
        df = df[df["split"] == split]
    result = []
    wells = df[["well_id", "well_type"]].drop_duplicates().astype(str)
    for well, well_type in wells.itertuples(index=False):
        for parameter in config.DETECTION_PARAMS[well_type]:
            series = transform.make_series(df, well, parameter)
            if len(series) >= MIN_LENGTH:
                result.append({"well_id": well, "well_type": well_type,
                               "parameter": parameter, "series": series})
    return result


def make_synthetic(real_values, rng, copies=3):
    """Делает несколько синтетических рядов из одного настоящего.
    Возвращает список пар (значения, истинные точки смены)."""
    x = pd.Series(real_values)
    residuals = (x - x.rolling(9, center=True, min_periods=1).median()).to_numpy()
    spread = max(detectors.robust_std(residuals), 0.01 * (abs(np.median(real_values)) + 1e-6))
    length = len(real_values)
    base = float(np.median(real_values))
    output = []
    for _ in range(copies):
        # Шум: случайные куски настоящих остатков по 8 точек (сохраняем их структуру)
        blocks = []
        while sum(len(b) for b in blocks) < length:
            start = rng.integers(0, max(1, length - 8))
            blocks.append(residuals[start:start + 8])
        noise = np.concatenate(blocks)[:length]

        # Истинные смены режима: 1-3 штуки, не ближе 30 точек друг к другу и к краям
        max_changes = 3 if length >= 200 else 2
        n_changes = int(rng.integers(1, max_changes + 1))
        true_points = []
        for _attempt in range(200):
            candidate = sorted(rng.choice(np.arange(20, length - 20), n_changes, replace=False))
            gaps = np.diff([0] + list(candidate) + [length])
            if np.all(gaps >= 30):
                true_points = [int(c) for c in candidate]
                break

        level = np.full(length, base)
        current = base
        bounds = true_points + [length]
        for i, point in enumerate(true_points):
            current = current + rng.choice([-1, 1]) * rng.uniform(3, 6) * spread
            level[point:bounds[i + 1]] = current
        values = level + noise

        # Короткие выбросы на 1-2 точки (сбои датчика), это НЕ смена режима
        for _ in range(int(rng.integers(0, 3))):
            start = int(rng.integers(5, length - 10))
            values[start:start + int(rng.integers(1, 3))] += rng.choice([-1, 1]) * rng.uniform(4, 8) * spread
        output.append((values, true_points))
    return output


def build_benchmark(series_list, copies=3):
    """Синтетический набор для всех рядов одной выборки."""
    rng = np.random.default_rng(config.SEED)
    data = []
    for item in series_list:
        for values, true_points in make_synthetic(item["series"].to_numpy(), rng, copies):
            data.append({"well_type": item["well_type"], "parameter": item["parameter"],
                         "values": values, "true": true_points})
    return data


def match_points(true_points, found_points, tolerance=TOLERANCE):
    """Сопоставляет найденные точки с истинными (в пределах допуска).
    Возвращает число верных находок, ложных, пропущенных и список задержек."""
    found_left = list(found_points)
    tp, delays = 0, []
    for point in true_points:
        if not found_left:
            break
        nearest = min(found_left, key=lambda f: abs(f - point))
        if abs(nearest - point) <= tolerance:
            tp += 1
            delays.append(abs(nearest - point))
            found_left.remove(nearest)
    fp = len(found_left)
    fn = len(true_points) - tp
    return tp, fp, fn, delays


def score(benchmark, method, params):
    """Precision, recall, F1, средняя задержка и время для метода на наборе рядов."""
    tp = fp = fn = 0
    delays = []
    start = time.perf_counter()
    for item in benchmark:
        found = detectors.detect(item["values"], method, params)
        a, b, c, d = match_points(item["true"], found)
        tp, fp, fn = tp + a, fp + b, fn + c
        delays.extend(d)
    elapsed = time.perf_counter() - start
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "mean_delay_h": float(np.mean(delays)) * HOURS_PER_POINT if delays else None,
            "time_ms": 1000 * elapsed / len(benchmark)}


def peak_memory_kb(values, method, params):
    """Пиковая память (КБ), которую метод занимает на одном ряде."""
    tracemalloc.start()
    detectors.detect(values, method, params)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024


def tune_and_evaluate(df):
    """Подбирает параметры каждого метода на train и проверяет на test."""
    train_bench = build_benchmark(collect_series(df, "train"))
    test_bench = build_benchmark(collect_series(df, "test"))

    tuning_rows, final_rows, best_params = [], [], {}
    longest = max(train_bench, key=lambda b: len(b["values"]))["values"]
    for method, grid in detectors.PARAM_GRID.items():
        best = None
        for params in grid:
            result = score(train_bench, method, params)
            tuning_rows.append({"method": method, "params": json.dumps(params), **result})
            if best is None or result["f1"] > best[1]["f1"]:
                best = (params, result)
        best_params[method] = best[0]
        memory = peak_memory_kb(longest, method, best[0])
        for split, bench in [("train", train_bench), ("test", test_bench)]:
            result = best[1] if split == "train" else score(bench, method, best[0])
            final_rows.append({"method": method, "split": split, "params": json.dumps(best[0]),
                               "precision": round(result["precision"], 3),
                               "recall": round(result["recall"], 3),
                               "f1": round(result["f1"], 3),
                               "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
                               "mean_delay_h": round(result["mean_delay_h"], 1)
                               if result["mean_delay_h"] is not None else None,
                               "time_ms": round(result["time_ms"], 3),
                               "memory_kb": round(memory, 1)})

    info = {"train_series": len(train_bench), "test_series": len(test_bench),
            "train_true_changes": sum(len(b["true"]) for b in train_bench),
            "test_true_changes": sum(len(b["true"]) for b in test_bench),
            "tolerance_hours": TOLERANCE * HOURS_PER_POINT}
    return pd.DataFrame(tuning_rows), pd.DataFrame(final_rows), best_params, info


def evaluate_real_events(df, best_params):
    """Проверяет, находят ли методы реальные события в нужном окне.
    Для dynamometer_area берем значения БЕЗ нормировки (log10), чтобы сдвиг был виден."""
    rows = []
    for event in REAL_EVENTS:
        series = transform.make_series(df, event["well_id"], event["parameter"])
        for method, params in best_params.items():
            found = detectors.detect(series.to_numpy(), method, params)
            times = [series.index[i] for i in found]
            inside = [t for t in times if pd.Timestamp(event["start"]) <= t <= pd.Timestamp(event["end"])]
            rows.append({"well_id": event["well_id"], "parameter": event["parameter"],
                         "window": f"{event['start']} .. {event['end']}", "method": method,
                         "detected": bool(inside),
                         "detected_at": str(inside[0]) if inside else "",
                         "changes_in_series": len(found), "source": event["source"]})
    table = pd.DataFrame(rows)
    summary = (table.groupby("method")
                    .agg(events_found=("detected", "sum"), events_total=("detected", "size"),
                         avg_changes_per_series=("changes_in_series", "mean"))
                    .reset_index())
    summary["event_recall"] = (summary["events_found"] / summary["events_total"]).round(3)
    return table, summary


def detect_all(df, best_params):
    """Ищет смены режима во всех рядах за весь период всеми методами.
    Возвращает список найденных точек и таблицу режимов."""
    changes, regimes = [], []
    for item in collect_series(df):
        series = item["series"]
        for method, params in best_params.items():
            found = detectors.detect(series.to_numpy(), method, params)
            parts = detectors.segments(series, found)
            for i, index in enumerate(found):
                changes.append({"well_id": item["well_id"], "parameter": item["parameter"],
                                "method": method, "change_ts": series.index[index],
                                "mean_before": parts[i]["mean"], "mean_after": parts[i + 1]["mean"]})
            for part in parts:
                regimes.append({"well_id": item["well_id"], "well_type": item["well_type"],
                                "parameter": item["parameter"], "method": method, **part})
    return changes, pd.DataFrame(regimes)


def run_evaluation(df):
    """Весь этап сравнения методов. Сохраняет таблицы в results/tables."""
    config.ensure_dirs()
    tuning, final, best_params, info = tune_and_evaluate(df)
    events, events_summary = evaluate_real_events(df, best_params)
    tuning.to_csv(config.TABLES / "methods_tuning_train.csv", index=False)
    final.to_csv(config.TABLES / "methods_comparison.csv", index=False)
    events.to_csv(config.TABLES / "real_events_detail.csv", index=False)
    events_summary.to_csv(config.TABLES / "real_events_summary.csv", index=False)
    with open(config.TABLES / "best_params.json", "w", encoding="utf-8") as f:
        json.dump({"best_params": best_params, "benchmark": info}, f, ensure_ascii=False, indent=2)
    return final, events_summary, best_params, info
