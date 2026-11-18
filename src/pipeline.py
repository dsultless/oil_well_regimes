"""Шаги проекта по порядку. Каждый шаг можно запустить из меню (main.py)
или все сразу командой: python main.py --all"""

import json

import pandas as pd

from src import config, data_loader, database, eda, evaluation, plots, quality, transform


def load_clean():
    """Читает train и test из Parquet, помечает выборку, убирает значения с флагами
    и отмечает часы остановки скважины."""
    train = data_loader.load_dataset("train").assign(split="train")
    test = data_loader.load_dataset("test").assign(split="test")
    df = pd.concat([train, test], ignore_index=True)
    df["split"] = df["split"].astype("category")
    for col in ["well_id", "well_type", "parameter", "unit", "flag"]:
        df[col] = df[col].astype("category")
    return transform.mark_stopped(transform.clean_values(df))


def step_prepare():
    """Шаг 1. Скачать данные, прочитать по частям, сжать, сохранить Parquet."""
    stats = data_loader.prepare_data()
    print(f"Строк: {stats['rows']}")
    print(f"Память до оптимизации: {stats['memory_before_mb']} МБ, после: "
          f"{stats['memory_after_mb']} МБ (экономия {stats['memory_saving_percent']}%)")
    print(f"Чтение целиком: {stats['read_full_seconds']} с, пик памяти {stats['read_full_peak_mb']} МБ")
    print(f"Чтение по частям: {stats['read_chunks_seconds']} с, пик памяти {stats['read_chunks_peak_mb']} МБ")
    print(f"CSV: {stats['csv_size_mb']} МБ, чтение {stats['csv_read_seconds']} с")
    print(f"Parquet: {stats['parquet_size_mb']} МБ, чтение {stats['parquet_read_seconds']} с")
    return stats


def step_quality():
    """Шаг 2. Аудит качества данных и графики качества."""
    df = data_loader.load_dataset("all")
    summary, tables = quality.run_audit(df)
    plots.plot_completeness(tables["completeness_by_month"])
    plots.plot_flags(tables["flags_summary"])
    print(f"Строк с флагами: {summary['flagged_rows']} ({summary['flagged_percent']}%)")
    print(f"Дубликатов: {summary['duplicates']}, разрывов больше суток: {summary['gaps_over_24h']}")
    issues = summary["known_issues"]
    print("Рост медианы dynamometer_area после мая (во сколько раз):",
          issues["area_median_ratio_after_before"])
    print(f"NK-7: медиана SPM до июня {issues['nk7_spm_median_before_june']}, "
          f"после {issues['nk7_spm_median_after_june']}, "
          f"нулей после июня {issues['nk7_spm_zero_percent_after_june']}%")
    return summary


def step_database():
    """Шаг 3. Создать базу SQLite и сравнить один расчёт в SQL и Pandas."""
    train = data_loader.load_dataset("train")
    test = data_loader.load_dataset("test")
    counts = database.build_database(train, test)
    for table, count in counts.items():
        print(f"{table}: {count} строк")
    comparison = database.pandas_vs_sql(pd.concat([train, test], ignore_index=True))
    print(f"SQL: {comparison['sql_seconds']} с, Pandas: {comparison['pandas_seconds']} с, "
          f"результаты совпадают: {comparison['same_groups'] and comparison['same_values']}")
    with open(config.TABLES / "pandas_vs_sql.json", "w", encoding="utf-8") as f:
        json.dump({"tables": counts, **comparison}, f, ensure_ascii=False, indent=2)
    return counts, comparison


def step_queries():
    """Шаг 4. Выполнить все SQL-запросы и показать начало каждого результата."""
    results = database.run_all_queries()
    for title, table in results.items():
        print(f"\n{title} (строк: {len(table)})")
        print(table.head(8).to_string(index=False))
    return results


def step_eda():
    """Шаг 5. Разведочный анализ: статистика, суточные паттерны, корреляции, графики."""
    df = load_clean()
    norm, shift_dates = transform.normalize_area(df)
    wide = transform.add_operating_state(transform.to_wide(norm))
    wide.to_parquet(config.DATA_PROCESSED / "wells_wide.parquet", index=False)

    describe = eda.describe_by_type(norm)
    describe.to_csv(config.TABLES / "eda_describe_by_type.csv")
    profile, strength = eda.hourly_profile(wide)
    strength.to_csv(config.TABLES / "eda_daily_amplitude.csv", index=False)
    matrices = eda.correlations(wide)
    for well_type, matrix in matrices.items():
        matrix.to_csv(config.TABLES / f"eda_correlations_{well_type}.csv")

    plots.plot_area_shift(df, norm, shift_dates)
    plots.plot_boxplots(wide)
    plots.plot_hourly(profile)
    plots.plot_correlations(matrices)
    print("Даты сдвига масштаба dynamometer_area:",
          {w: str(d.date()) for w, d in shift_dates.items()})
    print("Сила суточного паттерна (размах, % от среднего):")
    print(strength.to_string(index=False))
    return describe, strength


def step_compare_methods():
    """Шаг 6. Подбор параметров на train, проверка на test, реальные события."""
    df = load_clean()
    final, events_summary, best_params, info = evaluation.run_evaluation(df)
    extra = {"sensor_zero_removed": df.attrs.get("sensor_zero_removed", 0),
             "series_train": info["train_series"], "series_test": info["test_series"]}
    with open(config.TABLES / "report_extra.json", "w", encoding="utf-8") as f:
        json.dump(extra, f, ensure_ascii=False, indent=2)
    database.save_experiments(final.to_dict("records"))
    plots.plot_methods(final)
    plots.plot_detection(df, "NK-7", "SPM", best_params, "07_detection_NK7_SPM.png",
                         "NK-7, SPM: смена режима в июне 2026")
    plots.plot_detection(df, "NK-75", "tubing_pressure", best_params,
                         "08_detection_NK75_tubing.png",
                         "NK-75, давление в НКТ: смена режима в начале июня 2026")
    print(f"Синтетических рядов: train {info['train_series']}, test {info['test_series']}")
    print(final[["method", "split", "params", "precision", "recall", "f1",
                 "mean_delay_h", "time_ms", "memory_kb"]].to_string(index=False))
    print("\nРеальные события:")
    print(events_summary.to_string(index=False))
    return final, events_summary, best_params


def step_detect_all():
    """Шаг 7. Найти режимы во всех рядах, сохранить в базу, сравнить типы скважин."""
    df = load_clean()
    with open(config.TABLES / "best_params.json", encoding="utf-8") as f:
        best_params = json.load(f)["best_params"]
    changes, regimes = evaluation.detect_all(df, best_params)
    database.save_detected_changes(changes)
    regimes.to_csv(config.TABLES / "regimes_all.csv", index=False)

    norm, _ = transform.normalize_area(df)
    wide = transform.add_operating_state(transform.to_wide(norm))
    events = transform.stop_events(wide)
    events.to_csv(config.TABLES / "stop_events.csv", index=False)
    total_hours = quality.expected_hours(df)
    per_well, tests = eda.compare_well_types(wide, events, regimes, total_hours)
    per_well.to_csv(config.TABLES / "well_summary.csv", index=False)
    tests.to_csv(config.TABLES / "well_type_tests.csv", index=False)
    plots.plot_well_types(per_well)
    plots.interactive_overview(df, regimes)
    print(f"Найдено точек смены режима (все методы): {len(changes)}")
    print(per_well.to_string(index=False))
    print("\nСравнение типов скважин:")
    print(tests.to_string(index=False))
    return per_well, tests


def run_all():
    """Запускает все шаги по порядку."""
    steps = [step_prepare, step_quality, step_database, step_eda,
             step_compare_methods, step_detect_all, step_queries]
    for step in steps:
        print(f"\n===== {step.__doc__.splitlines()[0]} =====")
        step()
    print("\nГотово. Результаты: папки results/tables и results/figures, база database/wells.db")
