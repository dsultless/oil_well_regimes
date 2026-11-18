"""Разведочный анализ (EDA): описательная статистика по типам скважин,
суточные паттерны, корреляции параметров и сравнение фонтанных и насосных скважин."""

import numpy as np
import pandas as pd
from scipy import stats

from src import config


def describe_by_type(df):
    """Описательная статистика каждого параметра по типу скважины (без флагов,
    dynamometer_area уже нормирована по периодам)."""
    clean = df.dropna(subset=["value_clean"])
    table = (clean.groupby(["well_type", "parameter"], observed=True)["value_clean"]
                  .describe(percentiles=[0.25, 0.5, 0.75]).round(3))
    # Выбросы по правилу IQR: дальше 1.5 межквартильного размаха от квартилей
    def iqr_outliers(values):
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        return float(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).mean() * 100)
    table["iqr_outliers_percent"] = (clean.groupby(["well_type", "parameter"], observed=True)
                                          ["value_clean"].apply(iqr_outliers).round(2))
    return table


def hourly_profile(wide):
    """Средние значения по часам суток (0-23) только в часы работы скважины.
    Каждую скважину сначала делим на ее среднее, чтобы скважины были сравнимы."""
    running = wide[wide["state"] == "running"].copy()
    running["hour"] = running["timestamp"].dt.hour
    rows = []
    for well_type, params in config.DETECTION_PARAMS.items():
        part = running[running["well_type"] == well_type]
        for parameter in params:
            if parameter not in part.columns:
                continue
            values = part[["well_id", "hour", parameter]].dropna()
            means = values.groupby("well_id", observed=True)[parameter].transform("mean")
            values = values[means != 0]
            values["relative"] = values[parameter] / means[means != 0]
            profile = values.groupby("hour")["relative"].mean()
            for hour, value in profile.items():
                rows.append({"well_type": well_type, "parameter": parameter,
                             "hour": int(hour), "relative_value": float(value)})
    table = pd.DataFrame(rows)
    # Сила суточного паттерна: размах профиля (max - min) в процентах от среднего
    strength = (table.groupby(["well_type", "parameter"])["relative_value"]
                     .agg(lambda s: (s.max() - s.min()) * 100).round(2)
                     .rename("daily_amplitude_percent").reset_index())
    return table, strength


def correlations(wide):
    """Корреляционные матрицы параметров (Пирсон) отдельно для каждого типа,
    только в часы работы."""
    running = wide[wide["state"] == "running"]
    result = {}
    for well_type, params in config.PARAMS_BY_TYPE.items():
        part = running[running["well_type"] == well_type]
        columns = [p for p in params if p in part.columns]
        result[well_type] = part[columns].corr().round(3)
    return result


def compare_well_types(wide, events, regimes, total_hours):
    """Сводка по каждой скважине и сравнение двух типов:
    полнота данных, доля остановок, число и длительность остановок,
    частота смен режима, изменчивость основного параметра."""
    rows = []
    for (well, well_type), part in wide.groupby(["well_id", "well_type"], observed=True):
        main = "SPM" if well_type == config.PUMP else "tubing_pressure"
        running = part[part["state"] == "running"][main].dropna()
        known = part[part["state"] != "unknown"]
        well_events = events[events["well_id"] == well]
        well_regimes = regimes[(regimes["well_id"] == well) &
                               (regimes["method"] == "binary_segmentation")]
        series_count = well_regimes["parameter"].nunique()
        changes = len(well_regimes) - series_count
        days = total_hours / 24
        rows.append({
            "well_id": well, "well_type": well_type,
            "hours_with_data_percent": round(100 * len(part) / total_hours, 1),
            "stopped_percent": round(100 * (known["state"] == "stopped").mean(), 2),
            "stop_events": int(len(well_events)),
            "mean_stop_hours": round(float(well_events["hours"].mean()), 1) if len(well_events) else 0.0,
            "longest_stop_hours": float(well_events["hours"].max()) if len(well_events) else 0.0,
            "changes_per_30_days": round(changes / max(series_count, 1) / days * 30, 2),
            "main_parameter": main,
            "main_cv_percent": round(100 * running.std() / running.mean(), 1) if len(running) else None,
        })
    per_well = pd.DataFrame(rows)

    # Сравнение типов: средние по скважинам + тест Манна-Уитни (выборки маленькие: 3 и 7)
    tests = []
    for column in ["stopped_percent", "stop_events", "mean_stop_hours",
                   "changes_per_30_days", "main_cv_percent"]:
        fountain = per_well[per_well["well_type"] == config.FOUNTAIN][column].dropna()
        pump = per_well[per_well["well_type"] == config.PUMP][column].dropna()
        p_value = stats.mannwhitneyu(fountain, pump, alternative="two-sided").pvalue \
            if len(fountain) and len(pump) else np.nan
        tests.append({"metric": column, "fountain_mean": round(fountain.mean(), 2),
                      "pump_mean": round(pump.mean(), 2), "mann_whitney_p": round(float(p_value), 3)})
    return per_well, pd.DataFrame(tests)
