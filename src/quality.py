"""Аудит качества данных: пропуски, флаги, нули, разрывы во времени
и проверка известных проблем датасета."""

import json

import pandas as pd

from src import config


def expected_hours(df):
    """Сколько часов должно быть за весь период (с первого до последнего часа)."""
    return int((df["timestamp"].max() - df["timestamp"].min()) / pd.Timedelta(hours=1)) + 1


def completeness(df):
    """Полнота данных: доля часов, в которых есть значение параметра."""
    total = expected_hours(df)
    table = (df.groupby(["well_id", "well_type", "parameter"], observed=True)
               .size().rename("hours_present").reset_index())
    table["hours_expected"] = total
    table["completeness_percent"] = (100 * table["hours_present"] / total).round(1)
    return table


def completeness_by_month(df):
    """Полнота по месяцам для тепловой карты: скважина x месяц, % часов с данными."""
    hours = df.drop_duplicates(["well_id", "timestamp"]).copy()
    hours["month"] = hours["timestamp"].dt.to_period("M").astype(str)
    present = hours.groupby(["well_id", "month"], observed=True).size().unstack()
    days = pd.Series({m: pd.Period(m).days_in_month for m in present.columns})
    return (100 * present / (days * 24)).round(1)


def flags_summary(df):
    """Сколько строк с каждым флагом у каждой скважины."""
    flagged = df[df["flag"].notna()]
    table = flagged.groupby(["well_id", "flag"], observed=True).size().unstack(fill_value=0)
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False)


def zero_share(df):
    """Доля нулевых значений (нули обычно означают остановку скважины)."""
    grouped = df.groupby(["well_id", "parameter"], observed=True)["value"]
    table = (grouped.apply(lambda s: (s == 0).mean() * 100).round(2)
             .rename("zero_percent").reset_index())
    return table[table["zero_percent"] > 0].sort_values("zero_percent", ascending=False)


def time_gaps(df, min_hours=6):
    """Разрывы во времени длиннее min_hours часов по каждой скважине."""
    rows = []
    hours = df.drop_duplicates(["well_id", "timestamp"]).sort_values("timestamp")
    for well, part in hours.groupby("well_id", observed=True):
        diff = part["timestamp"].diff()
        for end_time, delta in zip(part["timestamp"], diff):
            if pd.notna(delta) and delta > pd.Timedelta(hours=min_hours):
                rows.append({"well_id": well, "gap_start": end_time - delta,
                             "gap_end": end_time, "gap_hours": delta / pd.Timedelta(hours=1)})
    return pd.DataFrame(rows).sort_values("gap_hours", ascending=False)


def known_issues(df):
    """Проверяет две известные проблемы из описания датасета."""
    result = {}
    clean = df[df["flag"].isna()]

    # 1. Сдвиг масштаба dynamometer_area в мае 2026
    area = clean[clean["parameter"] == "dynamometer_area"]
    ratios = {}
    for well, part in area.groupby("well_id", observed=True):
        before = part[part["timestamp"] < "2026-05-01"]["value"].median()
        after = part[part["timestamp"] >= "2026-05-15"]["value"].median()
        if pd.notna(before) and before > 0:
            ratios[str(well)] = round(float(after / before), 1)
    result["area_median_ratio_after_before"] = ratios

    # 2. Смена режима NK-7 с июня 2026
    spm = clean[(clean["well_id"] == "NK-7") & (clean["parameter"] == "SPM")]
    before = spm[spm["timestamp"] < "2026-06-01"]["value"]
    after = spm[spm["timestamp"] >= "2026-06-15"]["value"]
    result["nk7_spm_median_before_june"] = round(float(before.median()), 2)
    result["nk7_spm_median_after_june"] = round(float(after.median()), 2)
    result["nk7_spm_zero_percent_after_june"] = round(float((after == 0).mean() * 100), 1)
    return result


def run_audit(df):
    """Запускает весь аудит, сохраняет таблицы в results/tables и возвращает сводку."""
    config.ensure_dirs()
    tables = {
        "completeness": completeness(df),
        "completeness_by_month": completeness_by_month(df),
        "flags_summary": flags_summary(df),
        "zero_share": zero_share(df),
        "time_gaps": time_gaps(df),
    }
    for name, table in tables.items():
        table.to_csv(config.TABLES / f"quality_{name}.csv", index=name in
                     ["completeness_by_month", "flags_summary"])

    summary = {
        "rows": int(len(df)),
        "wells": int(df["well_id"].nunique()),
        "period": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "expected_hours": expected_hours(df),
        "flagged_rows": int(df["flag"].notna().sum()),
        "flagged_percent": round(float(df["flag"].notna().mean() * 100), 2),
        "duplicates": int(df.duplicated(["well_id", "parameter", "timestamp"]).sum()),
        "gaps_over_24h": int((tables["time_gaps"]["gap_hours"] > 24).sum()),
        "known_issues": known_issues(df),
    }
    with open(config.TABLES / "quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary, tables
