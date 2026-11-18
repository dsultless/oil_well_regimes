"""Преобразования данных: очистка флагов, нормировка dynamometer_area,
перевод из long в wide формат, рабочие состояния скважин, ряды для анализа."""

import numpy as np
import pandas as pd

from src import config


def clean_values(df):
    """Убирает значения с флагами качества (переполнение, отрицательная площадь и т.д.).
    Строку не удаляем, а ставим NaN, чтобы было видно, что час был, но значение плохое."""
    df = df.copy()
    flagged = df["flag"].notna()
    df["value_clean"] = df["value"].astype("float64")
    df.loc[flagged, "value_clean"] = np.nan
    return df


def find_area_shift_dates(df):
    """Для каждой насосной скважины ищет день, когда dynamometer_area выросла в разы.
    Сравниваем суточную медиану с медианой апреля; первый день в мае,
    где медиана больше апрельской в 5 раз, считаем датой сдвига."""
    area = df[(df["parameter"] == "dynamometer_area") & df["flag"].isna()]
    shift_dates = {}
    for well, part in area.groupby("well_id", observed=True):
        daily = part.set_index("timestamp")["value"].astype("float64").resample("1D").median()
        april = daily["2026-04-01":"2026-04-30"].median()
        may = daily["2026-04-25":"2026-05-31"]
        jumped = may[may > 5 * april]
        if len(jumped) > 0:
            shift_dates[str(well)] = jumped.index[0]
    return shift_dates


def normalize_area(df):
    """Делит dynamometer_area на медиану своего периода (до и после сдвига),
    чтобы смена масштаба в мае не мешала анализу."""
    df = df.copy()
    shift_dates = find_area_shift_dates(df)
    df["area_period"] = pd.NA
    mask_area = df["parameter"] == "dynamometer_area"
    for well, shift_date in shift_dates.items():
        mask_well = mask_area & (df["well_id"] == well)
        before = mask_well & (df["timestamp"] < shift_date)
        after = mask_well & (df["timestamp"] >= shift_date)
        for mask, label in [(before, "before_shift"), (after, "after_shift")]:
            median = df.loc[mask, "value_clean"].median()
            if pd.notna(median) and median != 0:
                df.loc[mask, "value_clean"] = df.loc[mask, "value_clean"] / median
            df.loc[mask, "area_period"] = label
    return df, shift_dates


def to_wide(df, value_col="value_clean"):
    """Long -> wide: одна строка = одна скважина в один час, колонки = параметры."""
    wide = df.pivot_table(index=["well_id", "well_type", "timestamp"],
                          columns="parameter", values=value_col,
                          aggfunc="mean", observed=True)
    wide.columns.name = None
    return wide.reset_index()


def add_operating_state(wide):
    """Определяет состояние скважины в каждый час.
    Насосная: стоит, если SPM < 0.5. Фонтанная: стоит, если давление в НКТ
    и в линии меньше 0.5 атм. Если нужных значений нет, состояние неизвестно."""
    wide = wide.copy()
    states = []
    for row in wide.itertuples(index=False):
        if row.well_type == config.PUMP:
            spm = getattr(row, "SPM")
            if pd.isna(spm):
                states.append("unknown")
            elif spm < config.STOP_THRESHOLD:
                states.append("stopped")
            else:
                states.append("running")
        else:
            tubing = getattr(row, "tubing_pressure")
            line = getattr(row, "line_pressure")
            if pd.isna(tubing) and pd.isna(line):
                states.append("unknown")
            elif (pd.isna(tubing) or tubing < config.STOP_THRESHOLD) and \
                    (pd.isna(line) or line < config.STOP_THRESHOLD):
                states.append("stopped")
            else:
                states.append("running")
    wide["state"] = states
    return wide


def stop_events(wide):
    """Склеивает подряд идущие часы «stopped» в события остановки.
    Возвращает таблицу: скважина, начало, конец, длительность в часах."""
    events = []
    for well, part in wide.sort_values("timestamp").groupby("well_id", observed=True):
        current_start = None
        previous_time = None
        for ts, state in zip(part["timestamp"], part["state"]):
            is_stop = state == "stopped"
            # Разрыв больше часа тоже закрывает событие
            gap = previous_time is not None and (ts - previous_time) > pd.Timedelta(hours=1)
            if current_start is not None and (not is_stop or gap):
                events.append((well, current_start, previous_time))
                current_start = None
            if is_stop and current_start is None:
                current_start = ts
            previous_time = ts
        if current_start is not None:
            events.append((well, current_start, previous_time))
    table = pd.DataFrame(events, columns=["well_id", "start", "end"])
    table["hours"] = (table["end"] - table["start"]) / pd.Timedelta(hours=1) + 1
    return table


def mark_stopped(df):
    """Добавляет колонку stopped: True, если в этот час скважина стояла.
    Остановки анализируем отдельно, а режимы ищем только в часы работы,
    иначе каждая короткая остановка выглядела бы как смена режима."""
    df = df.copy()
    hourly = df.pivot_table(index=["well_id", "timestamp"], columns="parameter",
                            values="value_clean", aggfunc="mean", observed=True)
    spm = hourly.get("SPM")
    tubing = hourly.get("tubing_pressure")
    line = hourly.get("line_pressure")
    pump_stop = spm < config.STOP_THRESHOLD
    fountain_stop = (tubing.fillna(0) < config.STOP_THRESHOLD) & \
                    (line.fillna(0) < config.STOP_THRESHOLD) & \
                    (tubing.notna() | line.notna())
    stopped = (pump_stop.fillna(False) | fountain_stop).rename("stopped").reset_index()
    stopped["well_id"] = stopped["well_id"].astype(str)
    df["well_id_str"] = df["well_id"].astype(str)
    df = df.merge(stopped, left_on=["well_id_str", "timestamp"],
                  right_on=["well_id", "timestamp"], how="left", suffixes=("", "_s"))
    df = df.drop(columns=["well_id_s", "well_id_str"])
    df["stopped"] = df["stopped"].fillna(False).astype(bool)

    # Ноль веса штанги или ноль одного давления, когда скважина РАБОТАЕТ,
    # физически невозможен: это пропуск сигнала датчика, заменяем на NaN
    sensor_zero = (~df["stopped"]) & (df["value_clean"] == 0) & df["parameter"].isin(
        ["min_rod_weight", "max_rod_weight", "tubing_pressure", "casing_pressure", "line_pressure"])
    df.loc[sensor_zero, "value_clean"] = np.nan
    df.attrs["sensor_zero_removed"] = int(sensor_zero.sum())
    return df


def make_series(df, well, parameter, rule=config.RESAMPLE_RULE, start=None, end=None,
                value_col="value_clean"):
    """Ряд одного параметра одной скважины в часы работы: медиана за каждые 6 часов,
    пустые интервалы удаляются. dynamometer_area берём в log10, чтобы видеть сдвиг масштаба."""
    part = df[(df["well_id"] == well) & (df["parameter"] == parameter)]
    if "stopped" in part.columns:
        part = part[~part["stopped"]]
    series =part.set_index("timestamp")[value_col].astype("float64").sort_index()
    if parameter == "dynamometer_area":
        series = np.log10(series.where(series > 0))
    if start is not None:
        series = series[series.index >= pd.Timestamp(start)]
    if end is not None:
        series = series[series.index <= pd.Timestamp(end)]
    return series.resample(rule).median().dropna()
