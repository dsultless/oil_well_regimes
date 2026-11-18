"""База данных SQLite: нормализованная схема (3НФ), загрузка данных,
аналитические SQL-запросы и сравнение одного расчёта в SQL и в Pandas."""

import sqlite3
import time

import numpy as np
import pandas as pd

from src import config

SCHEMA = """
CREATE TABLE well_types (
    type_id     INTEGER PRIMARY KEY,
    type_name   TEXT NOT NULL UNIQUE,
    description TEXT
);
CREATE TABLE wells (
    well_id TEXT PRIMARY KEY,
    type_id INTEGER NOT NULL REFERENCES well_types(type_id)
);
CREATE TABLE parameters (
    parameter_id INTEGER PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    unit         TEXT,
    type_id      INTEGER NOT NULL REFERENCES well_types(type_id)
);
CREATE TABLE measurements (
    measurement_id INTEGER PRIMARY KEY,
    well_id        TEXT NOT NULL REFERENCES wells(well_id),
    parameter_id   INTEGER NOT NULL REFERENCES parameters(parameter_id),
    ts             TEXT NOT NULL,
    value          REAL NOT NULL,
    n_obs          INTEGER NOT NULL,
    split          TEXT NOT NULL CHECK (split IN ('train', 'test')),
    UNIQUE (well_id, parameter_id, ts)
);
CREATE TABLE flags (
    flag_id   INTEGER PRIMARY KEY,
    flag_name TEXT NOT NULL UNIQUE,
    rule      TEXT
);
CREATE TABLE measurement_flags (
    measurement_id INTEGER NOT NULL REFERENCES measurements(measurement_id),
    flag_id        INTEGER NOT NULL REFERENCES flags(flag_id),
    PRIMARY KEY (measurement_id, flag_id)
);
CREATE TABLE detected_changes (
    change_id    INTEGER PRIMARY KEY,
    well_id      TEXT NOT NULL REFERENCES wells(well_id),
    parameter_id INTEGER NOT NULL REFERENCES parameters(parameter_id),
    method       TEXT NOT NULL,
    change_ts    TEXT NOT NULL,
    mean_before  REAL,
    mean_after   REAL
);
CREATE TABLE experiments (
    experiment_id INTEGER PRIMARY KEY,
    method        TEXT NOT NULL,
    split         TEXT NOT NULL,
    params        TEXT,
    precision     REAL,
    recall        REAL,
    f1            REAL,
    mean_delay_h  REAL,
    time_ms       REAL,
    memory_kb     REAL
);
CREATE INDEX idx_meas_well_param_ts ON measurements(well_id, parameter_id, ts);
"""

FLAG_RULES = {
    "weight_overflow": "min/max rod weight >= 30000 (переполнение 16-битного датчика)",
    "area_negative": "dynamometer_area < 0",
    "area_extreme": "|dynamometer_area| > 1e6",
    "spm_outlier": "SPM > 9",
}

TYPE_DESCRIPTIONS = {
    config.FOUNTAIN: "Фонтанная скважина: нефть поднимается за счёт пластового давления",
    config.PUMP: "Скважина со штанговым глубинным насосом (станок-качалка)",
}


def connect():
    """Открывает соединение с базой и включает проверку внешних ключей."""
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def build_database(train_df, test_df):
    """Создаёт базу с нуля и загружает данные в нормализованные таблицы."""
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    conn = connect()
    conn.executescript(SCHEMA)

    data = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")],
                     ignore_index=True)

    # Справочник типов скважин
    type_ids = {}
    for i, type_name in enumerate(sorted(data["well_type"].astype(str).unique()), start=1):
        type_ids[type_name] = i
        conn.execute("INSERT INTO well_types VALUES (?, ?, ?)",
                     (i, type_name, TYPE_DESCRIPTIONS.get(type_name)))

    # Скважины
    wells = data[["well_id", "well_type"]].astype(str).drop_duplicates()
    conn.executemany("INSERT INTO wells VALUES (?, ?)",
                     [(w, type_ids[t]) for w, t in wells.itertuples(index=False)])

    # Параметры с единицами измерения
    params = data[["parameter", "unit", "well_type"]].astype(str).drop_duplicates("parameter")
    param_ids = {}
    for i, (name, unit, type_name) in enumerate(params.itertuples(index=False), start=1):
        param_ids[name] = i
        conn.execute("INSERT INTO parameters VALUES (?, ?, ?, ?)",
                     (i, name, unit, type_ids[type_name]))

    # Справочник флагов
    flag_ids = {}
    for i, (name, rule) in enumerate(FLAG_RULES.items(), start=1):
        flag_ids[name] = i
        conn.execute("INSERT INTO flags VALUES (?, ?, ?)", (i, name, rule))

    # Измерения (основная таблица)
    data = data.reset_index(drop=True)
    data["measurement_id"] = data.index + 1
    rows = zip(data["measurement_id"].tolist(), data["well_id"].astype(str).tolist(),
               data["parameter"].astype(str).map(param_ids).tolist(),
               data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
               data["value"].astype(float).tolist(), data["n_obs"].astype(int).tolist(),
               data["split"].tolist())
    conn.executemany("INSERT INTO measurements VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    # Связь «многие-ко-многим»: у одного измерения может быть несколько флагов
    links = []
    flagged = data[data["flag"].notna()]
    for measurement_id, flag_text in zip(flagged["measurement_id"], flagged["flag"].astype(str)):
        for flag_name in flag_text.split(";"):
            links.append((int(measurement_id), flag_ids[flag_name.strip()]))
    conn.executemany("INSERT INTO measurement_flags VALUES (?, ?)", links)

    conn.commit()
    counts = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
              for table in ["well_types", "wells", "parameters", "measurements",
                            "flags", "measurement_flags"]}
    conn.close()
    return counts


# Аналитические запросы: название -> текст SQL
QUERIES = {
    "Q1. Объём данных по типам скважин (JOIN + GROUP BY)": """
        SELECT wt.type_name, w.well_id,
               COUNT(*)              AS rows_count,
               ROUND(AVG(m.n_obs), 2) AS avg_raw_records_per_hour,
               MIN(m.ts)             AS first_ts,
               MAX(m.ts)             AS last_ts
        FROM measurements m
        JOIN wells w       ON w.well_id = m.well_id
        JOIN well_types wt ON wt.type_id = w.type_id
        GROUP BY wt.type_name, w.well_id
        ORDER BY wt.type_name, rows_count DESC;
    """,
    "Q2. Флаги качества по скважинам (связь многие-ко-многим + HAVING)": """
        SELECT m.well_id, f.flag_name, COUNT(*) AS flagged_rows
        FROM measurement_flags mf
        JOIN measurements m ON m.measurement_id = mf.measurement_id
        JOIN flags f        ON f.flag_id = mf.flag_id
        GROUP BY m.well_id, f.flag_name
        HAVING COUNT(*) >= 10
        ORDER BY flagged_rows DESC;
    """,
    "Q3. Разрывы в данных больше 24 часов (CTE + LAG)": """
        WITH hours AS (
            SELECT DISTINCT well_id, ts FROM measurements
        ),
        with_prev AS (
            SELECT well_id, ts,
                   LAG(ts) OVER (PARTITION BY well_id ORDER BY ts) AS prev_ts
            FROM hours
        )
        SELECT well_id, prev_ts AS gap_start, ts AS gap_end,
               ROUND((julianday(ts) - julianday(prev_ts)) * 24, 0) AS gap_hours
        FROM with_prev
        WHERE prev_ts IS NOT NULL
          AND (julianday(ts) - julianday(prev_ts)) * 24 > 24
        ORDER BY gap_hours DESC;
    """,
    "Q4. Среднемесячные значения параметров (GROUP BY по месяцу)": """
        SELECT m.well_id, p.name AS parameter, substr(m.ts, 1, 7) AS month,
               ROUND(AVG(m.value), 2) AS avg_value, COUNT(*) AS hours
        FROM measurements m
        JOIN parameters p ON p.parameter_id = m.parameter_id
        WHERE m.measurement_id NOT IN (SELECT measurement_id FROM measurement_flags)
          AND p.name IN ('SPM', 'tubing_pressure')
        GROUP BY m.well_id, p.name, month
        ORDER BY m.well_id, month;
    """,
    "Q5. Рейтинг насосных скважин по заполнению насоса за месяц (RANK)": """
        WITH monthly AS (
            SELECT m.well_id, substr(m.ts, 1, 7) AS month, AVG(m.value) AS avg_fillage
            FROM measurements m
            JOIN parameters p ON p.parameter_id = m.parameter_id
            WHERE p.name = 'pump_fillage'
            GROUP BY m.well_id, month
        )
        SELECT month, well_id, ROUND(avg_fillage, 1) AS avg_fillage,
               RANK() OVER (PARTITION BY month ORDER BY avg_fillage DESC) AS rank_in_month
        FROM monthly
        ORDER BY month, rank_in_month;
    """,
    "Q6. Скользящее среднее SPM за 24 часа, скважина NK-7 (оконная функция)": """
        SELECT m.ts, ROUND(m.value, 2) AS spm,
               ROUND(AVG(m.value) OVER (ORDER BY m.ts
                     ROWS BETWEEN 23 PRECEDING AND CURRENT ROW), 2) AS spm_rolling_24h
        FROM measurements m
        JOIN parameters p ON p.parameter_id = m.parameter_id
        WHERE m.well_id = 'NK-7' AND p.name = 'SPM'
          AND m.ts BETWEEN '2026-05-25' AND '2026-06-20'
        ORDER BY m.ts;
    """,
    "Q7. Часы остановки насосных скважин по месяцам (SPM < 0.5, HAVING)": """
        SELECT m.well_id, substr(m.ts, 1, 7) AS month,
               SUM(CASE WHEN m.value < 0.5 THEN 1 ELSE 0 END) AS stopped_hours,
               COUNT(*) AS total_hours,
               ROUND(100.0 * SUM(CASE WHEN m.value < 0.5 THEN 1 ELSE 0 END) / COUNT(*), 1)
                   AS stopped_percent
        FROM measurements m
        JOIN parameters p ON p.parameter_id = m.parameter_id
        WHERE p.name = 'SPM'
        GROUP BY m.well_id, month
        HAVING stopped_hours > 0
        ORDER BY stopped_hours DESC;
    """,
    "Q8. Фонтанные скважины с давлением в НКТ выше среднего по типу (подзапрос)": """
        SELECT m.well_id, ROUND(AVG(m.value), 2) AS avg_tubing_pressure,
               (SELECT ROUND(AVG(m2.value), 2)
                FROM measurements m2
                JOIN parameters p2 ON p2.parameter_id = m2.parameter_id
                WHERE p2.name = 'tubing_pressure') AS fountain_average
        FROM measurements m
        JOIN parameters p ON p.parameter_id = m.parameter_id
        WHERE p.name = 'tubing_pressure'
        GROUP BY m.well_id
        HAVING AVG(m.value) > (SELECT AVG(m3.value)
                               FROM measurements m3
                               JOIN parameters p3 ON p3.parameter_id = m3.parameter_id
                               WHERE p3.name = 'tubing_pressure');
    """,
    "Q9. Изменение dynamometer_area месяц к месяцу (CTE + LAG): сдвиг масштаба в мае": """
        WITH monthly AS (
            SELECT m.well_id, substr(m.ts, 1, 7) AS month, AVG(m.value) AS avg_area
            FROM measurements m
            JOIN parameters p ON p.parameter_id = m.parameter_id
            WHERE p.name = 'dynamometer_area'
              AND m.measurement_id NOT IN (SELECT measurement_id FROM measurement_flags)
            GROUP BY m.well_id, month
        )
        SELECT well_id, month, ROUND(avg_area, 0) AS avg_area,
               ROUND(avg_area / LAG(avg_area) OVER (PARTITION BY well_id ORDER BY month), 1)
                   AS ratio_to_prev_month
        FROM monthly
        WHERE month BETWEEN '2026-04' AND '2026-06'
        ORDER BY well_id, month;
    """,
    "Q10. Суточный профиль по часам для двух типов скважин": """
        SELECT wt.type_name, p.name AS parameter,
               CAST(substr(m.ts, 12, 2) AS INTEGER) AS hour_of_day,
               ROUND(AVG(m.value), 2) AS avg_value
        FROM measurements m
        JOIN wells w       ON w.well_id = m.well_id
        JOIN well_types wt ON wt.type_id = w.type_id
        JOIN parameters p  ON p.parameter_id = m.parameter_id
        WHERE p.name IN ('pump_fillage', 'tubing_pressure')
        GROUP BY wt.type_name, p.name, hour_of_day
        ORDER BY wt.type_name, hour_of_day;
    """,
    "Q11. Найденные смены режима по методам и типам скважин (результаты анализа)": """
        SELECT dc.method, wt.type_name, COUNT(*) AS changes,
               COUNT(DISTINCT dc.well_id) AS wells_with_changes,
               ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT dc.well_id || p.name), 2)
                   AS changes_per_series
        FROM detected_changes dc
        JOIN wells w       ON w.well_id = dc.well_id
        JOIN well_types wt ON wt.type_id = w.type_id
        JOIN parameters p  ON p.parameter_id = dc.parameter_id
        GROUP BY dc.method, wt.type_name
        ORDER BY dc.method, wt.type_name;
    """,
    "Q12. Лучший метод на каждой выборке по F1 (ROW_NUMBER)": """
        WITH ranked AS (
            SELECT split, method, f1, precision, recall, time_ms,
                   ROW_NUMBER() OVER (PARTITION BY split ORDER BY f1 DESC) AS place
            FROM experiments
        )
        SELECT split, place, method, ROUND(f1, 3) AS f1, ROUND(precision, 3) AS precision,
               ROUND(recall, 3) AS recall, ROUND(time_ms, 2) AS time_ms
        FROM ranked
        ORDER BY split, place;
    """,
}


def run_query(sql, params=None):
    """Выполняет SQL-запрос и возвращает результат как таблицу Pandas."""
    conn = connect()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def run_all_queries(save=True):
    """Выполняет все запросы из QUERIES и сохраняет результаты в CSV."""
    results = {}
    for number, (title, sql) in enumerate(QUERIES.items(), start=1):
        try:
            table = run_query(sql)
        except Exception as error:
            print(f"Запрос {title} не выполнился: {error}")
            continue
        results[title] = table
        if save:
            table.to_csv(config.TABLES / f"sql_q{number:02d}.csv", index=False)
    return results


def save_detected_changes(changes):
    """Записывает найденные точки смены режима в таблицу detected_changes."""
    conn = connect()
    conn.execute("DELETE FROM detected_changes")
    param_ids = dict(conn.execute("SELECT name, parameter_id FROM parameters").fetchall())
    rows = [(c["well_id"], param_ids[c["parameter"]], c["method"], str(c["change_ts"]),
             c["mean_before"], c["mean_after"]) for c in changes]
    conn.executemany("""INSERT INTO detected_changes
                        (well_id, parameter_id, method, change_ts, mean_before, mean_after)
                        VALUES (?, ?, ?, ?, ?, ?)""", rows)
    conn.commit()
    conn.close()


def save_experiments(rows):
    """Записывает результаты сравнения методов в таблицу experiments."""
    conn = connect()
    conn.execute("DELETE FROM experiments")
    conn.executemany("""INSERT INTO experiments
                        (method, split, params, precision, recall, f1, mean_delay_h,
                         time_ms, memory_kb) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     [(r["method"], r["split"], r["params"], r["precision"], r["recall"],
                       r["f1"], r["mean_delay_h"], r["time_ms"], r["memory_kb"]) for r in rows])
    conn.commit()
    conn.close()


def pandas_vs_sql(df, repeats=5):
    """Один и тот же расчёт (среднее по скважине, параметру и месяцу)
    делаем в SQL и в Pandas, сравниваем время и проверяем, что ответы совпадают."""
    sql = """
        SELECT m.well_id, p.name AS parameter, substr(m.ts, 1, 7) AS month,
               AVG(m.value) AS avg_value
        FROM measurements m
        JOIN parameters p ON p.parameter_id = m.parameter_id
        GROUP BY m.well_id, p.name, month
        ORDER BY m.well_id, p.name, month;
    """

    def with_sql():
        return run_query(sql)

    def with_pandas():
        temp = df[["well_id", "parameter", "timestamp", "value"]].copy()
        temp["month"] = temp["timestamp"].dt.strftime("%Y-%m")
        result = (temp.groupby(["well_id", "parameter", "month"], observed=True)["value"]
                      .mean().rename("avg_value").reset_index())
        result["well_id"] = result["well_id"].astype(str)
        result["parameter"] = result["parameter"].astype(str)
        return result.sort_values(["well_id", "parameter", "month"]).reset_index(drop=True)

    timings = {}
    outputs = {}
    for name, func in [("sql", with_sql), ("pandas", with_pandas)]:
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            outputs[name] = func()
            times.append(time.perf_counter() - start)
        timings[name] = min(times)

    left = outputs["sql"].sort_values(["well_id", "parameter", "month"]).reset_index(drop=True)
    right = outputs["pandas"]
    same_keys = left[["well_id", "parameter", "month"]].equals(right[["well_id", "parameter", "month"]])
    # float32 в Pandas даёт небольшую погрешность, поэтому сравниваем с допуском
    same_values = bool(np.allclose(left["avg_value"], right["avg_value"], rtol=1e-4))
    return {
        "groups": int(len(left)),
        "sql_seconds": round(timings["sql"], 4),
        "pandas_seconds": round(timings["pandas"], 4),
        "same_groups": bool(same_keys),
        "same_values": same_values,
        "max_abs_difference": float(np.max(np.abs(left["avg_value"] - right["avg_value"]))),
    }
