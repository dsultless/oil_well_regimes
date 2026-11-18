"""Общие настройки проекта: пути к папкам, ссылки на данные, списки параметров."""

from pathlib import Path

# Корневая папка проекта (папка, в которой лежит main.py)
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DB_PATH = ROOT / "database" / "wells.db"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

# Датасет преподавателя на HuggingFace (лицензия Apache 2.0)
HF_BASE = "https://huggingface.co/datasets/Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2/resolve/main"
RAW_FILES = ["wells_dataset.csv", "train.csv", "test.csv"]

# Типы скважин
FOUNTAIN = "fountain"
PUMP = "sucker_rod_pump"

# Основные параметры каждого типа скважин
PARAMS_BY_TYPE = {
    FOUNTAIN: ["tubing_pressure", "casing_pressure", "line_pressure"],
    PUMP: ["SPM", "pump_fillage", "max_rod_weight", "min_rod_weight", "dynamometer_area"],
}

# Параметры, по которым ищем смену режима (dynamometer_area берём отдельно, в логарифме)
DETECTION_PARAMS = {
    FOUNTAIN: ["tubing_pressure", "casing_pressure", "line_pressure"],
    PUMP: ["SPM", "pump_fillage", "max_rod_weight", "min_rod_weight"],
}

# Шаг агрегации для поиска смены режима: медиана за 6 часов
RESAMPLE_RULE = "6h"
POINTS_PER_DAY = 4

# Порог «скважина стоит»: SPM меньше 0.5 качания в минуту или давление меньше 0.5 атм
STOP_THRESHOLD = 0.5

# Дата смены масштаба dynamometer_area (из описания датасета: май 2026)
AREA_SHIFT_MONTH = "2026-05"

# Зерно генератора случайных чисел, чтобы результаты повторялись
SEED = 42


def ensure_dirs():
    """Создаёт все рабочие папки, если их ещё нет."""
    for folder in [DATA_RAW, DATA_PROCESSED, DB_PATH.parent, FIGURES, TABLES]:
        folder.mkdir(parents=True, exist_ok=True)
