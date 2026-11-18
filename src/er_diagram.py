"""Рисует ER-диаграмму базы данных (PNG) с помощью matplotlib."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src import config

# Таблица: (x, y верхнего края, список колонок). PK = первичный ключ, FK = внешний ключ
TABLES = {
    "well_types": (0.4, 9.6, ["PK type_id", "type_name", "description"]),
    "wells": (4.3, 9.6, ["PK well_id", "FK type_id"]),
    "parameters": (0.4, 6.6, ["PK parameter_id", "name", "unit", "FK type_id"]),
    "measurements": (4.3, 6.6, ["PK measurement_id", "FK well_id", "FK parameter_id",
                                "ts", "value", "n_obs", "split"]),
    "measurement_flags": (8.3, 6.6, ["PK,FK measurement_id", "PK,FK flag_id"]),
    "flags": (12.0, 6.6, ["PK flag_id", "flag_name", "rule"]),
    "detected_changes": (8.3, 9.6, ["PK change_id", "FK well_id", "FK parameter_id",
                                    "method", "change_ts", "mean_before", "mean_after"]),
    "experiments": (12.0, 9.6, ["PK experiment_id", "method", "split", "params",
                                "precision, recall, f1", "mean_delay_h", "time_ms, memory_kb"]),
}

# Связи: (откуда, куда, подпись)
LINKS = [
    ("wells", "well_types", "N : 1"),
    ("parameters", "well_types", "N : 1"),
    ("measurements", "wells", "N : 1"),
    ("measurements", "parameters", "N : 1"),
    ("measurement_flags", "measurements", "N : 1"),
    ("measurement_flags", "flags", "N : 1"),
    ("detected_changes", "wells", "N : 1"),
    ("detected_changes", "parameters", "N : 1"),
]

WIDTH = 3.2
ROW = 0.36


def _box(name):
    """Координаты прямоугольника таблицы: левый x, нижний y, ширина, высота."""
    x, top, columns = TABLES[name]
    height = ROW * (len(columns) + 1)
    return x, top - height, WIDTH, height


def draw_er_diagram():
    """Сохраняет results/figures/er_diagram.png и возвращает путь."""
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(3.6, 10.0)
    ax.axis("off")

    for source, target, label in LINKS:
        sx, sy, sw, sh = _box(source)
        tx, ty, tw, th = _box(target)
        s_top, t_top = TABLES[source][1], TABLES[target][1]
        if s_top == t_top:
            # Таблицы в одном ряду: стрелка горизонтально между краями
            y = s_top - ROW * 1.5
            start = (sx, y) if tx < sx else (sx + sw, y)
            end = (tx + tw, y) if tx < sx else (tx, y)
            points = [start, end]
        elif abs(sx - tx) < 0.01:
            # Таблицы одна под другой: стрелка вертикально
            x = sx + sw / 2
            points = [(x, s_top), (x, ty)]
        else:
            # Обход сверху: вверх, влево до края, вниз к таблице
            x_start = sx + sw / 2
            points = [(x_start, s_top), (x_start, 9.85), (0.12, 9.85),
                      (0.12, t_top - ROW * 1.5), (tx, t_top - ROW * 1.5)]
        xs, ys = zip(*points)
        ax.plot(xs[:-1] + (xs[-1],), ys, color="#777777", lw=1.2)
        ax.annotate("", xy=points[-1], xytext=points[-2],
                    arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.2))
        mid = len(points) // 2
        lx = (points[mid - 1][0] + points[mid][0]) / 2
        ly = (points[mid - 1][1] + points[mid][1]) / 2
        ax.text(lx, ly, label, fontsize=8, color="#444444", ha="center", va="center",
                backgroundcolor="white")

    for name, (x, top, columns) in TABLES.items():
        bx, by, bw, bh = _box(name)
        results_table = name in ["detected_changes", "experiments"]
        ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.02",
                                    fc="#fff7ea" if results_table else "#eef5fb",
                                    ec="#333333", lw=1.2))
        ax.add_patch(FancyBboxPatch((bx, top - ROW), bw, ROW, boxstyle="round,pad=0.02",
                                    fc="#f4b183" if results_table else "#8fbcd4", ec="#333333", lw=1.2))
        ax.text(bx + bw / 2, top - ROW / 2, name, ha="center", va="center",
                fontsize=10, fontweight="bold")
        for i, column in enumerate(columns):
            ax.text(bx + 0.12, top - ROW * (i + 1.5), column, ha="left", va="center", fontsize=8.5)

    ax.set_title("ER-диаграмма базы wells.db (3НФ). Синие: исходные данные, оранжевые: результаты анализа",
                 fontsize=11)
    config.ensure_dirs()
    path = config.FIGURES / "er_diagram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
