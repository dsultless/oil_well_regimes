"""Графики: статичные PNG (matplotlib, seaborn) для отчёта и презентации
и интерактивный HTML (plotly)."""

import matplotlib

matplotlib.use("Agg")  # рисуем в файл, без окна
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config, detectors, transform

sns.set_theme(style="whitegrid")
METHOD_COLORS = {"threshold_baseline": "#9e9e9e", "rolling_window": "#1f77b4",
                 "binary_segmentation": "#d62728"}


def _save(fig, name):
    """Сохраняет рисунок в results/figures и закрывает его."""
    config.ensure_dirs()
    path = config.FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_completeness(month_table):
    """Тепловая карта: доля часов с данными по скважинам и месяцам."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(month_table, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=100,
                cbar_kws={"label": "% часов с данными"}, ax=ax)
    ax.set_title("Полнота данных по скважинам и месяцам, %")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Скважина")
    return _save(fig, "01_completeness_heatmap.png")


def plot_flags(flags_table):
    """Столбцы: число строк с флагами качества по скважинам."""
    data = flags_table.drop(columns="total")
    fig, ax = plt.subplots(figsize=(8, 4))
    data.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_title("Флаги качества по скважинам")
    ax.set_xlabel("Скважина")
    ax.set_ylabel("Число строк")
    ax.legend(title="Флаг")
    return _save(fig, "02_quality_flags.png")


def plot_area_shift(df_clean, df_norm, shift_dates):
    """dynamometer_area до и после нормировки: виден сдвиг масштаба в мае."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for well in ["NK-68", "NK-77", "NK-91", "NK-93"]:
        raw = df_clean[(df_clean["well_id"] == well) & (df_clean["parameter"] == "dynamometer_area")]
        raw = raw.set_index("timestamp")["value_clean"].resample("1D").median()
        norm = df_norm[(df_norm["well_id"] == well) & (df_norm["parameter"] == "dynamometer_area")]
        norm = norm.set_index("timestamp")["value_clean"].resample("1D").median()
        axes[0].plot(raw.index, raw.values, label=well)
        axes[1].plot(norm.index, norm.values, label=well)
    axes[0].set_yscale("log")
    axes[0].set_title("dynamometer_area: исходные значения (лог. шкала)")
    axes[1].set_title("После деления на медиану своего периода")
    for ax in axes:
        ax.set_xlabel("Дата")
        ax.legend()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
    axes[0].set_ylabel("Площадь динамограммы")
    axes[1].set_ylabel("Значение / медиана периода")
    return _save(fig, "03_area_scale_shift.png")


def plot_boxplots(wide):
    """Boxplot основного параметра по скважинам в часы работы."""
    running = wide[wide["state"] == "running"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pump = running[running["well_type"] == config.PUMP]
    sns.boxplot(data=pump, x="well_id", y="pump_fillage", ax=axes[0], color="#8fbcd4")
    axes[0].set_title("Насосные скважины: заполнение насоса, %")
    axes[0].set_xlabel("Скважина")
    axes[0].set_ylabel("pump_fillage, %")
    fountain = running[running["well_type"] == config.FOUNTAIN]
    sns.boxplot(data=fountain, x="well_id", y="tubing_pressure", ax=axes[1], color="#f4b183")
    axes[1].set_title("Фонтанные скважины: давление в НКТ, атм")
    axes[1].set_xlabel("Скважина")
    axes[1].set_ylabel("tubing_pressure, атм")
    return _save(fig, "04_boxplots_by_well.png")


def plot_hourly(profile):
    """Суточные паттерны: значение в каждый час суток относительно среднего скважины."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, well_type, title in [(axes[0], config.PUMP, "Насосные скважины"),
                                 (axes[1], config.FOUNTAIN, "Фонтанные скважины")]:
        part = profile[profile["well_type"] == well_type]
        for parameter, values in part.groupby("parameter"):
            ax.plot(values["hour"], values["relative_value"], marker="o", ms=3, label=parameter)
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.set_title(f"{title}: суточный профиль")
        ax.set_xlabel("Час суток")
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Значение / среднее скважины")
    return _save(fig, "05_hourly_profiles.png")


def plot_correlations(matrices):
    """Корреляционные матрицы параметров двух типов скважин."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    titles = {config.PUMP: "Насосные скважины", config.FOUNTAIN: "Фонтанные скважины"}
    for ax, (well_type, matrix) in zip(axes, matrices.items()):
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Корреляции: {titles[well_type]}")
    return _save(fig, "06_correlations.png")


def plot_detection(df, well, parameter, best_params, filename, title):
    """Ряд параметра и точки смены режима, найденные тремя методами."""
    series = transform.make_series(df, well, parameter)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    for ax, (method, params) in zip(axes, best_params.items()):
        found = detectors.detect(series.to_numpy(), method, params)
        ax.plot(series.index, series.values, color="#555555", lw=0.8)
        for part in detectors.segments(series, found):
            ax.hlines(part["mean"], part["start"], part["end"], colors=METHOD_COLORS[method], lw=2.5)
        for index in found:
            ax.axvline(series.index[index], color=METHOD_COLORS[method], ls="--", lw=1)
        ax.set_title(f"{detectors.METHOD_TITLES[method]}: найдено смен режима {len(found)}", fontsize=10)
        ax.set_ylabel(parameter)
    axes[-1].set_xlabel("Дата")
    fig.suptitle(title)
    return _save(fig, filename)


def plot_methods(final):
    """Сравнение методов на test: precision, recall, F1 и время."""
    test = final[final["split"] == "test"].set_index("method")
    test.index = [detectors.METHOD_TITLES[m] for m in test.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    test[["precision", "recall", "f1"]].plot(kind="bar", ax=axes[0], rot=0,
                                             color=["#8fbcd4", "#f4b183", "#2e7d32"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Качество на test (полусинтетическая проверка)")
    axes[0].set_ylabel("Значение метрики")
    test["time_ms"].plot(kind="bar", ax=axes[1], rot=0, color="#7e57c2")
    axes[1].set_title("Время на один ряд, мс")
    axes[1].set_ylabel("мс")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    return _save(fig, "09_methods_comparison.png")


def plot_well_types(per_well):
    """Доля времени остановки и частота смен режима по скважинам."""
    data = per_well.sort_values(["well_type", "well_id"])
    colors = ["#f4b183" if t == config.FOUNTAIN else "#8fbcd4" for t in data["well_type"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(data["well_id"], data["stopped_percent"], color=colors)
    axes[0].set_title("Доля часов остановки, %")
    axes[0].set_ylabel("%")
    axes[1].bar(data["well_id"], data["changes_per_30_days"], color=colors)
    axes[1].set_title("Смен режима за 30 дней (бинарная сегментация)")
    axes[1].set_ylabel("Смен на ряд")
    for ax in axes:
        ax.set_xlabel("Скважина (оранжевые: фонтанные, синие: насосные)")
    return _save(fig, "10_well_type_comparison.png")


def interactive_overview(df, regimes, best_method="binary_segmentation"):
    """Интерактивный HTML: основной параметр каждой скважины и найденные режимы."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Библиотека plotly не установлена: pip install -r requirements.txt")
        return None

    fig = go.Figure()
    buttons = []
    wells = df[["well_id", "well_type"]].drop_duplicates().astype(str).sort_values("well_id")
    traces_per_well = []
    for well, well_type in wells.itertuples(index=False):
        parameter = "SPM" if well_type == config.PUMP else "tubing_pressure"
        series = transform.make_series(df, well, parameter)
        parts = regimes[(regimes["well_id"] == well) & (regimes["parameter"] == parameter) &
                        (regimes["method"] == best_method)]
        count = 0
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                                 name=f"{well} {parameter}", line=dict(color="#555555", width=1),
                                 visible=len(traces_per_well) == 0))
        count += 1
        for part in parts.itertuples():
            fig.add_trace(go.Scatter(x=[part.start, part.end], y=[part.mean, part.mean],
                                     mode="lines", line=dict(color="#d62728", width=3),
                                     name=f"режим {part.regime}", showlegend=False,
                                     visible=len(traces_per_well) == 0))
            count += 1
        traces_per_well.append((well, parameter, count))

    total = sum(c for _, _, c in traces_per_well)
    position = 0
    for well, parameter, count in traces_per_well:
        visible = [False] * total
        visible[position:position + count] = [True] * count
        buttons.append(dict(label=well, method="update",
                            args=[{"visible": visible},
                                  {"title": f"{well}: {parameter} и режимы (бинарная сегментация)"}]))
        position += count
    first_well, first_param, _ = traces_per_well[0]
    fig.update_layout(title=f"{first_well}: {first_param} и режимы (бинарная сегментация)",
                      xaxis_title="Дата", yaxis_title="Значение (медиана за 6 часов)",
                      updatemenus=[dict(buttons=buttons, direction="down", x=1.0, y=1.15)],
                      template="plotly_white", height=500)
    path = config.FIGURES / "interactive_regimes.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path
