"""Мини-дашборд Streamlit: выбор скважины, параметра, периода и метода.

Запуск (после python main.py --all):
    streamlit run app.py
"""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, detectors, pipeline, transform

st.set_page_config(page_title="Режимы работы скважин NK", layout="wide")


@st.cache_data
def get_data():
    """Загружает очищенные данные один раз и хранит в кэше."""
    return pipeline.load_clean()


@st.cache_data
def get_best_params():
    """Подобранные на train параметры методов."""
    with open(config.TABLES / "best_params.json", encoding="utf-8") as f:
        return json.load(f)["best_params"]


def read_table(name):
    """Читает таблицу результатов из results/tables, если она есть."""
    path = config.TABLES / name
    return pd.read_csv(path) if path.exists() else None


st.title("Анализ режимов работы нефтяных скважин NK")
st.caption("Тема 27. Данные: NK Oil Well Sensor Monitoring v2 (HuggingFace), январь - сентябрь 2026")

try:
    df = get_data()
    best_params = get_best_params()
except FileNotFoundError:
    st.error("Нет подготовленных данных. Сначала выполните: python main.py --all")
    st.stop()

# ----- Фильтры в боковой панели -----
wells = sorted(df["well_id"].astype(str).unique())
well = st.sidebar.selectbox("Скважина", wells, index=wells.index("NK-7"))
well_type = str(df.loc[df["well_id"] == well, "well_type"].iloc[0])
st.sidebar.write("Тип:", "фонтанная" if well_type == config.FOUNTAIN else "штанговый насос")
parameter = st.sidebar.selectbox("Параметр", config.PARAMS_BY_TYPE[well_type])
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()
period = st.sidebar.date_input("Период", (min_date, max_date), min_value=min_date, max_value=max_date)
method = st.sidebar.radio("Метод", list(detectors.METHODS),
                          format_func=lambda m: detectors.METHOD_TITLES[m], index=2)

if isinstance(period, tuple) and len(period) == 2:
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(hours=23)
else:
    st.warning("Выберите начало и конец периода.")
    st.stop()

tab_regimes, tab_stops, tab_types, tab_methods = st.tabs(
    ["Режимы", "Остановки", "Фонтанные vs насосные", "Сравнение методов"])

# ----- Вкладка 1: ряд и найденные режимы -----
with tab_regimes:
    series = transform.make_series(df, well, parameter, start=start, end=end)
    if len(series) < 20:
        st.info("Слишком мало данных за выбранный период.")
    else:
        params = best_params[method]
        found = detectors.detect(series.to_numpy(), method, params)
        parts = detectors.segments(series, found)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                                 name=parameter, line=dict(color="#555555", width=1)))
        for part in parts:
            fig.add_trace(go.Scatter(x=[part["start"], part["end"]], y=[part["mean"]] * 2,
                                     mode="lines", line=dict(color="#d62728", width=3),
                                     name=f"режим {part['regime']}", showlegend=False))
        for index in found:
            fig.add_vline(x=series.index[index], line_dash="dash", line_color="#d62728")
        y_title = "log10(площадь)" if parameter == "dynamometer_area" else parameter
        fig.update_layout(title=f"{well}: {parameter}, медиана за 6 часов в часы работы",
                          xaxis_title="Дата", yaxis_title=y_title, height=450,
                          template="plotly_white")
        st.plotly_chart(fig, width="stretch")

        col1, col2, col3 = st.columns(3)
        col1.metric("Найдено смен режима", len(found))
        col2.metric("Точек в ряду", len(series))
        col3.metric("Параметры метода", json.dumps(params))

        table = pd.DataFrame(parts)
        table["start"] = table["start"].dt.strftime("%Y-%m-%d %H:%M")
        table["end"] = table["end"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(table.round(3), width="stretch", hide_index=True)

# ----- Вкладка 2: остановки -----
with tab_stops:
    events = read_table("stop_events.csv")
    if events is None:
        st.info("Нет таблицы остановок: выполните пункт меню 7.")
    else:
        events["start"] = pd.to_datetime(events["start"])
        chosen = events[(events["well_id"] == well) & (events["start"] >= start) & (events["start"] <= end)]
        st.write(f"Остановок у {well} за период: {len(chosen)}, "
                 f"суммарно {chosen['hours'].sum():.0f} ч")
        if len(chosen):
            monthly = chosen.groupby(chosen["start"].dt.to_period("M").astype(str))["hours"].sum()
            st.bar_chart(monthly, x_label="Месяц", y_label="Часы остановки")
            st.dataframe(chosen.sort_values("hours", ascending=False), hide_index=True,
                         width="stretch")

# ----- Вкладка 3: сравнение типов скважин -----
with tab_types:
    summary = read_table("well_summary.csv")
    tests = read_table("well_type_tests.csv")
    if summary is not None:
        st.dataframe(summary, hide_index=True, width="stretch")
        st.image(str(config.FIGURES / "10_well_type_comparison.png"))
    if tests is not None:
        st.write("Средние по типам и тест Манна-Уитни (3 фонтанные и 7 насосных скважин):")
        st.dataframe(tests, hide_index=True, width="stretch")

# ----- Вкладка 4: сравнение методов -----
with tab_methods:
    comparison = read_table("methods_comparison.csv")
    real_events = read_table("real_events_summary.csv")
    if comparison is not None:
        st.write("Полусинтетическая проверка: параметры подобраны на train, проверка на test.")
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.image(str(config.FIGURES / "09_methods_comparison.png"))
    if real_events is not None:
        st.write("Реальные события из описания датасета:")
        st.dataframe(real_events, hide_index=True, width="stretch")
